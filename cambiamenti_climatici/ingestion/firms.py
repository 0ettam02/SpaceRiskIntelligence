"""Resumable FIRMS acquisition. Standard library only; never invent missing observations."""
import argparse
import csv
import hashlib
from http.client import HTTPException
import io
import json
import os
import re
from pathlib import Path
import time
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

BASE = "https://firms.modaps.eosdis.nasa.gov/api"
REQUIRED = {"latitude", "longitude", "acq_date", "acq_time", "frp"}


def days(start, end):
    while start <= end:
        yield start
        start += timedelta(days=1)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def request(url, attempts=4, sleep=time.sleep):
    """No exception includes the URL: FIRMS puts the credential in its path."""
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8-sig")
        except HTTPError as exc:
            # FIRMS reports its transaction quota as HTTP 400 rather than 429.
            body = exc.read(1024) if exc.code == 400 else b""
            quota = b"exceeding allowed transaction limit" in body.lower()
            retryable = quota or exc.code == 429 or exc.code >= 500
            reason = "quota_exceeded" if quota else f"HTTP {exc.code}"
            retry_after = "60" if quota else (exc.headers.get("Retry-After", "") if exc.headers else "")
        except (URLError, TimeoutError, OSError, HTTPException, UnicodeDecodeError):
            retryable, reason, retry_after = True, "network_error", ""
        if not retryable or attempt == attempts - 1:
            raise RuntimeError(reason) from None
        delay = min(60, int(retry_after)) if retry_after.isdigit() else min(60, 2 ** (attempt + 1))
        sleep(delay)


def parse_detections(payload, start, end):
    reader = csv.DictReader(io.StringIO(payload))
    fields = reader.fieldnames or []
    # A blank body or a 200 response containing an error is NOT evidence of no detections.
    if not REQUIRED.issubset(fields):
        raise ValueError("invalid_detection_csv")
    rows, seen = [], set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("malformed_detection_row")
        observed = date.fromisoformat(row["acq_date"])
        if not start <= observed <= end:
            raise ValueError("observation_outside_requested_dates")
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("invalid_coordinates")
        identity = tuple(row[field] for field in fields)
        if identity not in seen:
            seen.add(identity)
            rows.append(row)
    return fields, rows


def gap_dates(segments, start, end):
    """Legacy segments identify absent dates, not sensor/cell observation coverage."""
    covered = set()
    with Path(segments).open() as stream:
        for row in csv.DictReader(stream):
            covered.update(days(date.fromisoformat(row["start"]), date.fromisoformat(row["end"])))
    return set(days(start, end)) - covered


def blocks(dates, size=5):
    block = []
    for day in sorted(dates):
        if block and (len(block) == size or day != block[-1] + timedelta(days=1)):
            yield block
            block = []
        block.append(day)
    if block:
        yield block


def acquire(key, source, start, end, output, area="world", selected=None, fetch=request):
    if start > end:
        raise ValueError("start must precede end")
    if not source.endswith(("_SP", "_NRT")) or not source.startswith(("MODIS_", "VIIRS_")):
        raise ValueError("Choose one MODIS or VIIRS active-fire product explicitly")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    config = {"source": source, "area": area}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"config": config, "days": {}}
    if manifest["config"] != config:
        raise ValueError("Output directory belongs to another source/area")
    requested = set(days(start, end)) if selected is None else set(selected) & set(days(start, end))
    availability = list(csv.DictReader(io.StringIO(fetch(f"{BASE}/data_availability/csv/{quote(key, safe='')}/{source}"))))
    matching = [row for row in availability if row.get("data_id") == source]
    if len(matching) != 1:
        raise ValueError("invalid_availability_response")
    lower, upper = (date.fromisoformat(matching[0][name]) for name in ("min_date", "max_date"))
    atomic_text(output / "availability.json", json.dumps(matching[0], indent=2))
    pending = []
    for day in sorted(requested):
        stamp = day.isoformat()
        entry = manifest["days"].get(stamp, {})
        path = output / "daily" / f"{stamp}.csv"
        if entry.get("status") in {"downloaded", "empty"} and path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() == entry.get("sha256"):
                continue
        if not lower <= day <= upper:
            manifest["days"][stamp] = {"status": "unavailable", "count": None}
        else:
            pending.append(day)
    atomic_text(manifest_path, json.dumps(manifest, indent=2))
    for block in blocks(pending):
        first, last = block[0], block[-1]
        url = f"{BASE}/area/csv/{quote(key, safe='')}/{source}/{quote(area, safe=',-')}/{len(block)}/{first}"
        try:
            fields, rows = parse_detections(fetch(url), first, last)
            by_day = {day.isoformat(): [] for day in block}
            for row in rows:
                by_day[row["acq_date"]].append(row)
            for stamp, records in by_day.items():
                path = output / "daily" / f"{stamp}.csv"
                atomic_text(path, csv_text(records, fields))
                manifest["days"][stamp] = {
                    "status": "downloaded" if records else "empty", "count": len(records),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            print(f"{source} {first}..{last}: {len(rows)} records", flush=True)
        except (RuntimeError, ValueError) as exc:
            # Do not retain arbitrary response/error text that could contain credentials.
            reason = str(exc) if re.fullmatch(r"HTTP \d{3}|network_error|quota_exceeded", str(exc)) else type(exc).__name__
            for day in block:
                manifest["days"][day.isoformat()] = {"status": "failed", "count": None, "error": reason}
            print(f"{source} {first}..{last}: failed ({reason})", flush=True)
        atomic_text(manifest_path, json.dumps(manifest, indent=2))
    coverage = [{"date": day.isoformat(), "source": source, **{
        name: manifest["days"][day.isoformat()][name] for name in ("status", "count")
    }} for day in sorted(requested)]
    atomic_text(output / "coverage.csv", csv_text(coverage, ["date", "source", "status", "count"]))
    export(output, manifest, requested)
    return coverage


def export(output, manifest, requested):
    """Export only successful selected dates, streaming; source kept, no cross-sensor dedup."""
    selected = [day.isoformat() for day in sorted(requested)
                if manifest["days"][day.isoformat()]["status"] in {"downloaded", "empty"}]
    fields = []
    for stamp in selected:
        with (output / "daily" / f"{stamp}.csv").open() as stream:
            for field in next(csv.reader(stream)):
                if field not in fields:
                    fields.append(field)
    fields = fields + [name for name in ["dataset", "chunk_start", "chunk_end"] if name not in fields]
    temporary = output / "recovered.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stamp in selected:
            with (output / "daily" / f"{stamp}.csv").open() as daily:
                for row in csv.DictReader(daily):
                    writer.writerow({**row, "dataset": manifest["config"]["source"], "chunk_start": stamp, "chunk_end": stamp})
    temporary.replace(output / "recovered.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--source", required=True)
    parser.add_argument("--area", default="world")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gap-segments", type=Path, help="Recover only dates absent from this legacy segment CSV")
    args = parser.parse_args()
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        parser.error("Set FIRMS_MAP_KEY in the environment")
    selected = gap_dates(args.gap_segments, args.start, args.end) if args.gap_segments else None
    try:
        coverage = acquire(key, args.source, args.start, args.end, args.output, args.area, selected)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"Acquisition stopped ({type(exc).__name__}); check configuration/connectivity.\n")
    unresolved = sum(row["status"] in {"failed", "unavailable"} for row in coverage)
    print(f"Days: {len(coverage)}; unresolved: {unresolved}")
    return 2 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
