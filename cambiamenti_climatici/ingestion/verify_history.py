"""Verify a complete FIRMS download offline and produce an auditable report."""
import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

from firms import atomic_text, days, parse_detections


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory, start, end):
    directory = Path(directory)
    expected = {str(day) for day in days(start, end)}
    require(bool(expected), "Empty date interval")
    manifest = json.loads((directory / "manifest.json").read_text())
    source = manifest["config"]["source"]
    with (directory / "coverage.csv").open() as stream:
        coverage = list(csv.DictReader(stream))
    require(len(coverage) == len(expected), "Coverage row count mismatch")
    require({row["date"] for row in coverage} == expected, "Missing or unexpected dates")
    counts = {}
    for row in coverage:
        stamp = row["date"]
        entry = manifest["days"][stamp]
        require(row["source"] == source, "Source mismatch")
        require(row["status"] in {"downloaded", "empty"}, f"Unresolved day: {stamp}")
        require(entry["status"] == row["status"], f"Status mismatch: {stamp}")
        payload = (directory / "daily" / f"{stamp}.csv").read_bytes()
        require(hashlib.sha256(payload).hexdigest() == entry["sha256"], f"Checksum mismatch: {stamp}")
        _, records = parse_detections(payload.decode(), date.fromisoformat(stamp), date.fromisoformat(stamp))
        counts[stamp] = len(records)
        require(len(records) == entry["count"] == int(row["count"]), f"Count mismatch: {stamp}")
        require((row["status"] == "empty") == (len(records) == 0), f"Empty status mismatch: {stamp}")
    export_counts = Counter()
    previous = ""
    with (directory / "recovered.csv").open() as stream:
        for row in csv.DictReader(stream):
            stamp = row["acq_date"]
            require(stamp in expected and stamp >= previous, "Unexpected or unsorted export date")
            require(row["dataset"] == source, "Export source mismatch")
            require(row["chunk_start"] == row["chunk_end"] == stamp, "Export provenance mismatch")
            export_counts[stamp] += 1
            previous = stamp
    require(all(export_counts[stamp] == count for stamp, count in counts.items()), "Export counts mismatch")
    digest = hashlib.sha256()
    with (directory / "recovered.csv").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        **manifest["config"], "start": str(start), "end": str(end),
        "requested_days": len(expected), "verified_days": len(counts),
        "days_with_records": sum(count > 0 for count in counts.values()),
        "unresolved_days": 0, "records": sum(counts.values()),
        "minimum_records_per_day": min(counts.values()),
        "maximum_records_per_day": max(counts.values()),
        "export_bytes": (directory / "recovered.csv").stat().st_size,
        "export_sha256": digest.hexdigest(),
        "checks": ["complete daily calendar", "daily checksums", "CSV schema/dates/coordinates",
                   "daily deduplication and counts", "export dates/source/counts"],
        "limitations": ["Daily detections do not establish observation coverage of each cell.",
                        "Single-source reconstruction replaces, rather than replicates, the old raw dataset.",
                        "Existing ML artifacts have not been regenerated."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.directory, args.start, args.end)
    atomic_text(args.report, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
