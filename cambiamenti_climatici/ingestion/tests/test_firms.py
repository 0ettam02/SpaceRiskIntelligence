import csv
from http.client import IncompleteRead
from datetime import date
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firms import acquire, blocks, gap_dates, parse_detections, request
from verify_history import verify

HEADER = "latitude,longitude,acq_date,acq_time,frp\n"
ROW = "42,12,2025-01-02,1200,5\n"
AVAIL = "data_id,min_date,max_date\nMODIS_SP,2025-01-02,2025-01-04\n"


class AcquisitionTests(unittest.TestCase):
    def test_firms_http_400_quota_is_retried(self):
        error = HTTPError("https://secret", 400, "quota", {},
                          io.BytesIO(b"Exceeding allowed transaction limit."))
        with patch("firms.urlopen") as mocked:
            response = mocked.return_value
            response.__enter__.return_value.read.return_value = b"complete"
            mocked.side_effect = [error, response]
            pauses = []
            self.assertEqual(request("https://secret", sleep=pauses.append), "complete")
            self.assertEqual(pauses, [60])

    def test_truncated_http_response_is_retried(self):
        with patch("firms.urlopen") as mocked:
            mocked.return_value.__enter__.return_value.read.side_effect = [
                IncompleteRead(b"partial"), b"complete"
            ]
            pauses = []
            self.assertEqual(request("https://secret", sleep=pauses.append), "complete")
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(pauses, [2])

    def test_offline_verification_rejects_incomplete_and_corrupt_export(self):
        day = date(2025, 1, 2)
        with tempfile.TemporaryDirectory() as folder:
            acquire("secret", "MODIS_SP", day, day, folder,
                    fetch=lambda url: AVAIL if "data_availability" in url else HEADER + ROW)
            report = verify(folder, day, day)
            self.assertEqual(report["records"], 1)
            with self.assertRaisesRegex(ValueError, "Coverage row count mismatch"):
                verify(folder, day, date(2025, 1, 3))
            exported = Path(folder) / "recovered.csv"
            exported.write_text(exported.read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "Export counts mismatch"):
                verify(folder, day, day)

    def test_expand_gap_recovery_to_full_history(self):
        calls = []
        def fetch(url):
            calls.append(url)
            if "data_availability" in url:
                return AVAIL
            day = url.rsplit("/", 1)[1]
            return HEADER + ROW.replace("2025-01-02", day)
        with tempfile.TemporaryDirectory() as folder:
            start, end = date(2025, 1, 2), date(2025, 1, 4)
            acquire("secret", "MODIS_SP", start, end, folder,
                    selected={date(2025, 1, 3)}, fetch=fetch)
            calls.clear()
            coverage = acquire("secret", "MODIS_SP", start, end, folder, fetch=fetch)
            self.assertEqual(len(coverage), 3)
            area_calls = [url for url in calls if "/area/" in url]
            self.assertEqual(len(area_calls), 2)
            self.assertTrue(area_calls[0].endswith("/1/2025-01-02"))
            self.assertTrue(area_calls[1].endswith("/1/2025-01-04"))
            with (Path(folder) / "recovered.csv").open() as stream:
                self.assertEqual([r["acq_date"] for r in csv.DictReader(stream)],
                                 ["2025-01-02", "2025-01-03", "2025-01-04"])

    def test_dates_dedup_and_invalid_responses(self):
        start, end = date(2025, 1, 2), date(2025, 1, 4)
        self.assertEqual(len(parse_detections(HEADER + ROW * 2, start, end)[1]), 1)
        self.assertEqual(parse_detections(HEADER, start, end)[1], [])
        for bad in ["", "Invalid MAP_KEY", HEADER + ROW.replace("2025-01-02", "2025-01-05")]:
            with self.assertRaises(ValueError):
                parse_detections(bad, start, end)

    def test_availability_empty_resume_and_corrupt_cache(self):
        calls = []
        def fetch(url):
            calls.append(url)
            return AVAIL if "data_availability" in url else HEADER + ROW * 2
        with tempfile.TemporaryDirectory() as folder:
            start, end = date(2025, 1, 1), date(2025, 1, 5)
            rows = acquire("secret", "MODIS_SP", start, end, folder, fetch=fetch)
            self.assertEqual([r["status"] for r in rows], ["unavailable", "downloaded", "empty", "empty", "unavailable"])
            self.assertTrue(calls[-1].endswith("/3/2025-01-02"))
            self.assertNotIn("secret", (Path(folder) / "manifest.json").read_text())
            calls.clear()
            acquire("secret", "MODIS_SP", start, end, folder, fetch=fetch)
            self.assertEqual(len(calls), 1)
            (Path(folder) / "daily/2025-01-02.csv").write_text("corrupted")
            calls.clear()
            acquire("secret", "MODIS_SP", start, end, folder, fetch=fetch)
            self.assertTrue(calls[-1].endswith("/1/2025-01-02"))
            with (Path(folder) / "recovered.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            with self.assertRaises(ValueError):
                acquire("secret", "MODIS_SP", start, end, folder, area="1,2,3,4", fetch=fetch)

    def test_failed_blocks_are_retried_on_resume(self):
        def failed(url):
            if "data_availability" in url:
                return AVAIL
            raise RuntimeError("network_error")
        with tempfile.TemporaryDirectory() as folder:
            day = date(2025, 1, 2)
            rows = acquire("secret", "MODIS_SP", day, day, folder, fetch=failed)
            self.assertEqual(rows[0]["status"], "failed")
            rows = acquire("secret", "MODIS_SP", day, day, folder,
                           fetch=lambda url: AVAIL if "data_availability" in url else HEADER + ROW)
            self.assertEqual(rows[0]["status"], "downloaded")

    def test_retry_and_secret_redaction(self):
        with patch("firms.urlopen", side_effect=HTTPError("https://secret", 429, "limit", {}, None)) as mocked:
            pauses = []
            with self.assertRaisesRegex(RuntimeError, "^HTTP 429$"):
                request("https://secret", attempts=3, sleep=pauses.append)
            self.assertEqual(mocked.call_count, 3)
            self.assertEqual(pauses, [2, 4])
        with patch("firms.urlopen", side_effect=HTTPError("https://secret", 401, "bad", {}, None)) as mocked:
            with self.assertRaisesRegex(RuntimeError, "^HTTP 401$"):
                request("https://secret", sleep=lambda _: None)
            self.assertEqual(mocked.call_count, 1)

    def test_gaps_and_blocks_do_not_bridge_observed_days(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "segments.csv"
            path.write_text("start,end\n2025-01-02,2025-01-03\n")
            missing = gap_dates(path, date(2025, 1, 1), date(2025, 1, 5))
            self.assertEqual(list(blocks(missing)), [[date(2025, 1, 1)], [date(2025, 1, 4), date(2025, 1, 5)]])


if __name__ == "__main__":
    unittest.main()
