"""Process already downloaded Italian ERA5-Land archives and merge with FIRMS."""
import argparse
import json

from era5_italy_pipeline import OUTPUT, process_and_merge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-04-26")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    print(json.dumps(process_and_merge(args.start, args.end, args.output), indent=2))


if __name__ == "__main__":
    main()
