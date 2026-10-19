"""Export observed local v2 results for the frontend's explicitly labelled demo mode."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from app.research_results import ROOT, load_research_results


def export(output, root=ROOT):
    result = load_research_results(root)
    if result['unavailable'] or len(result['studies']) != 2:
        raise ValueError('Both verified studies must be present before exporting a demo snapshot')
    result['source'] = 'snapshot'
    result['exportedAt'] = datetime.now(timezone.utc).isoformat()
    names = ['evaluation_firms_verified/report.json', 'evaluation_arco_paired_f1/report.json',
             'firms_features/manifest.json', 'weather_arco_features_diagnostic/manifest.json',
             'weather/availability_arco.json', 'weather/plan.json', 'weather/daily_arco.csv.gz']
    result['artifactSha256'] = {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in names}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'frontend/data/research-results.json')
    args = parser.parse_args()
    export(args.output)
