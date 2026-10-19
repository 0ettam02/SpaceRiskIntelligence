"""Run from repository root: python -m cambiamenti_climatici.ingestion.italy_v2 --help."""
import argparse
import json
from pathlib import Path

from .common import DEFAULT_CONFIG, ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze-v1')
    freeze.add_argument('--output', type=Path, required=True)
    build = commands.add_parser('build-panel')
    build.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    build.add_argument('--history', type=Path, default=ROOT / 'cambiamenti_climatici/ingestion/output/modis_sp_history')
    build.add_argument('--countries', type=Path, default=ROOT / 'frontend/public/data/countries.geojson')
    build.add_argument('--output', type=Path, required=True)
    plan = commands.add_parser('plan-weather')
    plan.add_argument('--panel', type=Path, required=True)
    plan.add_argument('--cache', type=Path, default=ROOT / 'dati/era5_italy')
    plan.add_argument('--output', type=Path, required=True)
    weather = commands.add_parser('weather')
    weather.add_argument('--plan', type=Path, required=True)
    weather.add_argument('--download', action='store_true')
    weather.add_argument('--max-requests', type=int, default=0)
    weather.add_argument('--area-mode', action='store_true', help='Use monthly 4-degree tiles from the standard ERA5-Land product')
    weather.add_argument('--arco-mode', action='store_true', help='Read ERA5-Land directly from the official ARCO Zarr archives')
    weather.add_argument('--parallelism', type=int, default=1, help='Concurrent CDS requests in area mode (CDS queue limited; default 1)')
    features = commands.add_parser('build-features')
    features.add_argument('--panel', type=Path, required=True)
    features.add_argument('--weather', type=Path)
    features.add_argument('--allow-partial-weather', action='store_true')
    features.add_argument('--include-observation-day', action='store_true', help='Use accepted FIRMS detections observed through date t to predict t+1..t+7')
    features.add_argument('--output', type=Path, required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--dataset', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--models', nargs='+', choices=['constant', 'logistic', 'random_forest_baseline', 'random_forest_regularized', 'random_forest_firms_only'])
    evaluate.add_argument('--max-folds', type=int)
    evaluate.add_argument('--threshold-objective', choices=['false_alert_budget', 'f1'], default='false_alert_budget')
    args = parser.parse_args()
    if args.command == 'freeze-v1':
        from .build import freeze_v1
        result = freeze_v1(args.output)
    elif args.command == 'build-panel':
        from .build import build
        result = build(args.history, args.countries, args.output, load_config(args.config))
    elif args.command == 'plan-weather':
        from .weather import plan
        result = plan(args.panel, args.output, args.cache)
    elif args.command == 'weather':
        if args.arco_mode:
            from .weather_area import materialize_arco
            result = materialize_arco(args.plan)
        elif args.area_mode:
            from .weather_area import materialize_areas
            result = materialize_areas(args.plan, args.download, args.max_requests, args.parallelism)
        else:
            from .weather import materialize
            result = materialize(args.plan, args.download, args.max_requests)
    elif args.command == 'build-features':
        from .features import build_features
        result = build_features(args.panel, args.output, args.weather, args.allow_partial_weather, args.include_observation_day)
    else:
        from .evaluate import evaluate
        result = evaluate(args.dataset, args.output, args.models, args.max_folds, args.threshold_objective)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
