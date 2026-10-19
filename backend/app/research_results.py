"""Expose observed v2 evaluation artifacts without promoting a forecasting model."""
import json
import logging
from pathlib import Path

from app.config import REPO_ROOT

logger = logging.getLogger(__name__)
ROOT = REPO_ROOT / 'dati' / 'italy_v2'
METRICS = ['rows', 'positive_rows', 'prevalence', 'precision', 'recall', 'f1',
           'balanced_accuracy', 'roc_auc', 'average_precision', 'brier',
           'false_alerts_per_cell_week', 'true_positive', 'false_positive',
           'false_negative', 'true_negative']
STUDIES = [
    ('firms', 'FIRMS · griglia nazionale', 'evaluation_firms_verified', 'firms_features'),
    ('era5', 'FIRMS + ERA5-Land · Italia (copertura parziale)', 'evaluation_arco_paired_f1', 'weather_arco_features_diagnostic'),
]


def load_research_results(root=ROOT):
    root = Path(root)
    studies, unavailable = [], []
    for key, title, evaluation, dataset in STUDIES:
        try:
            report = json.loads((root / evaluation / 'report.json').read_text())
            manifest = json.loads((root / dataset / 'manifest.json').read_text())
            if report['schema_version'] != 2 or report['status'] != 'exploratory_not_production':
                raise ValueError('Unsupported evaluation report')
            results = []
            for row in report['results']:
                results.append({
                    'fold': row['fold'], 'model': row['model'], 'periods': row['periods'],
                    'threshold': row['threshold'], 'thresholdPolicy': row['threshold_policy'],
                    'calibrationStatus': row['calibration_status'], 'featureCount': len(row['features']),
                    'test': {scope: {'metrics': {name: value['metrics'].get(name) for name in METRICS},
                                     'episodes': value['episodes']}
                             for scope, value in row['test'].items()},
                })
            studies.append({'id': key, 'title': title, 'scope': report['scope'],
                            'cells': manifest['cells'], 'rows': manifest['rows'],
                            'positiveRows': manifest['positive_rows'], 'results': results,
                            'target': manifest['target_definition'],
                        'thresholdObjective': report.get('threshold_objective', 'false_alert_budget'),
                        'start': report['config']['start'], 'end': report['config']['end'],
                            'falseAlertBudget': report['config']['evaluation']['false_alert_budget_per_cell_week']})
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning('V2 evaluation unavailable or invalid: %s', evaluation)
            unavailable.append(key)
    coverage = None
    try:
        weather = json.loads((root / 'weather' / 'availability_arco.json').read_text())
        plan = json.loads((root / 'weather' / 'plan.json').read_text())
        coverage = {'availableCells': weather['available_cells'], 'requiredCells': weather['required_cells'],
                    'missingCells': weather['required_cells'] - weather['available_cells'],
                    'complete': weather['complete'],
                    'uniquePoints': plan['unique_points']}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {'source': 'api', 'studies': studies, 'unavailable': unavailable,
            'weatherCoverage': coverage, 'productionPromoted': False}
