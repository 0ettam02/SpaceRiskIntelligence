"""Expanding temporal evaluation with separate calibration/threshold periods and spatial holdout."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..build_ml_dataset import FEATURES as FIRMS_FEATURES
from .common import code_fingerprint, digest, environment, new_directory, verify_artifacts, write_json

TARGET = 'fire_next_7d'
MODEL_NAMES = ['constant', 'logistic', 'random_forest_baseline', 'random_forest_regularized', 'random_forest_firms_only']


def temporal_folds(start, end, cfg):
    """Calendar dates, not the rank of dates remaining after censoring."""
    ev = cfg['evaluation']
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    train_days = ev['minimum_train_days']
    result = []
    while True:
        train_end = start + pd.Timedelta(days=train_days - 1)
        partitions = {'train': (start, train_end)}
        previous = train_end
        for name, length in [('calibration', ev['calibration_days']), ('validation', ev['validation_days']), ('test', ev['test_days'])]:
            first = previous + pd.Timedelta(days=ev['embargo_days'] + 1)
            previous = first + pd.Timedelta(days=length - 1)
            partitions[name] = (first, previous)
        if previous > end:
            break
        result.append(partitions)
        train_days += ev['step_days']
    if not result:
        raise ValueError('Insufficient complete calendar for the configured evaluation')
    return result


def select_threshold(labels, probability, false_alert_budget, objective='false_alert_budget'):
    """Select operating point on validation only: F1, or recall under alert budget."""
    y, p = np.asarray(labels, dtype=int), np.asarray(probability, dtype=float)
    if not len(y) or not y.sum() or y.sum() == len(y):
        return float(np.nextafter(1., 2.)), 'insufficient_validation_classes; predict_no_alert'
    order = np.argsort(-p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(p) - 1]
    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp
    if objective == 'f1':
        fn = int(y.sum()) - tp
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
        best = max(range(len(ends)), key=lambda i: (f1[i], -fp[i], sorted_p[ends[i]]))
        return float(sorted_p[ends[best]]), 'validation_max_f1'
    if objective != 'false_alert_budget':
        raise ValueError(f'Unknown threshold objective: {objective}')
    eligible = np.flatnonzero(fp / len(y) * 7 <= false_alert_budget)
    if not len(eligible):
        return float(np.nextafter(1., 2.)), 'budget_requires_no_alert'
    # Maximum recall, then precision, then highest threshold. No access to test labels.
    best = max(eligible, key=lambda i: (tp[i], tp[i] / (ends[i] + 1), sorted_p[ends[i]]))
    return float(sorted_p[ends[best]]), 'validation_recall_under_false_alert_budget'


def metrics(labels, probability, threshold):
    y, p = np.asarray(labels, dtype=int), np.asarray(probability, dtype=float)
    if not len(y):
        return {'rows': 0, 'status': 'no_eligible_rows'}
    predicted = p >= threshold
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel()]
    positives = int(y.sum())
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, p)) if positives else None
    recall = tp / positives if positives else None
    specificity = tn / (tn + fp) if tn + fp else None
    reliability = []
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= left) & ((p < right) if right < 1 else (p <= right))
        if mask.any():
            reliability.append({'lower': float(left), 'upper': float(right), 'rows': int(mask.sum()),
                                'mean_probability': float(p[mask].mean()), 'observed_share': float(y[mask].mean())})
    return {'rows': len(y), 'positive_rows': positives, 'prevalence': prevalence,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
            'accuracy': (tp + tn) / len(y), 'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': recall, 'specificity': specificity,
            'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'roc_auc': float(roc_auc_score(y, p)) if 0 < positives < len(y) else None,
            'average_precision': ap, 'average_precision_lift': ap / prevalence if prevalence else None,
            'mcc': float(matthews_corrcoef(y, predicted)) if 0 < positives < len(y) else None,
            'brier': float(brier_score_loss(y, p)), 'false_alerts_per_cell_week': fp * 7 / len(y),
            'reliability_bins': reliability}


def logit(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(probability, labels):
    y = np.asarray(labels)
    if len(y) == 0 or min(int(y.sum()), int(len(y) - y.sum())) < 20:
        return None, 'skipped: fewer than 20 examples of either class in calibration'
    model = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    model.fit(logit(probability), y)
    return model, 'sigmoid fitted on separate chronological calibration period'


def episode_metrics(predictions, episodes):
    """Only episodes whose full seven-day pre-onset opportunity window was evaluated."""
    if predictions.empty:
        return {'eligible_episodes': 0, 'hit_episodes': 0, 'recall': None, 'median_lead_days': None}
    lookups = {cell: frame.set_index('date').prediction for cell, frame in predictions.groupby('cell_id')}
    eligible, leads = 0, []
    lower, upper = predictions.date.min(), predictions.date.max() + pd.Timedelta(days=7)
    for event in episodes[episodes.onset.between(lower + pd.Timedelta(days=1), upper)].itertuples():
        if event.cell_id not in lookups:
            continue
        dates = pd.date_range(event.onset - pd.Timedelta(days=7), event.onset - pd.Timedelta(days=1))
        values = lookups[event.cell_id].reindex(dates)
        if values.isna().any():
            continue
        eligible += 1
        hits = values[values.eq(1)]
        if len(hits):
            leads.append(int((event.onset - hits.index.min()).days))
    return {'eligible_episodes': eligible, 'hit_episodes': len(leads),
            'recall': len(leads) / eligible if eligible else None,
            'median_lead_days': float(np.median(leads)) if leads else None,
            'definition': 'Within-cell detection clusters separated by >7 days; not independently verified wildfire incidents.'}


def block_intervals(predictions, seed, repetitions=500):
    """Resample whole spatial blocks, preserving all within-block temporal dependence."""
    if predictions.empty or predictions.spatial_block.nunique() < 2:
        return {'status': 'insufficient_spatial_blocks'}
    frame = predictions.copy()
    frame['tp'] = frame.prediction.eq(1) & frame[TARGET].eq(1)
    frame['fp'] = frame.prediction.eq(1) & frame[TARGET].eq(0)
    counts = frame.groupby('spatial_block').agg(tp=('tp', 'sum'), fp=('fp', 'sum'), positives=(TARGET, 'sum'), rows=(TARGET, 'size')).to_numpy()
    rng = np.random.default_rng(seed)
    samples = counts[rng.integers(len(counts), size=(repetitions, len(counts)))].sum(axis=1)
    result = {'method': 'spatial-block percentile bootstrap; neighbouring blocks may remain correlated',
              'blocks': len(counts), 'repetitions': repetitions, 'percentiles': [2.5, 97.5]}
    for name, numerator, denominator in [('recall', samples[:, 0], samples[:, 2]),
                                         ('precision', samples[:, 0], samples[:, 0] + samples[:, 1]),
                                         ('false_alerts_per_cell_week', 7 * samples[:, 1], samples[:, 3])]:
        valid = denominator > 0
        result[name] = {'valid_replicates': int(valid.sum()),
                        'interval': np.quantile(numerator[valid] / denominator[valid], [.025, .975]).tolist() if valid.any() else None}
    return result


def estimator(name, cfg):
    ev = cfg['evaluation']
    if name == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=cfg['random_state']))
    kwargs = dict(n_estimators=ev['rf_trees'], max_features='sqrt', n_jobs=-1, random_state=cfg['random_state'])
    if name in ['random_forest_baseline', 'random_forest_firms_only']:
        return RandomForestClassifier(**kwargs, min_samples_leaf=5, class_weight='balanced_subsample')
    if name == 'random_forest_regularized':
        return RandomForestClassifier(**kwargs, min_samples_leaf=ev['rf_regularized_leaf'], max_depth=ev['rf_regularized_depth'])
    raise ValueError(name)


def evaluate(dataset_dir, output, models=None, max_folds=None, threshold_objective='false_alert_budget'):
    dataset_dir = Path(dataset_dir)
    manifest = verify_artifacts(dataset_dir, ['ml.csv.gz', 'episodes.csv'])
    cfg, features = manifest['config'], manifest['features']
    names = ([name for name in MODEL_NAMES if name != 'random_forest_firms_only' or len(features) > len(FIRMS_FEATURES)]
             if models is None else models)
    if not names or set(names) - set(MODEL_NAMES):
        raise ValueError('Unknown/empty model selection')
    data = pd.read_csv(dataset_dir / 'ml.csv.gz', parse_dates=['date'], dtype={f: 'float32' for f in features})
    if data.duplicated(['cell_id', 'date']).any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError('Invalid evaluation keys/labels')
    if not np.isfinite(data[features]).all().all():
        raise ValueError('Non-finite evaluation features')
    episodes = pd.read_csv(dataset_dir / 'episodes.csv', parse_dates=['onset', 'last_detection'])
    start = pd.Timestamp(cfg['start']) + pd.Timedelta(days=cfg['history_days'])
    end = pd.Timestamp(cfg['end']) - pd.Timedelta(days=cfg['horizon_days'])
    folds = temporal_folds(start, end, cfg)
    if max_folds is not None:
        if max_folds <= 0:
            raise ValueError('max_folds must be positive')
        folds = folds[-max_folds:]
    out = new_directory(output)
    summaries, group_metrics = [], []
    for index, dates in enumerate(folds):
        parts = {key: data[data.date.between(*period)] for key, period in dates.items()}
        # Held-out spatial blocks are absent from training, calibration and threshold selection.
        development = {key: value[value.spatial_holdout.eq(0)] for key, value in parts.items()}
        train = development['train']
        if train.empty:
            raise ValueError('No development cells available')
        y = train[TARGET]
        prevalence = float(y.mean())
        for name in names:
            print(f'fold={index} model={name} train_rows={len(train)}', flush=True)
            model_features = FIRMS_FEATURES if name == 'random_forest_firms_only' else features
            X = train[model_features]
            fitted = None
            if name != 'constant' and y.nunique() == 2:
                fitted = estimator(name, cfg).fit(X, y)
            def raw_probability(frame):
                if frame.empty:
                    return np.empty(0)
                return fitted.predict_proba(frame[model_features])[:, 1] if fitted is not None else np.full(len(frame), prevalence)
            calibration = development['calibration']
            calibrator, calibration_status = (calibrate(raw_probability(calibration), calibration[TARGET])
                                               if fitted is not None else (None, 'constant_training_prevalence'))
            def probability(frame):
                p = raw_probability(frame)
                return calibrator.predict_proba(logit(p))[:, 1] if calibrator is not None and len(p) else p
            validation = development['validation']
            threshold, policy = select_threshold(validation[TARGET], probability(validation), cfg['evaluation']['false_alert_budget_per_cell_week'], threshold_objective)
            fold_dir = out / f'fold_{index}_{name}'
            fold_dir.mkdir()
            joblib.dump({'estimator': fitted, 'calibrator': calibrator, 'training_prevalence': prevalence,
                         'features': model_features, 'threshold': threshold, 'target': TARGET,
                         'dataset_manifest_sha256': digest(dataset_dir / 'manifest.json')}, fold_dir / 'model.joblib')
            result = {'fold': index, 'model': name, 'features': model_features,
                      'periods': {k: [str(d.date()) for d in v] for k, v in dates.items()},
                      'development_counts': {k: {'rows': len(v), 'positive': int(v[TARGET].sum())} for k, v in development.items()},
                      'threshold': threshold, 'threshold_policy': policy, 'calibration_status': calibration_status,
                      'estimator_status': 'fitted' if fitted is not None else 'constant_or_single_class_training',
                      'validation': metrics(validation[TARGET], probability(validation), threshold), 'test': {}}
            for scope, test in [('seen_cells', development['test']), ('unseen_spatial_blocks', parts['test'][parts['test'].spatial_holdout.eq(1)])]:
                p = probability(test)
                predicted = test[['cell_id', 'spatial_block', 'date', TARGET]].copy()
                predicted['probability'] = p
                predicted['prediction'] = (p >= threshold).astype('int8')
                predicted.to_csv(fold_dir / f'{scope}_predictions.csv.gz', index=False)
                raw_metrics = metrics(test[TARGET], raw_probability(test), threshold)
                result['test'][scope] = {'metrics': metrics(test[TARGET], p, threshold),
                                         'uncalibrated_probability_metrics': {key: raw_metrics.get(key) for key in ['roc_auc', 'average_precision', 'brier']},
                                         'block_bootstrap': block_intervals(predicted, cfg['random_state']),
                                         'episodes': episode_metrics(predicted, episodes),
                                         'monthly': {str(month): metrics(frame[TARGET], frame.probability, threshold)
                                                     for month, frame in predicted.groupby(predicted.date.dt.to_period('M'))}}
                for block, frame in predicted.groupby('spatial_block'):
                    values = metrics(frame[TARGET], frame.probability, threshold)
                    values.pop('reliability_bins', None)
                    group_metrics.append({'fold': index, 'model': name, 'scope': scope, 'block': block, **values})
            summaries.append(result)
            write_json(fold_dir / 'metrics.json', result)
    pd.DataFrame(group_metrics).to_csv(out / 'spatial_metrics.csv', index=False)
    report = {'schema_version': 2, 'config': cfg, 'environment': environment(), 'code_sha256': code_fingerprint(),
              'dataset_manifest_sha256': digest(dataset_dir / 'manifest.json'),
              'scope': manifest['scope'], 'features': features, 'models': names,
              'threshold_objective': threshold_objective,
              'fold_count': len(folds), 'results': summaries,
              'status': 'exploratory_not_production',
              'limitations': ['Historical dates have already been inspected; reserve a new future period before final model selection.',
                              'Spatial blocks are fixed before fitting; neighbouring blocks may remain correlated (no geographic buffer).',
                              'Episode counts are detection clusters, not independently verified wildfire incidents.',
                              'Observed validation false-alert budget is not guaranteed on test or in operation.',
                              'No automatic model ranking, model selection or promotion from these test results.']}
    write_json(out / 'report.json', report)
    return {'report': str(out / 'report.json'), 'folds': len(folds), 'scope': manifest['scope'], 'status': report['status']}
