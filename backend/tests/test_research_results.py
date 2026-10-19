import json

from app.research_results import load_research_results


def test_missing_research_is_explicit_and_does_not_invent_metrics(tmp_path):
    result = load_research_results(tmp_path)
    assert result['studies'] == []
    assert result['unavailable'] == ['firms', 'era5']
    assert result['productionPromoted'] is False
    assert result['weatherCoverage'] is None


def test_observed_metrics_and_undefined_scores_survive_loading(tmp_path):
    evaluation = tmp_path / 'evaluation_firms_verified'
    dataset = tmp_path / 'firms_features'
    evaluation.mkdir()
    dataset.mkdir()
    report = {'schema_version': 2, 'status': 'exploratory_not_production', 'scope': 'national_grid',
          'config': {'start': '2025-01-01', 'end': '2026-01-28',
                     'evaluation': {'false_alert_budget_per_cell_week': .1}},
              'results': [{'fold': 0, 'model': 'constant', 'periods': {'test': ['2026-01-01', '2026-01-28']},
                           'threshold': 1.0000000000000002, 'threshold_policy': 'no_alert',
                           'calibration_status': 'constant', 'features': ['x'],
                           'test': {'seen_cells': {'metrics': {'rows': 28, 'positive_rows': 0, 'roc_auc': None, 'recall': None},
                                                   'episodes': {'eligible_episodes': 0, 'hit_episodes': 0}}}}]}
    (evaluation / 'report.json').write_text(json.dumps(report))
    (dataset / 'manifest.json').write_text(json.dumps({'cells': 1, 'rows': 100, 'positive_rows': 5, 'target_definition': 'observed detections'}))
    result = load_research_results(tmp_path)
    assert result['unavailable'] == ['era5']
    study = result['studies'][0]
    assert study['cells'] == 1
    metric = study['results'][0]['test']['seen_cells']['metrics']
    assert metric['rows'] == 28
    assert metric['roc_auc'] is None
    assert metric['recall'] is None
    json.dumps(result, allow_nan=False)


def test_corrupt_artifact_is_unavailable(tmp_path):
    folder = tmp_path / 'evaluation_firms_verified'
    folder.mkdir()
    (folder / 'report.json').write_text('{invalid')
    assert load_research_results(tmp_path)['studies'] == []


def test_arco_evaluation_and_coverage_are_loaded_from_verified_artifacts(tmp_path):
    evaluation = tmp_path / 'evaluation_arco_paired_f1'
    dataset = tmp_path / 'weather_arco_features_diagnostic'
    weather = tmp_path / 'weather'
    for folder in (evaluation, dataset, weather):
        folder.mkdir()
    report = {'schema_version': 2, 'status': 'exploratory_not_production', 'scope': 'partial_diagnostic',
              'threshold_objective': 'f1', 'config': {'start': '2025-04-26', 'end': '2026-07-17',
              'evaluation': {'false_alert_budget_per_cell_week': .1}}, 'results': []}
    (evaluation / 'report.json').write_text(json.dumps(report))
    (dataset / 'manifest.json').write_text(json.dumps({'cells': 3185, 'rows': 1000, 'positive_rows': 12,
                                                        'target_definition': 'observed detections'}))
    (weather / 'availability_arco.json').write_text(json.dumps({'available_cells': 3185, 'required_cells': 3475,
                                                                 'complete': False}))
    (weather / 'plan.json').write_text(json.dumps({'unique_points': 3419}))

    result = load_research_results(tmp_path)

    study = next(item for item in result['studies'] if item['id'] == 'era5')
    assert study['cells'] == 3185
    assert study['thresholdObjective'] == 'f1'
    assert study['start'] == '2025-04-26'
    assert result['weatherCoverage'] == {'availableCells': 3185, 'requiredCells': 3475,
                                         'missingCells': 290, 'complete': False, 'uniquePoints': 3419}
