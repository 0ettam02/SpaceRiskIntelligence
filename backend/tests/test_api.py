def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["cellsLoaded"] == 15000
    assert "random_forest" in body["modelsTrained"]


def test_overview_returns_real_kpis(client):
    response = client.get("/overview")
    assert response.status_code == 200
    body = response.json()
    values = {kpi["id"]: kpi["value"] for kpi in body["kpis"]}
    assert values["raw-detections"] == 16255053
    assert values["sampled-cells"] == 15000
    assert values["ml-rows"] == 194833
    assert len(body["dailySeries"]) == 448


def test_models_are_sorted_with_random_forest_recommended_first(client):
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["recommendedSlug"] == "random-forest"
    assert body["models"][0]["slug"] == "random-forest"
    assert body["models"][0]["recommended"] is True
    assert 0 <= body["models"][0]["accuracy"] <= 1


def test_model_detail_includes_observed_curves(client):
    response = client.get("/models/random-forest")
    assert response.status_code == 200
    body = response.json()
    assert body["curvesAreObserved"] is True
    assert len(body["rocCurve"]) > 0
    assert body["confusionMatrix"]["testRows"] > 0


def test_unknown_model_slug_is_404(client):
    response = client.get("/models/does-not-exist")
    assert response.status_code == 404


def test_map_cells_probability_filter_only_returns_cells_with_predictions(client):
    response = client.get("/map/cells", params={"metric": "probability"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] > 0
    assert all(cell["hasPrediction"] for cell in body["cells"])


def test_map_cell_detail_includes_historical_series(client):
    listing = client.get("/map/cells").json()
    cell_id = listing["cells"][0]["id"]
    response = client.get(f"/map/cells/{cell_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == cell_id
    assert isinstance(body["historicalSeries"], list)


def test_map_cell_detail_404_for_unknown_id(client):
    response = client.get("/map/cells/does-not-exist")
    assert response.status_code == 404


def test_data_quality_matches_real_coverage_numbers(client):
    response = client.get("/data-quality")
    assert response.status_code == 200
    coverage = response.json()["coverage"]
    assert coverage == {
        "observedDays": 310,
        "totalDays": 448,
        "missingDays": 138,
        "sampledCells": 15000,
        "usableSegments": 3,
        "totalSegments": 6,
    }
