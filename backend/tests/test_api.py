def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["cellsLoaded"] == 19
    assert body["modelsTrained"] == ["random_forest_italia_era5"]


def test_overview_returns_real_kpis(client):
    response = client.get("/overview")
    assert response.status_code == 200
    body = response.json()
    values = {kpi["id"]: kpi["value"] for kpi in body["kpis"]}
    assert values["raw-detections"] == 75
    assert values["sampled-cells"] == 19
    assert values["ml-rows"] == 7809
    assert len(body["dailySeries"]) == 411


def test_only_italy_era5_random_forest_is_selected_for_predictions(client):
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["recommendedSlug"] == "random-forest-italia-era5"
    assert len(body["models"]) == 1
    assert body["models"][0]["slug"] == "random-forest-italia-era5"
    assert body["models"][0]["recommended"] is True
    assert 0 <= body["models"][0]["accuracy"] <= 1


def test_model_detail_includes_observed_curves(client):
    response = client.get("/models/random-forest-italia-era5")
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
    assert body["total"] == 19
    assert all(cell["hasPrediction"] for cell in body["cells"])
    assert all(cell["model"] == "Random Forest FIRMS + ERA5-Land" for cell in body["cells"])
    assert all(35 <= cell["lat"] <= 48 and 6 <= cell["lon"] <= 20 for cell in body["cells"])


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
    assert coverage["observedDays"] == 411
    assert coverage["totalDays"] == 411
    assert coverage["missingDays"] == 0
    assert coverage["sampledCells"] == 19
    assert coverage["totalSegments"] == len(response.json()["segments"])
