from app.geo import label_for


def test_labels_amazon_basin_as_south_america():
    assert label_for(-10, -60) == "Sud America"


def test_labels_central_africa():
    assert label_for(-15.9, 25.6) == "Africa"


def test_falls_back_when_no_region_matches():
    assert label_for(-89, 0) == "Regione non classificata"
