from app.risk import risk_level_for


def test_bassa_for_low_probability():
    assert risk_level_for(0.05) == "bassa"


def test_moderata_for_mid_probability():
    assert risk_level_for(0.3) == "moderata"


def test_elevata_for_high_probability():
    assert risk_level_for(0.6) == "elevata"


def test_molto_elevata_for_very_high_probability():
    assert risk_level_for(0.9) == "molto-elevata"


def test_boundaries_match_documented_thresholds():
    assert risk_level_for(0.25) == "moderata"
    assert risk_level_for(0.75) == "molto-elevata"


def test_none_for_missing_probability():
    assert risk_level_for(None) is None
