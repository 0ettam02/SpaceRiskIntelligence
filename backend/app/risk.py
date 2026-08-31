"""Scala qualitativa del rischio — deve restare identica a
``frontend/lib/risk-utils.js`` (RISK_LEVELS): è una configurazione
dimostrativa dell'interfaccia, non una soglia scientifica validata."""

RISK_LEVELS = [
    (0.0, 0.25, "bassa"),
    (0.25, 0.5, "moderata"),
    (0.5, 0.75, "elevata"),
    (0.75, 1.0001, "molto-elevata"),
]


def risk_level_for(probability):
    if probability is None:
        return None
    for lower, upper, label in RISK_LEVELS:
        if lower <= probability < upper:
            return label
    return RISK_LEVELS[-1][2]
