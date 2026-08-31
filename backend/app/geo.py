"""Etichetta geografica di cortesia (a livello di continente/macro-regione)
calcolata da lat/lon con bounding box approssimative. Serve solo a rendere
leggibile il pannello di dettaglio cella: non è una fonte geografica
autorevole né sostituisce le coordinate, sempre mostrate accanto ad essa."""

_REGIONS = [
    ("Groenlandia e Artico", -75, 180, 60, 90),
    ("Nord America", -170, -50, 15, 72),
    ("America centrale", -118, -60, 7, 15),
    ("Sud America", -82, -34, -56, 13),
    ("Europa", -25, 45, 35, 72),
    ("Africa", -18, 52, -35, 37),
    ("Asia settentrionale", 45, 180, 45, 78),
    ("Asia meridionale e orientale", 60, 150, -11, 45),
    ("Oceania", 110, 180, -50, -10),
]


def label_for(lat, lon):
    for label, lon_min, lon_max, lat_min, lat_max in _REGIONS:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return label
    return "Regione non classificata"
