"""Testi statici in italiano, tenuti coerenti con i mock del frontend
(``frontend/data/mock-*.js``) così l'esperienza non cambia visibilmente
passando da NEXT_PUBLIC_DATA_SOURCE=mock a =api."""

METHODOLOGY_WARNINGS = [
    "Un rilevamento satellitare non corrisponde necessariamente a un incendio fisico distinto: più rilevamenti possono derivare dallo stesso evento.",
    "Il campione di celle è condizionato: comprende prevalentemente celle già attive in passato e non consente di stimare prevalenze globali senza pesi di inclusione.",
    "Le metriche del modello clean3 + ERA5-Land derivano da un holdout casuale 80/20 e non da una validazione temporale o geografica indipendente.",
    "La generalizzazione geografica del modello a regioni non rappresentate nel campione non è stata verificata.",
]

MODEL_METHODOLOGY_NOTES = {
    "random_forest": [
        "Valutato su test temporale isolato (segmento più lungo disponibile) con 7 giorni di embargo rispetto a training e validation.",
        "La soglia di decisione è scelta esclusivamente sul set di validation, massimizzando l'F1, non sul test.",
        "Il recall elevato comporta una specificità più contenuta: una quota rilevante di celle negative viene classificata come positiva.",
    ],
    "regressione_logistica": [
        "Recall molto elevato ma specificità bassa: il modello tende a segnalare come positiva la quasi totalità delle celle.",
        "Utile come baseline lineare interpretabile, meno adatta quando il costo dei falsi positivi è rilevante.",
    ],
    "albero_decisionale": [
        "Prestazioni vicine alla Random Forest ma con maggiore varianza attesa fuori campione, essendo un singolo albero non ensemble.",
    ],
    "regressione_polinomiale": [
        "Estensione polinomiale della regressione logistica: migliora leggermente la separabilità ma introduce rischio di overfitting sulle feature di intensità.",
    ],
    "svm_rbf_approssimata": [
        "Approssimazione RBF con feature map casuale su un sottoinsieme del training per contenere i tempi di calcolo.",
    ],
    "random_forest_italia_firms": [
        "Modello italiano separato, confrontato sul test temporale delle 19 celle campionate.",
        "Usa le stesse 17 feature FIRMS della baseline globale; non è un modello di copertura nazionale completa.",
    ],
    "random_forest_italia_era5": [
        "RandomForestClassifier(n_estimators=100, random_state=42) sui lag e target del dataset dataset_pulito3_lag_target_final.csv, filtrato alla griglia italiana.",
        "Il confronto FIRMS-only / FIRMS + ERA5-Land usa le stesse righe e lo stesso split casuale 80/20; ERA5 aggiunge otto feature meteo.",
        "Le metriche del random split non misurano generalizzazione su date o celle nuove e possono essere ottimistiche.",
        "Il forecast usa feature FIRMS fino al 15/07/2026 per la settimana 16–22/07/2026; non descrive il rischio attuale.",
        "Per 21 celle costiere è stata associata la serie del più vicino punto ERA5 terrestre, entro 11,2 km.",
    ],
}

MODEL_LIMITATIONS = [
    "Le metriche derivano da un singolo split temporale isolato e non da una validazione incrociata multi-segmento.",
    "Il campione di celle è condizionato: include prevalentemente celle già attive in passato, non un campione casuale della superficie terrestre.",
    "La soglia di decisione è una configurazione dimostrativa dell'interfaccia; non è stata validata per un uso operativo.",
    "Il modello è addestrato in memoria all'avvio di questo backend e non è persistito su disco: riavviare il processo lo riaddestra da capo sugli stessi dati.",
    "In modalità mock, curve e distribuzioni sono dimostrative; con l'API derivano dalle predizioni del test set.",
]

ITALY_MODEL_LIMITATIONS = [
    "Il clean3 globale contiene 55.255 celle; la griglia FIRMS italiana seleziona 331 celle nel periodo 18/06–15/07/2026.",
    "Lo snapshot meteo copre le date del FIRMS; 310 celle hanno un match ERA5 esatto e 21 usano il punto terrestre vicino (massimo 11,2 km).",
    "Le probabilità mostrate si riferiscono alla finestra 16–22/07/2026 e non sono aggiornate al giorno corrente.",
    "Il target misura rilevamenti FIRMS presunti incendi di vegetazione, non incendi verificati indipendentemente.",
    "Le feature ERA5-Land sono stime su griglia 0,1° e non misure locali di stazione.",
    "Il test casuale non è una valutazione temporale o geografica indipendente; il modello non è validato per allerta operativa.",
]

DATA_QUALITY_WARNINGS = [
    "Possibili duplicati fra rilevamenti provenienti da fonti diverse dello stesso sensore.",
    "Le celle mai attive nel periodo osservato sono escluse dal campione.",
    "Il campione è condizionato alla storicità di attività, non casuale.",
    "La generalizzazione geografica del modello non è stata verificata.",
    "Assenza di un modello serializzato pronto per l'inferenza: questo backend lo riaddestra ad ogni avvio.",
]

RESEARCH_DISCLAIMER = (
    "SpaceRiskIntelligence è un prototipo di ricerca. Le stime mostrate non costituiscono un sistema operativo di "
    "allerta incendi e non devono essere utilizzate per decisioni di emergenza."
)
