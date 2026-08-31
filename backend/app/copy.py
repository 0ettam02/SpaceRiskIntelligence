"""Testi statici in italiano, tenuti coerenti con i mock del frontend
(``frontend/data/mock-*.js``) così l'esperienza non cambia visibilmente
passando da NEXT_PUBLIC_DATA_SOURCE=mock a =api."""

METHODOLOGY_WARNINGS = [
    "Un rilevamento satellitare non corrisponde necessariamente a un incendio fisico distinto: più rilevamenti possono derivare dallo stesso evento.",
    "Il campione di celle è condizionato: comprende prevalentemente celle già attive in passato e non consente di stimare prevalenze globali senza pesi di inclusione.",
    "Le metriche dei modelli derivano da un singolo split temporale isolato e non da una validazione incrociata su più segmenti.",
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
}

MODEL_LIMITATIONS = [
    "Le metriche derivano da un singolo split temporale isolato e non da una validazione incrociata multi-segmento.",
    "Il campione di celle è condizionato: include prevalentemente celle già attive in passato, non un campione casuale della superficie terrestre.",
    "La soglia di decisione è una configurazione dimostrativa dell'interfaccia; non è stata validata per un uso operativo.",
    "Il modello è addestrato in memoria all'avvio di questo backend e non è persistito su disco: riavviare il processo lo riaddestra da capo sugli stessi dati.",
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
