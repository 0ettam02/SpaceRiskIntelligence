# Risultati definitivi — previsione `fire_next_7d`

Valutazione su test temporale isolato del segmento aprile–settembre 2025, con 7 giorni di embargo fra train, validation e test. La soglia è scelta esclusivamente sulla validation.

| Modello | Accuracy | AUC ROC | PR-AUC | Precision | Recall | F1 | Soglia |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | **0,731** | **0,816** | **0,862** | **0,715** | 0,903 | **0,798** | 0,38 |
| Albero decisionale | 0,713 | 0,803 | 0,846 | 0,694 | 0,917 | 0,790 | 0,34 |
| Regressione logistica | 0,664 | 0,813 | 0,859 | 0,643 | **0,963** | 0,771 | 0,42 |
| Regressione polinomiale | 0,697 | 0,762 | 0,787 | 0,692 | 0,875 | 0,773 | 0,41 |
| SVM RBF approssimata | 0,427 | 0,339 | 0,504 | 0,511 | 0,644 | 0,570 | 0,57 |

Scelta raccomandata: **Random Forest**. Il dettaglio completo è in `output_definitivo/modelli/confronto_modelli_storico_v3.csv`.
