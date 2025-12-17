# Progetto_EmanueleAnnaGianmarco

Progetto Finale Academy Python &amp; Machine Learning
Partecipanti:

- Emanuele Anzellotti
- Anna Firinu
- Gianmarco Sorrentino

# Pulizia dei Dati (Gianmarco Sorrentino, Anna Firinu, Emanuele Anzellotti) e Analisi Esplorativa (Anna Firinu, Emanuele Anzellotti)

## 1. Introduzione

Questa sezione descrive il processo di **pulizia del dataset**, **analisi esplorativa dei dati (EDA)** e **feature engineering** svolto come fase preliminare alla costruzione di un **sistema di raccomandazione musicale basato sul contenuto**, con particolare attenzione al problema del **cold start**.

L’obiettivo è ottenere un dataset:

- pulito e coerente,
- interpretabile dal punto di vista musicale,
- adatto al calcolo di similarità tra brani.

---

## 2. Caricamento del Dataset e Ispezione Iniziale

Il dataset è stato caricato utilizzando la libreria `pandas` ed è stato inizialmente analizzato tramite:

- `head()` per una prima ispezione dei dati,
- `info()` per verificare i tipi di dato e la presenza di valori mancanti,
- `describe()` per analizzare le statistiche descrittive,
- conteggio dei valori mancanti e dei duplicati.

Questa fase ha permesso di comprendere la struttura generale del dataset e di individuare eventuali criticità.

---

## 3. Pulizia dei Dati

### 3.1 Gestione dei Valori Mancanti

I valori mancanti sono stati rimossi utilizzando il metodo `dropna()`.  
Questa scelta è stata ritenuta appropriata data la dimensione del dataset e la necessità di lavorare con feature complete per il calcolo delle similarità.

### 3.2 Gestione dei Duplicati

Per evitare la presenza di più istanze della stessa canzone, il dataset è stato aggregato utilizzando:

- `track_name`,
- `artists`,
- `album_name`.

Le feature numeriche sono state aggregate tramite media, mentre le feature categoriali sono state mantenute selezionando il primo valore disponibile.  
Questo garantisce una rappresentazione unica e coerente di ciascun brano.

---

## 4. Analisi dei Generi Musicali

### 4.1 Distribuzione dei Generi

È stata analizzata la distribuzione della variabile `track_genre` per individuare eventuali sbilanciamenti nel dataset.

### 4.2 Mappatura dei Generi

Per ridurre la dimensionalità e migliorare la robustezza nella fase di cold start, i generi originali sono stati mappati in un insieme ridotto di **macro-generi** (`main_genre`).  
Il genere originale è stato comunque mantenuto come `sub_genre`.

Questa scelta consente:

- raccomandazioni iniziali più stabili,
- una maggiore granularità quando si accumula feedback dell’utente.

---

## 5. Analisi delle Audio Feature

Le principali audio feature (ad esempio `energy`, `danceability`, `valence`, `acousticness`, ecc.) sono state analizzate tramite istogrammi al fine di studiarne:

- la distribuzione dei valori,
- eventuali asimmetrie,
- la presenza di valori estremi o concentrati.

L’analisi ha evidenziato che le feature catturano aspetti musicali differenti e non risultano ridondanti.

---

## 6. Analisi di Correlazione

È stata calcolata una matrice di correlazione tra le audio feature per analizzare le relazioni lineari tra le variabili.

I risultati mostrano:

- una forte correlazione positiva tra `energy` e `loudness`,
- una forte correlazione negativa tra `energy` e `acousticness`,
- una correlazione moderata tra `danceability` e `valence`,
- feature debolmente correlate come `instrumentalness`, `speechiness` e `tempo`.

Nessuna coppia di feature presenta una correlazione perfetta, rendendo il set di feature adatto a un sistema di raccomandazione basato sul contenuto.

---

## 7. Grafici di Supporto alla Correlazione

Per confermare visivamente i risultati della matrice di correlazione oltre agli scatter plot sono stati utilizzati grafici aggregati più leggibili, tra cui:

- barplot delle medie,
- boxplot,
- istogrammi condizionati.

Questi grafici hanno permesso di:

- confermare le relazioni positive o negative tra le feature,
- evidenziare l’assenza di pattern significativi (ad esempio tra `duration_ms` ed `energy`),
- giustificare l’esclusione di alcune feature dal modello di raccomandazione.

---

## 8. Feature Engineering

Sono state introdotte alcune **feature derivate** per rappresentare concetti musicali di livello più alto:

- `mood_score`: media di `energy` e `valence`, rappresenta il mood complessivo del brano.
- `dance_mood`: combinazione di `danceability` e `valence`.
- `electronic_index`: differenza tra `energy` e `acousticness`, utile per distinguere brani elettronici da brani acustici.
- `is_instrumental`: variabile binaria che identifica i brani strumentali.

Per le feature concettualmente più rilevanti sono stati utilizzati grafici descrittivi per verificarne la distribuzione e l’interpretabilità musicale.

---

## 9. Selezione Finale delle Feature

Sulla base dell’analisi esplorativa e dello studio delle correlazioni, è stato definito un set finale di feature composto da:

- audio feature originali non ridondanti,
- un numero limitato di feature ingegnerizzate.

La feature `duration_ms` è stata esclusa in quanto non mostra relazioni significative con gli altri attributi musicali.

---

## 10. Scaling delle Feature

Come ultimo passo della fase di preparazione dei dati, è stato applicato uno **scaling Min-Max** per riportare tutte le feature selezionate nell’intervallo \([0,1]\).

Questo passaggio è fondamentale per:

- rendere le feature confrontabili,
- evitare che differenze di scala influenzino il calcolo della similarità,
- preparare i dati all’uso in un sistema di raccomandazione basato sulla similarità del contenuto.

---

## 11. Conclusione

La fase di pulizia, analisi ed esplorazione ha prodotto un dataset:

- pulito e coerente,
- interpretabile dal punto di vista musicale,
- privo di forte multicollinearità,
- adatto alla costruzione di un sistema di raccomandazione musicale.
