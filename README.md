# Progetto_EmanueleAnnaGianmarco

## 👩‍💻 Autore

Anna Firinu, Emanuele Anzellotti, Gianmarco Sorrentino

## 🎧 Progetto

**The AI DJ – Sistema di Raccomandazione Musicale Interattivo**

---

## 📌 Descrizione generale

Questo progetto simula il funzionamento di un sistema di raccomandazione musicale simile a Spotify, basato su un ciclo di **Active Learning**.
Il sistema parte senza alcuna conoscenza dei gusti dell’utente e apprende progressivamente attraverso l’interazione diretta.

# Pulizia dei Dati (Gianmarco Sorrentino, Anna Firinu, Emanuele Anzellotti) e Analisi Esplorativa (Anna Firinu)

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

# 📘 Architettura del software (Fase A & Fase B & Fase C & Fase D)

## 🅰️ Fase A – Cold Start (Avvio a Freddo) (Firinu Anna)

### 🎯 Obiettivo

Gestire l’avvio del sistema quando il modello non ha ancora informazioni sui gusti dell’utente, raccogliendo le prime etichette necessarie per il training iniziale.

---

### ⚙️ Funzionalità implementate

- Caricamento del dataset musicale da file CSV
- Pulizia **temporanea e minimale** dei dati (rimozione valori nulli nelle feature numeriche),  
  effettuata **in attesa dell’integrazione del modulo dedicato al preprocessing dei dati**
- Estrazione casuale di **N canzoni** dal dataset
- Visualizzazione di:
  - Titolo
  - Artista
  - Genere
- Raccolta del voto dell’utente:
  - `1` → Mi piace
  - `0` → Non mi piace
- Creazione dello storico utente (`user_history`)
- Tracciamento delle canzoni già ascoltate (`seen_tracks`) per evitare ripetizioni

---

### 📊 Output della Fase A

- `user_history` → DataFrame contenente:
  - Feature audio numeriche
  - Voto dell’utente
  - Metadati (titolo, artista)
- `seen_tracks` → insieme di `track_id` già valutati

Questi output costituiscono il **dataset di training iniziale** per le fasi successive del progetto.

---

### 📁 File coinvolti

- `faseA.py`

---

# Fasi B e C (Emanuele Anzellotti)

## Obiettivo

Le fasi B e C del progetto hanno l’obiettivo di creare un modello di classificazione binaria che impari dai feedback dell’utente e predica la probabilità che un brano venga apprezzato:

$$P(\text{“mi piace”}=1 \mid \text{feature audio})$$

Il sistema è progettato per apprendere in tempo reale, aggiornando il modello dopo ogni voto e suggerendo brani con confidenza crescente o, alternativamente, esplorando quelli più incerti.

## Fase B – Training del modello

1. Input: feedback dell’utente (user_history) con feature audio numeriche e label 0/1.
2. Condizione pre-addestramento: il modello viene costruito solo se ci sono almeno due classi presenti (almeno un like e un dislike).
3. Scelta del modello:
   - Random Forest (RF) -> ideale per dataset piccoli, infatti messo di default.
   - MLP (Multi-Layer Perceptron) -> abilitato dopo 30 voti.
4. Pipeline: tutte le feature vengono scalate con MinMaxScaler per uniformità tra RF e MLP.
5. Training: ogni volta che arriva un nuovo feedback, il modello viene riaddestrato per incorporare la nuova informazione.
6. Output: pipeline addestrata salvata nello state["model"].

## Fase C – Predizione & Active Learning

1. Input: pool di brani non ancora ascoltati (candidate_df) e modello addestrato.
2. Predizione: il modello calcola la probabilità di like per ciascun brano.
3. Exploration/Exploitation:
   - Exploitation (70%) -> scegliere la canzone con probabilità di like più alta.
   - Exploration (30%) -> scegliere la canzone con probabilità più vicina a 0.5, dove il modello è più incerto.
4. Motivazione: proporre brani incerti permette al modello di imparare più velocemente.

### 📁 File coinvolti

- `faseBC.py`

## 🅳 Fase D – Interazione e Feedback Loop (Anna Firinu)

### 🎯 Obiettivo

Gestire l’interazione tra utente e sistema dopo che il modello è stato addestrato, chiudendo il ciclo di Active Learning.

---

### ⚙️ Funzionalità implementate

- Visualizzazione della canzone raccomandata dal modello
- Stampa della **probabilità stimata di gradimento**
- Raccolta del verdetto reale dell’utente (1 / 0)
- Aggiornamento dinamico dello storico utente (`user_history`)
- Aggiornamento delle canzoni già ascoltate (`seen_tracks`)
- Preparazione dei dati per il ri-addestramento del modello

---

### 🔁 Feedback Loop

Ogni nuova interazione:

1. Viene salvata nello storico utente
2. Arricchisce il dataset di training
3. Permette al modello di migliorare progressivamente le raccomandazioni

Questo meccanismo realizza un ciclo di **Apprendimento Attivo (Active Learning)**.

---

### 📁 File coinvolti

- `faseD.py`

# 🔚 Conclusione e Integrazione del Sistema

Oltre alle singole fasi descritte, una parte fondamentale del progetto ha riguardato
l’**integrazione complessiva del sistema di raccomandazione** e il collegamento tra
la logica applicativa e i dati preprocessati.

In particolare:

- **Anna Firinu** ed **Emanuele Anzellotti** si sono occupati della **creazione del file main**
  e dell’**orchestrazione delle diverse fasi del progetto** (Fase A, B, C e D),
  garantendo un flusso di esecuzione coerente e continuo.

- Il main gestisce:

  - l’avvio del sistema in modalità cold start,
  - il passaggio progressivo tra raccolta dei feedback, training del modello,
    predizione e interazione con l’utente,
  - la condivisione dello stato tra le varie fasi (storico utente, modello, canzoni viste).

- **Emanuele Anzellotti** si è inoltre occupato del **collegamento finale dell’intero sistema
  al dataset pulito, analizzato e preprocessato**, assicurando che:
  - le feature selezionate e ingegnerizzate venissero utilizzate correttamente,
  - lo scaling fosse coerente tra preprocessing ed esecuzione del modello,
  - il sistema di raccomandazione operasse su dati consistenti e affidabili.

Grazie a questa integrazione, il progetto non si limita a una collezione di moduli separati,
ma realizza un **sistema completo, modulare e interattivo**, capace di apprendere
progressivamente dai feedback dell’utente attraverso un ciclo di **Active Learning**.
