# The AI DJ – Sistema di Raccomandazione Musicale Interattivo

## Autori

Anna Firinu, Emanuele Anzellotti, Gianmarco Sorrentino

---

## Descrizione

Questo progetto simula il funzionamento di un sistema di raccomandazione musicale simile a Spotify, basato su un ciclo di **Active Learning**. Il DJ virtuale parte senza alcuna conoscenza dei gusti dell'utente, imparando i suoi gusti musicali man mano che vengono votate le canzoni: parte da un **cold start**, raccoglie feedback, addestra un modello (ML: Random Forest | DL: MLP) e aggiorna i suggerimenti in tempo reale con logica di **exploration/exploitation**.

---

## Indice

- [Quick Start](#-quick-start)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Tecnologie Utilizzate](#-tecnologie-utilizzate)
- [Pulizia dei Dati e Analisi Esplorativa](#-pulizia-dei-dati-e-analisi-esplorativa)
- [Architettura del Software](#-architettura-del-software)
- [Visualizzazione Grafica](#-visualizzazione-grafica)
- [Note Tecniche](#️-note-tecniche)
- [Conclusione](#-conclusione-e-integrazione-del-sistema)

---

## Quick Start

1. **Avvia il Cold Start** (opzione `1`): vota almeno 5 canzoni iniziali per permettere al sistema di apprendere i tuoi gusti
2. **Ricevi suggerimenti AI** (opzione `2`): il modello propone canzoni basate sui tuoi voti precedenti
   - Premi `1` se ti piace, `0` se non ti piace
   - Premi `Invio` per tornare al menu principale
3. **Visualizza il grafico** (opzione `3`): mostra la decision boundary del modello nello spazio Valence-Energy
4. **Esci** (opzione `0`): termina il programma

---

---

## Pulizia dei Dati e Analisi Esplorativa

**Autori**: Gianmarco Sorrentino, Anna Firinu, Emanuele Anzellotti (pulizia dati) | Anna Firinu (analisi esplorativa - EDA)

### Introduzione

Questa sezione descrive il processo di **pulizia del dataset**, **analisi esplorativa dei dati (EDA)** e **feature engineering** svolto come fase preliminare alla costruzione di un **sistema di raccomandazione musicale basato sul contenuto**, con particolare attenzione al problema del **cold start**.

**Obiettivo**: ottenere un dataset pulito, coerente, interpretabile dal punto di vista musicale e adatto al calcolo di similarità tra brani.

**Processo implementato** (`preprocessing.py` nasce dallimplementazione della preanalisi scritta nel notebook `pulizia.ipynb`):

- Rimozione valori NaN
- Deduplicazione per `track_name/artist/album` (aggregazione con media per feature numeriche)
- Mappatura dei generi in macro-categorie (`main_genre`)
- Aggiunta di feature derivate

**Feature utilizzate**:

- **9 feature base**: `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `loudness`
- **4 feature derivate**: `mood_score`, `electronic_index`, `is_instrumental`, `dance_mood`

Le feature finali sono definite in `FINAL_FEATURES` (`preprocessing.py`).

---

### Caricamento del Dataset e Ispezione Iniziale

Il dataset è stato caricato utilizzando la libreria `pandas` ed è stato inizialmente analizzato tramite:

- `head()` per una prima ispezione dei dati,
- `info()` per verificare i tipi di dato e la presenza di valori mancanti,
- `describe()` per analizzare le statistiche descrittive,
- conteggio dei valori mancanti e dei duplicati.

Questa fase ha permesso di comprendere la struttura generale del dataset e di individuare eventuali criticità.

---

### Pulizia dei Dati

#### Gestione dei Valori Mancanti

I valori mancanti sono stati rimossi utilizzando il metodo `dropna()`.  
Questa scelta è stata ritenuta appropriata data la dimensione del dataset e la necessità di lavorare con feature complete per il calcolo delle similarità.

#### Gestione dei Duplicati

Per evitare la presenza di più istanze della stessa canzone, il dataset è stato aggregato utilizzando:

- `track_name`,
- `artists`,
- `album_name`.

Le feature numeriche sono state aggregate tramite media, mentre le feature categoriali sono state mantenute selezionando il primo valore disponibile.  
Questo garantisce una rappresentazione unica e coerente di ciascun brano.

---

### Analisi dei Generi Musicali

#### Distribuzione dei Generi

È stata analizzata la distribuzione della variabile `track_genre` per individuare eventuali sbilanciamenti nel dataset.

#### Mappatura dei Generi

Per ridurre la dimensionalità e migliorare la robustezza nella fase di cold start, i generi originali sono stati mappati in un insieme ridotto di **macro-generi** (`main_genre`).  
Il genere originale è stato comunque mantenuto come `sub_genre`.

Questa scelta consente:

- raccomandazioni iniziali più stabili,
- una maggiore granularità quando si accumula feedback dell’utente.

---

### Analisi delle Audio Feature

Le principali audio feature (ad esempio `energy`, `danceability`, `valence`, `acousticness`, ecc.) sono state analizzate tramite istogrammi al fine di studiarne:

- la distribuzione dei valori,
- eventuali asimmetrie,
- la presenza di valori estremi o concentrati.

L’analisi ha evidenziato che le feature catturano aspetti musicali differenti e non risultano ridondanti.

---

### Analisi di Correlazione

È stata calcolata una matrice di correlazione tra le audio feature per analizzare le relazioni lineari tra le variabili.

I risultati mostrano:

- una forte correlazione positiva tra `energy` e `loudness`,
- una forte correlazione negativa tra `energy` e `acousticness`,
- una correlazione moderata tra `danceability` e `valence`,
- feature debolmente correlate come `instrumentalness`, `speechiness` e `tempo`.

Nessuna coppia di feature presenta una correlazione perfetta, rendendo il set di feature adatto a un sistema di raccomandazione basato sul contenuto.

---

### Grafici di Supporto alla Correlazione

Per confermare visivamente i risultati della matrice di correlazione oltre agli scatter plot sono stati utilizzati grafici aggregati più leggibili, tra cui:

- barplot delle medie,
- boxplot,
- istogrammi condizionati.

Questi grafici hanno permesso di:

- confermare le relazioni positive o negative tra le feature,
- evidenziare l’assenza di pattern significativi (ad esempio tra `duration_ms` ed `energy`),
- giustificare l’esclusione di alcune feature dal modello di raccomandazione.

---

### Feature Engineering

Sono state introdotte alcune **feature derivate** per rappresentare concetti musicali di livello più alto:

- `mood_score`: media di `energy` e `valence`, rappresenta il mood complessivo del brano.
- `dance_mood`: combinazione di `danceability` e `valence`.
- `electronic_index`: differenza tra `energy` e `acousticness`, utile per distinguere brani elettronici da brani acustici.
- `is_instrumental`: variabile binaria che identifica i brani strumentali.

Per le feature concettualmente più rilevanti sono stati utilizzati grafici descrittivi per verificarne la distribuzione e l’interpretabilità musicale.

---

### Selezione Finale delle Feature

Sulla base dell’analisi esplorativa e dello studio delle correlazioni, è stato definito un set finale di feature composto da:

- audio feature originali non ridondanti,
- un numero limitato di feature ingegnerizzate.

La feature `duration_ms` è stata esclusa in quanto non mostra relazioni significative con gli altri attributi musicali.

---

### Scaling delle Feature

Come ultimo passo della fase di preparazione dei dati, è stato applicato uno **scaling Min-Max** per riportare tutte le feature selezionate nell’intervallo \([0,1]\).

Questo passaggio è fondamentale per:

- rendere le feature confrontabili,
- evitare che differenze di scala influenzino il calcolo della similarità,
- preparare i dati all’uso in un sistema di raccomandazione basato sulla similarità del contenuto.

---

### Conclusione

La fase di pulizia, analisi ed esplorazione ha prodotto un dataset pulito, coerente, interpretabile dal punto di vista musicale, privo di forte multicollinearità e adatto alla costruzione di un sistema di raccomandazione musicale.

## Architettura del Software

### Fase A – Cold Start (Avvio a Freddo)

**Autore**: Anna Firinu

### Obiettivo

Gestire l’avvio del sistema quando il modello non ha ancora informazioni sui gusti dell’utente, raccogliendo le prime etichette necessarie per il training iniziale.

---

### Funzionalità implementate

- Caricamento del dataset musicale da file CSV
- Pulizia **temporanea e minimale** dei dati (rimozione valori nulli nelle feature numeriche),  
  effettuata **in attesa dell’integrazione del modulo dedicato al preprocessing dei dati**
- Estrazione casuale di **N canzoni** dal dataset
- Visualizzazione di:
  - Titolo
  - Artista
  - Genere
- Raccolta del voto dell’utente:
  - `1`  ->  Mi piace
  - `0`  ->  Non mi piace
- Creazione dello storico utente (`user_history`)
- Tracciamento delle canzoni già ascoltate (`seen_tracks`) per evitare ripetizioni

---

### Output della Fase A

- `user_history`  ->  DataFrame contenente:
  - Feature audio numeriche
  - Voto dell’utente
  - Metadati (titolo, artista)
- `seen_tracks`  ->  insieme di `track_id` già valutati

Questi output costituiscono il **dataset di training iniziale** per le fasi successive del progetto.

---

### File coinvolti

- `faseA.py`

---

### Fasi B e C – Training e Active Learning

**Autore**: Emanuele Anzellotti

#### Obiettivo

Le fasi B e C del progetto hanno l’obiettivo di creare un modello di classificazione binaria che impari dai feedback dell’utente e predica la probabilità che un brano venga apprezzato:

$$P(\text{“mi piace”}=1 \mid \text{feature audio})$$

Il sistema è progettato per apprendere in tempo reale, aggiornando il modello dopo ogni voto e suggerendo brani con confidenza crescente o, alternativamente, esplorando quelli più incerti.

#### Fase B – Training del Modello

1. **Input**: feedback dell'utente (`user_history`) con feature audio numeriche e label 0/1
2. **Condizione pre-addestramento**: il modello viene costruito solo se ci sono almeno due classi presenti (almeno un like e un dislike)
3. **Scelta automatica del modello**:
   - **Random Forest (RF)**: usato di default per dataset piccoli (< 80 voti o sbilanciamento)
   - **MLP (Multi-Layer Perceptron)**: abilitato automaticamente quando ci sono almeno 80 voti totali e almeno 15 like e 15 dislike
4. **Pipeline**: tutte le feature vengono scalate con `MinMaxScaler` per uniformità tra RF e MLP
5. **Training incrementale**: ogni volta che arriva un nuovo feedback, il modello viene riaddestrato per incorporare la nuova informazione
6. **Output**: pipeline addestrata salvata in `state["model"]`

#### Fase C – Predizione & Active Learning

1. **Input**: pool di brani non ancora ascoltati (`candidate_df`) e modello addestrato
2. **Predizione**: il modello calcola la probabilità di like per ciascun brano
3. **Exploration/Exploitation**:
   - **Exploitation (70%)**: scegliere la canzone con probabilità di like più alta
   - **Exploration (30%)**: scegliere la canzone con probabilità più vicina a 0.5, dove il modello è più incerto
4. **Motivazione**: proporre brani incerti permette al modello di imparare più velocemente identificando i casi borderline

### File coinvolti

- `faseBC.py`

### Fase D – Interazione e Feedback Loop

**Autore**: Anna Firinu

### Obiettivo

Gestire l’interazione tra utente e sistema dopo che il modello è stato addestrato, chiudendo il ciclo di Active Learning.

---

### Funzionalità implementate

- Visualizzazione della canzone raccomandata dal modello
- Stampa della **probabilità stimata di gradimento**
- Raccolta del verdetto reale dell’utente (1 / 0)
- Aggiornamento dinamico dello storico utente (`user_history`)
- Aggiornamento delle canzoni già ascoltate (`seen_tracks`)
- Preparazione dei dati per il ri-addestramento del modello

---

### Feedback Loop

Ogni nuova interazione:

1. Viene salvata nello storico utente
2. Arricchisce il dataset di training
3. Permette al modello di migliorare progressivamente le raccomandazioni

Questo meccanismo realizza un ciclo di **Apprendimento Attivo (Active Learning)**.

---

### File coinvolti

- `faseD.py`

## Visualizzazione Grafica

**Autore**: Emanuele Anzellotti

Il sistema include una funzionalità di visualizzazione avanzata (`graficoFinale.py`) che mostra:

- **Spazio Valence-Energy**: rappresentazione bidimensionale dei brani
- **Decision Boundary**: 
  - Linea nera: soglia di decisione (p = 0.5)
  - Linee grigie: soglie intermedie (p = 0.3 e p = 0.7)
- **Mappa di probabilità**: sfondo colorato (gradiente rosso -> giallo -> verde) che indica la probabilità di like
- **Feedback utente**: punti verdi (like) e rossi (dislike)

**Utilizzo**: Completa il Cold Start e almeno un ciclo di suggerimenti, poi seleziona l'opzione `3` dal menu principale.

---

## Note Tecniche

### Requisiti per il Training

- Il modello richiede **almeno un like e un dislike** per essere addestrato
- Se non ci sono abbastanza dati, ri-seleziona l'opzione `1` dal menu principale

### Interpretabilità

- **Random Forest**: Ogni 10 voti, vengono mostrate le top 5 feature più importanti
- **MLP**: Viene mostrato l'andamento della loss durante i vari ri-addestramenti

### Feature Utilizzate

Il sistema utilizza **13 feature** totali:
- **9 feature base**: `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `loudness`
- **4 feature derivate**: `mood_score`, `electronic_index`, `is_instrumental`, `dance_mood`

Tutte le feature vengono scalate con `MinMaxScaler` nell'intervallo [0, 1] prima dell'addestramento.

---

## Conclusione e Integrazione del Sistema

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

- **Emanuele Anzellotti** si è inoltre occupato del **collegamento finale dell’intero sistema al dataset pulito, analizzato e preprocessato**, assicurando che:
  - le feature selezionate e ingegnerizzate venissero utilizzate correttamente,
  - lo scaling fosse coerente tra preprocessing ed esecuzione del modello,
  - il sistema di raccomandazione operasse su dati consistenti e affidabili.

Grazie a questa integrazione, il progetto non si limita a una collezione di moduli separati,
ma realizza un **sistema completo, modulare e interattivo**, capace di apprendere
progressivamente dai feedback dell’utente attraverso un ciclo di **Active Learning**.
