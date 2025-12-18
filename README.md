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
  - [Fase A – Cold Start (Avvio a Freddo)](#fase-a--cold-start-avvio-a-freddo)
  - [Fasi B e C – Training e Active Learning](#fasi-b-e-c--training-e-active-learning)
  - [Fase D – Interazione e Feedback Loop](#fase-d--interazione-e-feedback-loop)
- [Visualizzazione Grafica](#-visualizzazione-grafica)
- [Creazione funzione Wrapped](#-creazione-funzione-wrapped)
- [Consigli Personalizzati AI](#-consigli-personalizzati-ai)
- [Note Tecniche](#️-note-tecniche)
- [Conclusione e Integrazione del Sistema](#-conclusione-e-integrazione-del-sistema)

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

Raccogliere i primi feedback dell’utente in assenza di un modello addestrato, costruendo uno storico iniziale (`user_history`) che verrà utilizzato nelle successive fasi di training e raccomandazione.

### Obiettivi principali

- Avviare l’interazione con l’utente quando il sistema non possiede alcuna informazione sui suoi gusti.
- Raccogliere voti espliciti su un insieme iniziale di canzoni.
- Gestire un Cold Start “guidato” attraverso la richiesta opzionale di un artista preferito.
- Salvare tutte le interazioni in una struttura dati coerente e riutilizzabile dal modello.

---

### Caricamento e preprocessing del dataset

Il dataset musicale viene caricato da file CSV e sottoposto alla **stessa pipeline di preprocessing e feature engineering** definita nel notebook di analisi preliminare (`pulizia.ipynb`), incapsulata nella funzione `preprocess_dataset`.

Questa pipeline include:

- rimozione dei valori mancanti;
- deduplicazione semantica per `(track_name, artists)`;
- mappatura dei generi musicali in macro-classi;
- creazione di feature derivate (es. `mood_score`, `dance_mood`, `electronic_index`, `is_instrumental`).

Il risultato è un DataFrame pronto per essere utilizzato in tutte le fasi successive del progetto.

---

### Raccolta dell’artista preferito (opzionale)

Prima di proporre canzoni casuali, il sistema chiede all’utente se ha un **artista preferito**.

La funzione `ask_favorite_artist`:

- normalizza l’input dell’utente (case-insensitive, gestione degli spazi);
- cerca nel dataset tutti gli artisti che contengono il pattern inserito;
- gestisce tre casi:
  - **nessuna corrispondenza** -> il Cold Start prosegue in modalità standard;
  - **una sola corrispondenza** -> l’artista viene selezionato automaticamente;
  - **più corrispondenze** -> l’utente può scegliere uno o più artisti, oppure selezionarli tutti.

Se vengono trovate canzoni dell’artista (o degli artisti) selezionato/i:

- tutte le relative canzoni vengono aggiunte allo `user_history` con voto positivo (`vote = 1`);
- le canzoni vengono segnate come già viste per evitare duplicazioni successive.

In ogni caso, **anche se viene inserito un artista preferito**, il sistema continua comunque a chiedere all’utente di votare un numero fisso di canzoni casuali, così da ottenere feedback espliciti e non solo impliciti.

---

### Votazione delle canzoni

Durante il Cold Start, all’utente vengono proposte canzoni casuali non ancora viste.

Per ogni canzone vengono mostrate:

- titolo;
- artista;
- genere musicale (macro-genere e sottogenere).

L’utente può esprimere un voto su una scala discreta:

- `0` -> Non mi piace (dislike forte)
- `1` -> Mi piace (like forte)
- `2` -> Indifferente
- `3` -> Forse sì (like debole)
- `4` -> Forse no (dislike debole)

Tutti i voti vengono salvati nello `user_history` insieme alle feature numeriche del brano.

---

### Output della Fase A

Al termine del Cold Start, la funzione restituisce:

- `user_history`: DataFrame contenente lo storico delle interazioni utente (brano, voto, feature audio);
- `seen_tracks`: insieme (`set`) degli identificativi dei brani già mostrati all’utente.

Queste strutture costituiscono l’input delle **Fasi B, C e D**, dove il modello viene addestrato, utilizzato per la raccomandazione e aggiornato in un ciclo di Active Learning.

---

### Considerazioni progettuali

- La Fase A non utilizza alcun modello di Machine Learning.
- Tutte le decisioni prese in questa fase hanno lo scopo di migliorare la qualità e la velocità di apprendimento del sistema nelle fasi successive.
- La gestione dell’artista preferito è progettata per essere robusta rispetto ad ambiguità e input parziali dell’utente.

### File coinvolti

- `faseA.py`

---

### Fasi B e C – Training e Active Learning

**Autore**: Emanuele Anzellotti

#### Obiettivo

Le fasi B e C del progetto hanno l’obiettivo di creare un modello che impari dai feedback dell’utente e predica la probabilità che un brano venga apprezzato:

$$P(\text{“mi piace”}=1 \mid \text{feature audio})$$

Il sistema è progettato per apprendere in tempo reale, aggiornando il modello dopo ogni voto e suggerendo brani con confidenza crescente o, alternativamente, esplorando quelli più incerti.

#### Fase B – Training del Modello

Considero per l'addestramento solo i voti informativi:
- 0 = dislike forte -> peso 1
- 1 = like forte -> peso 1
- 3 = forse sì  (like debole) -> peso 0.5
- 4 = forse no  (dislike debole) -> peso 0.5
- I voti 2 = indifferente vengono ignorati (nessun effetto sul modello).

1. **Input**: feedback dell'utente (`user_history`) con feature audio numeriche e label
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

visua
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
- **Feedback utente**: punti verdi (like), rossi (dislike), verde chiaro (forse sì), rosso/arancione (forse no), grigi (indifferente)

**Utilizzo**: Completa il Cold Start e almeno un ciclo di suggerimenti, poi seleziona l'opzione `3` dal menu principale.

## Creazione funzione Wrapped

**Autore**: Gianmarco Sorrentino

Il sistema include una funzione "Wrapped" che offre all'utente la possibilità di visualizzare un riepilogo delle proprie preferenze musicali.
Dopo aver votato l'utente può visualizzare

- **🎼 Top 3 generi preferiti**: calcolati in base ai brani votati positivamente.
- **🎤 Top 3 artisti preferiti**: gli artisti più ricorrenti tra i brani apprezzati.
- **💿 Statistiche audio**: medie di alcune feature musicali come valence, energy e mood_score, che aiutano a capire il tipo di atmosfera musicale preferita.

## Consigli Personalizzati AI

**Autore**: Emanuele Anzellotti

Questo modulo implementa il livello finale di raccomandazione personalizzata ad alto livello, sfruttando le probabilità di gradimento stimate dal modello ML/DL addestrato durante le Fasi B e C.
A differenza della raccomandazione “brano singolo” utilizzata durante l’Active Learning, questo modulo fornisce una visione aggregata delle preferenze dell’utente, suggerendo nuovi artisti e generi potenzialmente affini ai suoi gusti.

- **Obiettivo**: Stimare la probabilità che un utente apprezzi nuovi contenuti musicali e suggerire: artisti non ancora ascoltati, generi principali e sottogeneri; il tutto basandosi su un modello di Machine Learning/Deep Learning addestrato sui feedback dell’utente.
- **Stampa**: Il modulo stampa a schermo: fino a 5 artisti consigliati, fino a 5 generi principali, fino a 5 sottogeneri; ogni suggerimento è accompagnato da una stima della probabilità di apprezzamento, rendendo il consiglio interpretabile e trasparente.

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
