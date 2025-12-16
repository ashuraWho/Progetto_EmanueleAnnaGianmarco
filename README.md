# Progetto_EmanueleAnnaGianmarco

Progetto Finale Academy Python &amp; Machine Learning
Partecipanti:

- Emanuele Anzellotti
- Anna Firinu
- Gianmarco Sorrentino

# 📘 README – Moduli sviluppati (Fase A & Fase D)

## 👩‍💻 Autore

Anna F.

## 🎧 Progetto

**The AI DJ – Sistema di Raccomandazione Musicale Interattivo**

---

## 📌 Descrizione generale

Questo progetto simula il funzionamento di un sistema di raccomandazione musicale simile a Spotify, basato su un ciclo di **Active Learning**.
Il sistema parte senza alcuna conoscenza dei gusti dell’utente e apprende progressivamente attraverso l’interazione diretta.

Questo README descrive **le parti di progetto da me sviluppate**, ovvero:

- **Fase A – Cold Start**
- **Fase D – Interazione e Feedback Loop**

---

## 🅰️ Fase A – Cold Start (Avvio a Freddo)

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

- `cold_start.py`

---

## 🅳 Fase D – Interazione e Feedback Loop

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

- `interaction.py`

---

## 🧠 Nota progettuale

La fase di Processing dei dati, Fasi B (Training) e C (Predizione) sono state sviluppate da altri membri del gruppo.
Le mie implementazioni sono **modulari** e progettate per integrarsi facilmente nel loop principale del sistema senza sovrapporsi alle responsabilità degli altri moduli.

---
