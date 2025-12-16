# Progetto_EmanueleAnnaGianmarco

Progetto Finale Academy Python &amp; Machine Learning
Partecipanti:

- Emanuele Anzellotti
- Anna Firinu
- Gianmarco Sorrentino

-GIANMARCO SORRENTINO:

📖 Descrizione del progetto
Questo notebook realizza una pipeline di analisi esplorativa e pulizia dati su un dataset musicale (dataset.csv) utilizzando Python, Pandas, Seaborn, Matplotlib e Scikit-learn.

🔎 Cosa è stato fatto
Caricamento e ispezione iniziale

Lettura del dataset con pandas.

Controllo dimensioni (df.shape), prime righe (df.head()), e informazioni generali (df.info()).

Esplorazione delle variabili

Lista dei generi musicali e conteggio frequenze (value_counts()).

Lista degli artisti e conteggio frequenze.

Pulizia del dataset

Verifica duplicati (df.duplicated().sum()).

Verifica valori mancanti (df.isnull().sum()).

Rimozione righe con valori nulli (df.dropna()).

Stampa dimensioni finali del dataset pulito.

Analisi statistica

Calcolo della matrice di correlazione tra variabili numeriche.

Visualizzazione con heatmap per evidenziare relazioni tra feature audio.

Individuazione outlier

Creazione di boxplot per tempo e loudness.

Calcolo dell’IQR e definizione dei limiti accettabili.

Filtraggio e stampa dei brani considerati outlier.

Normalizzazione delle variabili

Utilizzo di MinMaxScaler per ridimensionare tempo, loudness e duration_ms su scala 0–1.

Controllo dei valori trasformati.

Visualizzazioni

Bar chart: Top 10 artisti più presenti nel dataset.

Scatterplot: distribuzione di loudness per genere musicale.

Scatterplot: distribuzione di tempo per genere musicale.

Linea di riferimento: media della loudness con axhline.
