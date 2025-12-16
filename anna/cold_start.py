import pandas as pd

# CONFIGURAZIONE

# In questa sezione definisco
# - quali colonne verranno usate come FEATURE numeriche per il ML
# - quali colonne verranno usate solo per stampare info all'utente

# Feature numeriche usate dal modello (X)
FEATURE_COLUMNS = [
    "danceability",        # Quanto è ballabile la canzone
    "energy",              # Intensità / carica
    "speechiness",         # Presenza di parlato
    "acousticness",        # Quanto è acustica
    "instrumentalness",    # Quanto è strumentale
    "liveness",            # Presenza di pubblico
    "valence",             # Quanto è "felice"
    "tempo",               # BPM
    "loudness"             # Volume medio
]

# Colonne informative non usate dal modello
DISPLAY_COLUMNS = [
    "track_name",
    "artists",
    "track_genre"
]

#Creo una funzione per caricare il dataset che mi servirà solo in questo momento prima di avere a disposizione
#la pulizia accurata del dataset

# Carico il dataset CSV e faccio una prima pulizia 
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURE_COLUMNS) #Rimuovo le righe con valori nulli

    return df 

#Creo una nuova funzione che estrae n canzoni casuali dal dataset, escludendo quelle già proposte all'utente
# (df-> dataset completo, seen_tracks-> insieme (set) degli ID già mostrati,n-> numero di canzoni da estrarre)
def sample_songs(df: pd.DataFrame, seen_tracks: set, n: int) -> pd.DataFrame:
    # Filtro il dataset escludendo le canzoni già ascoltate
    available_songs = df[~df["track_id"].isin(seen_tracks)]

    # Estraggo n canzoni casuali
    return available_songs.sample(n=n)


#Creo una nuova funzione che chiede all'utente un voto per ogni singola canzone
def ask_user_vote(song: pd.Series) -> int:
    # Stampo le informazioni principali
    print("\nCanzone:")
    print(f"Titolo : {song['track_name']}")
    print(f"Artista: {song['artists']}")
    print(f"Genere : {song['track_genre']}")
    
    # Continuo a chiedere finché l'input non è valido
    while True:
        vote = input("Ti piace? (1 = Sì, 0 = No): ")

        if vote in ["0", "1"]:
            return int(vote)

        print("Input non valido. Inserisci solo 0 o 1")


