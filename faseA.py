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

    # Estraggo n canzoni casuali (se ne restano meno di n, prendo tutto)
    n_sample = min(n, len(available_songs))
    return available_songs.sample(n=n_sample) if n_sample > 0 else available_songs.head(0)


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


#Ora creo la funzione principale cold_start che mi permette di raccogliere i risultati delle votazioni dell'utente
#(df-> dataset completo, n_songs-> numero di canzoni iniziali da votare)

def cold_start(df: pd.DataFrame, n_songs: int = 5):

    print("\n--- 🎧 BENVENUTO NEL TUO AI DJ ---")
    print(f"Vota {n_songs} canzoni casuali\n")

    # Lista temporanea dove verranno salvate tutte le interazioni con l'utente
    user_history = []

    # Set per tenere traccia delle canzoni già viste
    seen_tracks = set()

    # Estraggo le canzoni iniziali
    sampled_songs = sample_songs(df, seen_tracks, n_songs)

    # Ciclo sulle canzoni estratte
    for i, (_, song) in enumerate(sampled_songs.iterrows(), start=1):

        print(f"\n[Voto {i}/{n_songs}]")

        # Chiedo il voto all'utente
        vote = ask_user_vote(song)

        # Segno la canzone come già poposta
        seen_tracks.add(song["track_id"])

        # Creo i dati da salvare
        entry = {
            "track_id": song["track_id"],
            "track_name": song["track_name"],
            "artists": song["artists"],
            "vote": vote
        }

        # Aggiungo tutte le feature numeriche
        for feature in FEATURE_COLUMNS:
            entry[feature] = song[feature]

        # Salvo
        user_history.append(entry)

    # Converto la lista di dizionari in DataFrame
    user_history = pd.DataFrame(user_history)

    print("\nCold Start completato!")
    print(user_history[["track_name", "artists", "vote"]])

    return user_history, seen_tracks #Restituisce:user_history per il training del modello, seen_tracks per evitare ripetizioni