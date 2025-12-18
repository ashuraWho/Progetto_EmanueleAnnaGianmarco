import pandas as pd
from preprocessing import FINAL_FEATURES, preprocess_dataset

# CONFIGURAZIONE

# In questa sezione definisco
# - quali colonne verranno usate come FEATURE numeriche per il ML
# - quali colonne verranno usate solo per stampare info all'utente

# Feature numeriche usate dal modello (X)
# Uso le feature finali (base + derivate) definite nel modulo di preprocessing
FEATURE_COLUMNS = FINAL_FEATURES

# Colonne informative non usate dal modello
DISPLAY_COLUMNS = [
    "track_name",
    "artists",
    "track_genre"
]

#Creo una funzione per caricare il dataset che mi servirà solo in questo momento prima di avere a disposizione
#la pulizia accurata del dataset

# Carico il dataset CSV e applico la stessa pipeline di pulizia/feature engineering
# definita nel notebook `pulizia.ipynb`, incapsulata in `preprocess_dataset`.
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = preprocess_dataset(df)

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
    
    # Uso sia il genere principale (macro-classe) sia il sottogenere
    main_genre = song.get("main_genre", song.get("track_genre", "N/A"))
    sub_genre = song.get("track_genre")
    print(f"Genere : {main_genre} ({sub_genre})")
    
    # Continuo a chiedere finché l'input non è valido
    while True:
        vote = input("Ti piace? (0 = No, 1 = Sì, 2 = Indifferente, 3 = Forse sì, 4 = Forse no): ")
        # 0 = dislike forte
        # 1 = like forte
        # 2 = indifferente (non usato per addestrare il modello
        # 3 = forse sì (like debole)
        # 4 = forse no (dislike debole)
        
        if vote in ["0", "1", "2", "3", "4"]:
            return int(vote)

        print("Input non valido. Inserisci solo 0, 1, 2, 3 o 4")
        
#Creo una nuova funzione che chiede all'utente il suo artista preferito in modo tale da avere già una lista di canzoni se questo è presente nel dataset
def ask_favorite_artist() -> str | None:
    artist = input(
        "\nHai un artista preferito? (Scrivi il suo nome o premi Invio): "
    )

    # Rimuovo spazi iniziali/finali e normalizzo gli spazi interni
    artist = " ".join(artist.strip().split())

    # Converto tutto in lowercase per confronti case-insensitive
    artist = artist.lower()

    return artist if artist != "" else None


#Ora creo la funzione principale cold_start che mi permette di raccogliere i risultati delle votazioni dell'utente
#(df-> dataset completo, n_songs-> numero di canzoni iniziali da votare)

#Ora creo la funzione principale cold_start che mi permette di raccogliere i risultati delle votazioni dell'utente
#(df-> dataset completo, n_songs-> numero di canzoni iniziali da votare)

def cold_start(df: pd.DataFrame, n_songs: int = 10):

    print("\n--- BENVENUTO NEL TUO AI DJ ---")
    print(f"Vota {n_songs} canzoni casuali o inserisci il tuo artista preferito!!\n")
    
    favorite_artist = ask_favorite_artist()

    # Lista temporanea dove verranno salvate tutte le interazioni con l'utente
    user_history = []

    # Set per tenere traccia delle canzoni già viste
    seen_tracks = set()
    
    if favorite_artist is not None:
    
        artist_songs = df[
            df["artists"].str.lower().str.contains(favorite_artist, na=False)
        ]

        if not artist_songs.empty:
            print(f"\nHo trovato canzoni di {favorite_artist.title()}! Le considero come 'Mi piace'.")

            for _, song in artist_songs.iterrows():
                entry = {
                    "track_id": song["track_id"],
                    "track_name": song["track_name"],
                    "artists": song["artists"],
                    "vote": 1
                }

                for feature in FEATURE_COLUMNS:
                    entry[feature] = song[feature]

                user_history.append(entry)
                seen_tracks.add(song["track_id"])

            #user_history = pd.DataFrame(user_history)

            print("\nHo aggiunto alcune canzoni del tuo artista preferito!")
            print("Ora ti chiederò comunque di votare altre canzoni.\n")

        else:
            print(f"\nNon ho trovato canzoni di {favorite_artist.title()}.")
            print("Ti chiederò più voti iniziali.")
            n_songs = 10

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