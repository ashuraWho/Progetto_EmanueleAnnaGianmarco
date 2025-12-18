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
def ask_favorite_artist(df: pd.DataFrame) -> str | None:
    """
    Chiede all'utente un artista preferito e gestisce i casi di ambiguità:
    - se il pattern inserito non matcha nessun artista -> ritorna None
    - se matcha un solo artista -> lo prende direttamente
    - se matcha più artisti diversi -> fa scegliere all'utente quale intendeva
    - se scrivi "tutti" -> seleziona tutti gli artisti trovati
    """
    artist_input = input(
        "\nHai un artista preferito? (Scrivi il suo nome o premi Invio): "
    )

    # Rimuovo spazi iniziali/finali e normalizzo gli spazi interni
    query = " ".join(artist_input.strip().split()).lower()

    if query == "":
        return None

    # Trovo gli artisti distinti che contengono il pattern inserito
    mask = df["artists"].str.lower().str.contains(query, na=False)
    matched_artists = (
        df.loc[mask, "artists"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(matched_artists) == 0:
        print(f"\nNon ho trovato nessun artista che contenga '{query}'.")
        return None

    if len(matched_artists) == 1:
        selected = matched_artists[0]
        print(f"\nHo trovato l'artista: {selected}")
        return selected.lower()

    # Più artisti trovati: faccio scegliere all'utente
    print("\nHo trovato più artisti che corrispondono alla tua ricerca:")
    for idx, name in enumerate(matched_artists, start=1):
        print(f"{idx}. {name}")

    while True:
        choice = input(
            "Quale intendevi? (inserisci il numero corrispondente, 'tutti' per selezionarli tutti, oppure premi Invio per annullare): "
        )
        
        choice_clean = choice.strip().lower()

        if choice_clean == "":
            print("Nessun artista selezionato, procederò senza artista preferito.")
            return None

        if choice_clean in ["tutti", "all"]:
            print("Hai scelto tutti gli artisti trovati.")
            all_selected = [str(name).lower() for name in matched_artists]
            return all_selected

        if not choice_clean.isdigit():
            print("Per favore inserisci solo un numero valido oppure 'tutti'.")
            continue

        idx = int(choice_clean)
        if 1 <= idx <= len(matched_artists):
            selected = matched_artists[idx - 1]
            print(f"Hai selezionato: {selected}")
            return str(selected).lower()

        print("Numero non valido. Riprova.")


#Ora creo la funzione principale cold_start che mi permette di raccogliere i risultati delle votazioni dell'utente
#(df-> dataset completo, n_songs-> numero di canzoni iniziali da votare)

#Ora creo la funzione principale cold_start che mi permette di raccogliere i risultati delle votazioni dell'utente
#(df-> dataset completo, n_songs-> numero di canzoni iniziali da votare)

def cold_start(df: pd.DataFrame, n_songs: int):

    print("\n--- BENVENUTO NEL TUO AI DJ ---")
    print(f"Vota {n_songs} canzoni casuali o inserisci il tuo artista preferito!!\n")

    favorite_artist = ask_favorite_artist(df)

    # Lista temporanea dove verranno salvate tutte le interazioni con l'utente
    user_history = []

    # Set per tenere traccia delle canzoni già viste
    seen_tracks = set()
    
    if favorite_artist is not None:

        # Costruisco la maschera a seconda che l'utente abbia scelto uno o più artisti
        if isinstance(favorite_artist, list):
            artist_names = [name.lower() for name in favorite_artist]
            artist_mask = df["artists"].str.lower().isin(artist_names)
            pretty_names = ", ".join(sorted({name.title() for name in artist_names}))
        else:
            artist_mask = df["artists"].str.lower() == favorite_artist
            pretty_names = favorite_artist.title()

        artist_songs = df[artist_mask]

        if not artist_songs.empty:
            print(f"\nHo trovato canzoni di {pretty_names}! Le considero come 'Mi piace'.")

            for _, song in artist_songs.iterrows():
                entry = {
                    "track_id": song["track_id"],
                    "track_name": song["track_name"],
                    "artists": song["artists"],
                    "track_genre": song["track_genre"],
                    "vote": 1
                }

                for feature in FEATURE_COLUMNS:
                    entry[feature] = song[feature]

                user_history.append(entry)
                seen_tracks.add(song["track_id"])

            #user_history = pd.DataFrame(user_history)

            print("\nHo aggiunto alcune canzoni del tuo artista preferito!")
            print("Ora ti chiederò comunque di votare altre canzoni.\n")
            n_songs = 5

        else:
            print(f"\nNon ho trovato canzoni di {pretty_names}.")
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
            "track_genre": song["track_genre"],
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