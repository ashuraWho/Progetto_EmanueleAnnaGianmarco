import pandas as pd
from preprocessing import FINAL_FEATURES

# Importo le stesse feature usate nella Fase A (base + derivate)
FEATURE_COLUMNS = FINAL_FEATURES


# Creo la funzione interaction_step che mostra la canzone suggerita, chiede il voto reale all'utente, aggiorna user_history e seen_tracks
#(song:canzone consigliata, confidence:probabilità che piaccia,user_history:DataFrame con lo storico utente, seen_tracks: set con track_id già ascoltati)

def interaction_step(
    song: pd.Series,
    confidence: float,
    user_history: pd.DataFrame,
    seen_tracks: set,
):
    """Mostra il brano consigliato, chiede il voto, aggiorna storico e viste.

    Se l'utente preme solo Invio, esce dal loop di raccomandazioni
    restituendo None come voto.
    """

    # Stampa della canzone consigliata
    print("\nAI DJ")
    print(
        f"Secondo i miei calcoli, questa ti piacerà al "
        f"{confidence * 100:.0f}%:"
    )
    print("\nLa mia canzone suggerita per te è:")
    print(f"Titolo : {song['track_name']}")
    print(f"Artista: {song['artists']}")
    
    # Uso sia il genere principale (macro-classe) sia il sottogenere
    main_genre = song.get("main_genre", song.get("track_genre", "N/A"))
    sub_genre = song.get("track_genre")
    print(f"Genere : {main_genre} ({sub_genre})")

    # Risposta dell'utente su quanto la canzone sia effettivamente azzeccata
    while True:
        vote = input("Ti è piaciuta? (0 = No, 1 = Sì, 2 = Indifferente, 3 = Forse sì, 4 = Forse no, Invio = esci): ")
        if vote == "":
            print("Esco dal loop di raccomandazioni.")
            return user_history, seen_tracks, None
        if vote in ["0", "1", "2", "3", "4"]:
            vote = int(vote)
            break
        print("Inserisci solo 0, 1, 2, 3, 4 oppure premi Invio per uscire.")

    # Creo una nuova riga da aggiungere a user_history
    new_entry = {
        "track_id": song["track_id"],
        "track_name": song["track_name"],
        "artists": song["artists"],
        "vote": vote
    }

    # Aggiungo le feature numeriche
    for feature in FEATURE_COLUMNS:
        new_entry[feature] = song[feature]

    # Append al DataFrame
    user_history = pd.concat(
        [user_history, pd.DataFrame([new_entry])],
        ignore_index=True
    )

    # Segno la canzone come già ascoltata
    seen_tracks.add(song["track_id"])

    print("Fantastico! Ho imparato qualcosa di nuovo")
    print("Sto ricalcolando il prossimo brano...")

    # Restituisco tutto per il loop successivo
    return user_history, seen_tracks, vote