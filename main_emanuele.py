# ================================================================
# ================================================================

# Vado a provare il mio codice implementando la parte di Anna

# ================================================================
# ================================================================

import pandas as pd

# Importo le funzioni dal modulo cold_start
from faseA_anna import load_dataset, cold_start, ask_user_vote, FEATURE_COLUMNS, DISPLAY_COLUMNS
from fasiBC import train_model, select_next_song, print_feature_importance


import pandas as pd

# Importo le funzioni dai moduli
from faseA_anna import load_dataset, cold_start, FEATURE_COLUMNS, DISPLAY_COLUMNS
from fasiBC import train_model, select_next_song, print_feature_importance


def main():
    print("\n==============================")
    print("   TEST PROGETTO - AI DJ ")
    print("==============================\n")

    # Percorso del dataset
    dataset_path = "dataset.csv"

    # Caricamento dataset
    print("Caricamento dataset in corso...")
    df = load_dataset(dataset_path)
    print(f"Dataset caricato correttamente ({len(df)} canzoni)\n")

    state = None
    seen_tracks = set()

    while True:
        print("\n--- MENU ---")
        print("1. Avvia Cold Start")
        print("2. Continua con suggerimenti AI")
        print("0. Esci")

        choice = input("Scelta: ")

        match choice:
            case "1":
                user_history, seen_tracks = cold_start(df, n_songs=5)

                # Inizializzo lo stato per le fasi B/C
                state = {
                    "user_history": user_history,
                    "feature_cols": FEATURE_COLUMNS,
                    "meta_cols": ["track_id", "track_name", "artists"],
                    "model": None
                }

                print("\nCold Start completato!")
                print(user_history[["track_name", "artists", "vote"]])
                print("\nNumero canzoni viste:", len(seen_tracks))

            case "2":
                if state is None:
                    print("Devi prima fare il Cold Start!")
                    continue

                # Addestra/aggiorna modello
                print("\n[INFO] Addestramento/aggiornamento modello in corso...")
                model = train_model(state)

                if model is None:
                    print("[WARNING] Non ci sono abbastanza dati (almeno un like e un dislike) per addestrare il modello.")
                    continue

                if state["model_type"] == "rf":
                    print("[INFO] Modello ML (Random Forest) addestrato.")
                    print_feature_importance(state, top_k=5)
                else:
                    print("[INFO] Modello DL (MLP) addestrato.")
                    print("[INFO] DL non ha feature_importance disponibile.")

                # Pool di candidate (esclude già viste)
                candidate_df = df[~df["track_id"].isin(seen_tracks)]

                next_song = select_next_song(state, candidate_df)

                if next_song is None or next_song.empty:
                    print("Non ci sono più canzoni disponibili!")
                    continue

                song_row = next_song.iloc[0]

                # Mostra info e chiedi voto
                vote = ask_user_vote(song_row)

                # Segna la canzone come vista
                seen_tracks.add(song_row["track_id"])
                
                # ------->>>>>>> Qua va aggiunto il voto a user_history <<<<<--------

                # Log voto
                print(f"\nHai votato '{song_row['track_name']}' di {song_row['artists']} con voto = {vote}")
                print(f"Canzoni sentite totali: {len(seen_tracks)}")

            case "0":
                print("\nUscita dal programma. A presto! 👋")
                break

            case _:
                print("Scelta non valida. Riprova.")


if __name__ == "__main__":
    main()