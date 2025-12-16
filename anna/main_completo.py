# ================================================================
# ================================================================

# Vado a provare il mio codice implementando la parte di Anna

# ================================================================
# ================================================================

import pandas as pd

# Importo le funzioni dal modulo cold_start
from cold_start_file import load_dataset, cold_start, FEATURE_COLUMNS, DISPLAY_COLUMNS
from fasiBC import train_model, select_next_song, print_feature_importance


import pandas as pd

# Importo le funzioni dai moduli
from cold_start_file import load_dataset, cold_start, FEATURE_COLUMNS, DISPLAY_COLUMNS
from fasiBC import train_model, select_next_song, print_feature_importance
from interaction import interaction_step


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

    
                  # FASE B – TRAINING
                  train_model(state)

    
                  # FASE C – SELEZIONE BRANO
                  candidate_df = df[~df["track_id"].isin(seen_tracks)]
                  next_song = select_next_song(state, candidate_df)

                  if next_song is None or next_song.empty:
                     print("Non ci sono più canzoni disponibili!")
                     continue

                  song_row = next_song.iloc[0]

                 # Recupero la probabilità di like
                  if state["model"] is not None:
                     confidence = state["model"].predict_proba(
                      song_row[state["feature_cols"]].to_frame().T
                        )[0][1]
                  else:
                    confidence = 0.5  # caso limite (non dovrebbe succedere qui)

   
                 # FASE D – INTERAZIONE 
    
                    updated_history, seen_tracks = interaction_step(
                    song=song_row,
                    confidence=confidence,
                    user_history=state["user_history"],
                    seen_tracks=seen_tracks
                                  )

                 # Aggiorno lo stato globale
                    state["user_history"] = updated_history

   
                  # (OPZIONALE) FEATURE IMPORTANCE
                    print_feature_importance(state)


            case "0":
                print("\nUscita dal programma. A presto! 👋")
                break

            case _:
                print("Scelta non valida. Riprova.")


if __name__ == "__main__":
    main()