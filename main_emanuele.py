# ================================================================
# ================================================================

# Vado a provare il mio codice implementando la parte di Anna

# ================================================================
# ================================================================

import pandas as pd

from faseA_anna import load_dataset, cold_start, FEATURE_COLUMNS
from fasiBC_emanuele import train_model, select_next_song, print_feature_importance
from faseD_anna import interaction_step


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
                    # Contatore di quante canzoni sono state votate (serve per stampare insight ogni N turni)
                    "interaction_count": len(user_history),
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

                # Loop fino a Invio
                while True:
                    # Addestra/aggiorna modello
                    print("\n[INFO] Addestramento/aggiornamento modello in corso...")
                    model = train_model(state)

                    if model is None:
                        print("[WARNING] Non ci sono abbastanza dati (almeno un like e un dislike) per addestrare il modello.")
                        break

                    if state["model_type"] == "rf":
                        print("[INFO] Modello ML (Random Forest) addestrato.")
                        # Ogni 10 canzoni votate stampo le feature più importanti
                        total_interactions = len(state["user_history"])
                        if total_interactions % 10 == 0:
                            print("\n[Interpretabilità] Ho abbastanza dati per dirti cosa sto guardando di più:")
                            print_feature_importance(state, top_k=5)
                    else:
                        print("[INFO] Modello DL (MLP) addestrato.")
                        # Analisi qualitativa della loss durante i vari ri-addestramenti
                        loss_history = state.get("loss_history", [])
                        if len(loss_history) >= 2:
                            prev_loss = loss_history[-2]
                            curr_loss = loss_history[-1]
                            trend = "diminuita" if curr_loss < prev_loss else "aumentata"
                            print(
                                f"[Interpretabilità] Loss dell'ultimo training: {curr_loss:.4f} "
                                f"(prima era {prev_loss:.4f}, quindi è {trend})."
                            )
                        elif len(loss_history) == 1:
                            print(f"[Interpretabilità] Prima loss osservata del modello MLP: {loss_history[-1]:.4f}.")

                    # Pool di candidate -> esclude già viste
                    candidate_df = df[~df["track_id"].isin(seen_tracks)]

                    next_song = select_next_song(state, candidate_df)

                    if next_song is None or next_song.empty:
                        print("Non ci sono più canzoni disponibili!")
                        break

                    song_row = next_song.iloc[0]

                    # Mostro info, chiedo il voto e aggiorno storico + viste
                    confidence = song_row.get("like_prob", 0.5)
                    state["user_history"], seen_tracks, vote = interaction_step(
                        song_row, confidence, state["user_history"], seen_tracks
                    )

                    # Aggiorno il contatore di interazioni solo se l'utente ha effettivamente dato un voto
                    if vote is not None:
                        state["interaction_count"] = state.get("interaction_count", 0) + 1

                    # Se l'utente preme Invio, esco dal loop
                    if vote is None:
                        break

                    # Log voto
                    print(f"Canzoni sentite totali: {len(seen_tracks)}")


            case "0":
                print("\nUscita dal programma. A presto!")
                break


            case _:
                print("\nScelta non valida. Riprova.")


if __name__ == "__main__":
    main()