import pandas as pd

from faseA import load_dataset, cold_start, FEATURE_COLUMNS
from fasiBC import train_model, select_next_song, print_feature_importance
from faseD import interaction_step
from graficoFinale import plot_valence_energy_boundary
from wrapped import wrapped_utente, top_artists, top_generi


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

    user_history = None
    state = None
    seen_tracks = set()

    while True:
        print("\n--- MENU ---")
        print("1. Avvia Cold Start")
        print("2. Continua con suggerimenti AI")
        print("3. Mostra grafico finale (Valence vs Energy)")
        print("4. Mostra Wrapped Utente")
        print("0. Esci")

        choice = input("Scelta (0-4): ")

        match choice:
            case "1":
                user_history, seen_tracks = cold_start(df, n_songs=5)

                # Inizializzo lo stato per le fasi B/C
                state = {
                    "user_history": user_history,
                    "interaction_count": len(user_history),
                    "feature_cols": FEATURE_COLUMNS,
                    "meta_cols": ["track_id", "track_name", "artists", "track_genre"],
                    "model": None
                }

                # Mantieni solo le colonne utili nello storico
                cols_to_keep = state["meta_cols"] + state["feature_cols"] + ["vote"]
                state["user_history"] = state["user_history"][cols_to_keep]

                print("\nCold Start completato!")
                print(state["user_history"][["track_name", "artists", "vote"]])
                print("\nNumero canzoni votate:", len(seen_tracks))

            case "2":
                if state is None:
                    print("Devi prima fare il Cold Start!")
                    continue

                while True:
                    print("\n[INFO] Addestramento/aggiornamento modello in corso...")
                    model = train_model(state)

                    if model is None:
                        print("[WARNING] Non ci sono abbastanza dati (almeno un like e un dislike) per addestrare il modello.")
                        break

                    if state.get("model_type") == "rf":
                        print("[INFO] Modello ML (Random Forest) addestrato.")
                        total_interactions = len(state["user_history"])
                        if total_interactions % 10 == 0:
                            print("\n[Interpretabilità] Ho abbastanza dati per dirti cosa sto guardando di più:")
                            print_feature_importance(state, top_k=5)
                    else:
                        print("[INFO] Modello DL (MLP) addestrato.")
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

                    candidate_df = df[~df["track_id"].isin(seen_tracks)]
                    next_song = select_next_song(state, candidate_df)

                    if next_song is None or next_song.empty:
                        print("Non ci sono più canzoni disponibili!")
                        break

                    song_row = next_song.iloc[0]
                    confidence = song_row.get("like_prob", 0.5)

                    state["user_history"], seen_tracks, vote = interaction_step(
                        song_row, confidence, state["user_history"], seen_tracks
                    )

                    
                    cols_to_keep = state["meta_cols"] + state["feature_cols"] + ["vote"]
                    state["user_history"] = state["user_history"][cols_to_keep]

                    if vote is not None:
                        state["interaction_count"] = state.get("interaction_count", 0) + 1

                    if vote is None:
                        break

                    print(f"Canzoni votate totali: {len(seen_tracks)}")

            case "3":
                if state is None or state.get("model") is None or state["user_history"].empty:
                    print("\n[Grafico] Devi prima fare il Cold Start e almeno un ciclo di suggerimenti per addestrare il modello.")
                else:
                    print("\n[Grafico Finale] Mostro il rapporto tra Valence, Energy e decisione del modello...")
                    plot_valence_energy_boundary(state)

            case "4":
                if state is None or state["user_history"] is None or state["user_history"].empty:
                    print("Devi prima fare il Cold Start!")
                    continue
                wrapped_utente(state)
                top_artists(state)
                top_generi(state)

            case "0":
                print("\nUscita dal programma. A presto!")
                break

            case _:
                print("\nScelta non valida. Riprova.")


if __name__ == "__main__":
    main()
