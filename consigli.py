# ================================================================================

# In questo modulo definisco una funzione per fornire consigli personalizzati
# basati sulle canzoni che l'utente ha votato positivamente durante l'interazione.

# ================================================================================

import pandas as pd
from typing import Dict, Set

from fasiBC import train_model

# ----------------------------- FUNZIONE PRINCIPALE ----------------------------------

"""
    Stampa alcuni consigli personalizzati:
      - fino a 5 artisti raccomandati
      - fino a 5 generi principali (macro-generi)
      - fino a 5 sottogeneri (track_genre)

    Basato esclusivamente sulle canzoni che sono state votate in modo positivo
    (like forte = 1, forse sì = 3).
"""
    
def consigli_personalizzati(df: pd.DataFrame, state: Dict, seen_tracks: Set[str], n_artists: int = 5, n_genres: int = 5):

    """
    Consigli basati sul modello ML/DL:
      - suggerisco fino a 5 ARTISTI che non hai ancora votato
      - suggerisco fino a 5 GENERI principali che non hai ancora ascoltato
      - suggerisco fino a 5 SOTTOGENERI che non hai ancora ascoltato

    Usa le probabilità di "like" stimate dal modello sulle canzoni non ancora votate.
    """

    if state is None:
        print("\n[Consigli] Devi prima fare il Cold Start!")
        return

    user_history = state.get("user_history")
    if user_history is None or user_history.empty:
        print("\n[Consigli] Nessuna interazione trovata, vota prima qualche canzone.")
        return

    # Assicuro che esista un modello addestrato
    model = state.get("model")
    if model is None:
        print("\n[Consigli] Addestro il modello sui tuoi voti...")
        model = train_model(state)
        if model is None:
            print(
                "\n[Consigli] Non ho abbastanza dati per stimare le tue preferenze.\n"
                "Vota almeno una canzone che ti piace e una che non ti piace."
            )
            return

    feature_cols = state["feature_cols"]

    # Le canzoni candidate sono quelle che non hai ancora visto/votato
    if seen_tracks:
        candidate_df = df[~df["track_id"].isin(seen_tracks)].copy()
    else:
        candidate_df = df.copy()

    if candidate_df.empty:
        print("\n[Consigli] Non ci sono più canzoni nuove da suggerire.")
        return

    # Probabilità di like per ogni canzone candidata
    probs = model.predict_proba(candidate_df[feature_cols])[:, 1]
    candidate_df = candidate_df.copy()
    candidate_df["like_prob"] = probs

    # -----------------------
    
    # Informazioni dallo storico utente
    history = user_history.copy()

    # Artisti già ascoltati
    if "artists" in history.columns:
        history["artists"] = history["artists"].astype(str).str.strip()
        heard_artists = set(history["artists"].dropna().unique())
    else:
        heard_artists = set()

    # Generi già ascoltati
    heard_main = set()
    heard_sub = set()
    if "main_genre" in history.columns:
        heard_main = set(history["main_genre"].dropna().astype(str))
    if "sub_genre" in history.columns:
        heard_sub = set(history["sub_genre"].dropna().astype(str))

    # Pulizia colonne candidate
    candidate_df["artists"] = candidate_df["artists"].astype(str).str.strip()
    if "main_genre" in candidate_df.columns:
        candidate_df["main_genre"] = candidate_df["main_genre"].fillna("N/A").astype(str)
    else:
        candidate_df["main_genre"] = "N/A"

    if "sub_genre" in candidate_df.columns:
        candidate_df["sub_genre"] = candidate_df["sub_genre"].fillna("N/A").astype(str)
    else:
        candidate_df["sub_genre"] = "N/A"

    # -----------------------

    # ARTISTI NUOVI
    new_artist_mask = ~candidate_df["artists"].isin(heard_artists)
    candidates_new_artists = candidate_df[new_artist_mask]
    if candidates_new_artists.empty:
        # Se non ho artisti completamente nuovi, considero comunque tutti
        candidates_new_artists = candidate_df

    artist_scores = (
        candidates_new_artists
        .groupby("artists")["like_prob"]
        .max()
        .sort_values(ascending=False)
        .head(n_artists)
    )

    # -----------------------

    # MAIN GENRE NUOVI
    new_main_mask = ~candidate_df["main_genre"].isin(heard_main)
    candidates_new_main = candidate_df[new_main_mask]
    if candidates_new_main.empty:
        candidates_new_main = candidate_df

    main_genre_scores = (
        candidates_new_main
        .groupby("main_genre")["like_prob"]
        .mean()
        .sort_values(ascending=False)
        .head(n_genres)
    )

    # -----------------------

    # SUB GENRE NUOVI
    new_sub_mask = ~candidate_df["sub_genre"].isin(heard_sub)
    candidates_new_sub = candidate_df[new_sub_mask]
    if candidates_new_sub.empty:
        candidates_new_sub = candidate_df

    sub_genre_scores = (
        candidates_new_sub
        .groupby("sub_genre")["like_prob"]
        .mean()
        .sort_values(ascending=False)
        .head(n_genres)
    )

    # -----------------------

    # STAMPA 
    print("\n--- CONSIGLI PERSONALIZZATI ---")

    # Artisti consigliati
    if not artist_scores.empty:
        print("Artisti nuovi che probabilmente ti piaceranno:")
        for i, (artist, score) in enumerate(artist_scores.items(), start=1):
            print(f"({i}) {artist}  [prob. like circa {score:.2f}]")
    else:
        print("Nessun artista disponibile per i consigli.")

    # Generi principali
    if not main_genre_scores.empty:
        print("\nGeneri principali che non hai ancora esplorato (o poco esplorati):")
        for i, (genre, score) in enumerate(main_genre_scores.items(), start=1):
            print(f"({i}) {genre}  [prob. media like circa {score:.2f}]")
    else:
        print("\nNessun genere principale disponibile per i consigli.")

    # Sottogeneri
    if not sub_genre_scores.empty:
        print("\nSottogeneri che non hai ancora esplorato (o poco esplorati):")
        for i, (sub, score) in enumerate(sub_genre_scores.items(), start=1):
            print(f"({i}) {sub}  [prob. media like circa {score:.2f}]")
    else:
        print("\nNessun sottogenere disponibile per i consigli.")