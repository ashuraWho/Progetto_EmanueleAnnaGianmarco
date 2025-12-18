def wrapped_utente(state):
    if state is None or state["user_history"].empty:
        print("\n[Profilo] Occorre prima votare alcune canzoni!")
        return

    if "vote" not in state["user_history"].columns:
        print("\n[Profilo] Colonna 'vote' mancante nello storico.")
        return

    liked = state["user_history"][state["user_history"]["vote"] == 1]
    if liked.empty:
        print("\n[Profilo] Nessuna canzone votata positivamente.")
        return

    print("\n--- WRAPPED UTENTE ---")
    print(f"- Canzoni preferite: {len(liked)}")

    # Statistiche sulle feature audio
    for feature in ["valence", "energy", "mood_score"]:
        if feature in liked.columns:
            print(f"- {feature.capitalize()} medio: {liked[feature].mean():.2f}")


def top_artists(state):
    if state is None or state["user_history"].empty:
        print("\n[Wrapped] Occorre prima votare alcune canzoni!")
        return

    liked = state["user_history"][state["user_history"]["vote"] == 1]
    if "artists" in liked.columns and not liked.empty:
        s = liked["artists"].dropna().astype(str).str.strip()
        top3 = s.value_counts().head(3)
        if not top3.empty:
            print("\nTop 3 Artisti preferiti:")
            for i, (artist, count) in enumerate(top3.items(), start=1):
                print(f"({i}) {artist} ({count} canzoni)")
        else:
            print("\n[Wrapped] Nessun artista disponibile nei dati.")
    else:
        print("\n[Wrapped] Colonna 'artists' assente/nessun dato.")


def top_generi(state):
    if state is None or state["user_history"].empty:
        print("\n[Wrapped] Occorre prima votare alcune canzoni!")
        return
    
    liked = state["user_history"][state["user_history"]["vote"] == 1]

    if "track_genre" in liked.columns and not liked.empty:
        top3 = liked["track_genre"].value_counts().head(3)
        print("\nTop 3 Generi preferiti (track_genre):")
        for i, (genere, count) in enumerate(top3.items(), start=1):
            print(f"({i}) {genere} ({count} canzoni)")
    else:
        print("\n[Wrapped] Colonna 'track_genre' assente/nessun dato.")