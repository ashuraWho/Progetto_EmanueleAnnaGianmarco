# ==============================================================================
# =============== Qui implemento l'analisi fatta in pulizia.ipynb ==============
# ==============================================================================

import pandas as pd
from typing import List


# ========================== CONFIGURAZIONE FEATURE ==========================

# Feature di base (audio features originali dal dataset Spotify)
BASE_FEATURES: List[str] = [
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

# Feature derivate (feature engineering dal notebook di pulizia)
ENGINEERED_FEATURES: List[str] = [
    "mood_score",       # media tra energy e valence -> proxy del "mood" complessivo
    "electronic_index", # energy - acousticness -> quanto il brano è elettronico vs acustico
    "is_instrumental",  # flag binario brano strumentale
    "dance_mood",       # media tra danceability e valence
]

# Feature finali usate dal modello
FINAL_FEATURES: List[str] = BASE_FEATURES + ENGINEERED_FEATURES


# =========================== MAPPATURA DEI GENERI ==========================

GENRE_MAP = {
    # ACOUSTIC / FOLK
    "acoustic": "folk",
    "folk": "folk",
    "bluegrass": "folk",
    "singer-songwriter": "folk",
    "honky-tonk": "folk",

    # ROCK
    "rock": "rock",
    "alt-rock": "rock",
    "alternative": "rock",
    "indie": "rock",
    "indie-pop": "rock",
    "hard-rock": "rock",
    "punk": "rock",
    "punk-rock": "rock",
    "rock-n-roll": "rock",
    "rockabilly": "rock",
    "psych-rock": "rock",
    "grunge": "rock",
    "emo": "rock",
    "goth": "rock",

    # METAL
    "metal": "metal",
    "heavy-metal": "metal",
    "black-metal": "metal",
    "death-metal": "metal",
    "metalcore": "metal",
    "grindcore": "metal",
    "hardcore": "metal",

    # ELECTRONIC
    "electronic": "electronic",
    "edm": "electronic",
    "house": "electronic",
    "deep-house": "electronic",
    "progressive-house": "electronic",
    "chicago-house": "electronic",
    "techno": "electronic",
    "detroit-techno": "electronic",
    "minimal-techno": "electronic",
    "trance": "electronic",
    "club": "electronic",
    "breakbeat": "electronic",
    "drum-and-bass": "electronic",
    "dubstep": "electronic",
    "electro": "electronic",
    "idm": "electronic",
    "hardstyle": "electronic",
    "garage": "electronic",

    # POP
    "pop": "pop",
    "dance-pop": "pop",
    "power-pop": "pop",
    "synth-pop": "pop",
    "pop-film": "pop",
    "cantopop": "pop",
    "mandopop": "pop",
    "k-pop": "pop",
    "j-pop": "pop",
    "j-idol": "pop",

    # HIP HOP / R&B
    "hip-hop": "hip-hop",
    "r-n-b": "hip-hop",
    "groove": "hip-hop",

    # JAZZ / BLUES
    "jazz": "jazz",
    "blues": "jazz",

    # WORLD / LATIN
    "latin": "latin",
    "latino": "latin",
    "salsa": "latin",
    "samba": "latin",
    "pagode": "latin",
    "forro": "latin",
    "sertanejo": "latin",
    "mpb": "latin",
    "brazil": "latin",
    "tango": "latin",
    "reggae": "latin",
    "reggaeton": "latin",
    "ska": "latin",

    # WORLD (COUNTRIES)
    "french": "world",
    "german": "world",
    "spanish": "world",
    "swedish": "world",
    "turkish": "world",
    "iranian": "world",
    "indian": "world",
    "malay": "world",
    "british": "world",

    # CLASSICAL
    "classical": "classical",
    "opera": "classical",
    "piano": "classical",

    # SOUL / FUNK
    "soul": "soul",
    "funk": "soul",
    "gospel": "soul",

    # CHILDREN / SHOW
    "children": "children",
    "kids": "children",
    "disney": "children",
    "show-tunes": "children",
    "anime": "children",

    # COMEDY
    "comedy": "comedy",

    # AMBIENT / NEW AGE
    "ambient": "ambient",
    "new-age": "ambient",
    "chill": "ambient",
    "sleep": "ambient",
    "study": "ambient",

    # DANCE / PARTY
    "dance": "dance",
    "party": "dance",
    "disco": "dance",
    # "club" è già mappato in electronic sopra

    # MOOD
    "happy": "mood",
    "sad": "mood",
    "romance": "mood",

    # GENERIC
    "world-music": "world",
}


# ============================ FUNZIONI DI SUPPORTO =========================

"""
    Replica la logica del notebook:
      - rimozione valori mancanti
      - aggregazione per (track_name, artists) con:
          * media per le feature numeriche principali
          * moda per le feature categoriche principali
"""
    
def _drop_na_and_deduplicate(df: pd.DataFrame):

    # Rimuovo eventuali righe con NaN (come nel notebook)
    df = df.dropna()

    # Funzione di moda identica a quella definita nel notebook
    def _mode(series: pd.Series):
        return series.mode().iloc[0]

    # Dizionario di aggregazione che replica esattamente il notebook
    agg_dict = {
        # ---------- FEATURE NUMERICHE → MEDIA ----------
        "popularity": "mean",
        "duration_ms": "mean",
        "danceability": "mean",
        "energy": "mean",
        "loudness": "mean",
        "speechiness": "mean",
        "acousticness": "mean",
        "instrumentalness": "mean",
        "liveness": "mean",
        "valence": "mean",
        "tempo": "mean",

        # ---------- FEATURE CATEGORICHE → MODA ----------
        "album_name": _mode,
        "track_genre": _mode,
        "explicit": _mode,
        "key": _mode,
        "mode": _mode,
        "time_signature": _mode,

        # ---------- METADATO ----------
        "track_id": "first",
    }

    # Deduplicazione semantica: una riga per coppia (track_name, artists)
    df_dedup = (
        df
        .groupby(["track_name", "artists"], as_index=False)
        .agg(agg_dict)
    )
    
    return df_dedup

# --------------------------------------------------------------------------------

"""
    Aggiunge:
      - sub_genre (copia di track_genre)
      - main_genre (mappato con GENRE_MAP, 'other' se non mappato)
"""
    
def _add_genre_columns(df: pd.DataFrame):

    if "track_genre" not in df.columns:
        return df

    df = df.copy()
    df["sub_genre"] = df["track_genre"]
    df["main_genre"] = df["track_genre"].map(GENRE_MAP).fillna("other")
    
    return df

# --------------------------------------------------------------------------------

"""
    Aggiunge le feature derivate usate nel notebook:
      - mood_score
      - dance_mood
      - electronic_index
      - is_instrumental
"""
    
def _add_engineered_features(df: pd.DataFrame):
    
    df = df.copy()

    # Mood score: combina energia ed emozione
    if {"energy", "valence"}.issubset(df.columns):
        df["mood_score"] = (df["energy"] + df["valence"]) / 2.0

    # Dance mood: musica ballabile e positiva
    if {"danceability", "valence"}.issubset(df.columns):
        df["dance_mood"] = (df["danceability"] + df["valence"]) / 2.0

    # Electronic index: contrapposizione tra elettronico e acustico
    if {"energy", "acousticness"}.issubset(df.columns):
        df["electronic_index"] = df["energy"] - df["acousticness"]

    # Flag strumentale
    if "instrumentalness" in df.columns:
        df["is_instrumental"] = (df["instrumentalness"] > 0.5).astype(int)

    return df


# =========================== FUNZIONE PRINCIPALE ===========================

"""
    Applica tutta la pipeline di pulizia + feature engineering del notebook:

      1. Drop dei valori mancanti e deduplicazione per (track_name, artists)
      2. Aggiunta di colonne di genere musicale
      3. Aggiunta delle feature derivate

    Restituisce un DataFrame pronto per essere usato nelle fasi A/B/C/D.
"""
    
def preprocess_dataset(df: pd.DataFrame):
    
    df_clean = _drop_na_and_deduplicate(df)
    df_clean = _add_genre_columns(df_clean)
    df_clean = _add_engineered_features(df_clean)
    
    return df_clean


def load_and_preprocess_dataset(path: str): # Comoda se si vuole caricare e pulire il dataset in un colpo solo

    df = pd.read_csv(path)
    
    return preprocess_dataset(df)