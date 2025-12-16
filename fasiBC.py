# ===================================================================================
# ===================================================================================

# Obiettivo fase B:
# Voglio addestrare un modello di classificazione binaria che stimi
# la probabilità che un utente metta "like" a un brano audio,
#
#                   P(“piace”=1 | feature audio)
# 
# usando i feedback raccolti.
# Senza almeno un like e un dislike, l’addestramento viene rimandato.

# ------------------------------------------------------------------------------------

# Ora la fase C:
#
#	-> p circa 1.0 -> "sono quasi sicuro che ti piaccia :)”
#	-> p circa 0.0 -> “sono quasi sicuro che non ti piaccia :(”
#	-> p circa 0.5 -> “ah boh :/” -> credo che sia questo caso che fa imparare di più il modello
#
# Problema: non posso proporre solo quando p = 1.0 o p = 0.5 -> mi serve una via di mezzo
# Strategia: uso exploration/exploitation, ovvero:
#   - exploration: 30% di probabilità scelgo il brano con p più vicino a 0.5
#   - exploitation: 70% di probabilità scelgo il brano con p più alta

# ===================================================================================
# ===================================================================================

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from typing import Dict


# ------------------------------------------------------------------------------------
# ================================== FUNZIONE BASE ===================================

"""
    Vado a costruire una Pipeline con scaler MinMax + modello supervisionato, in questo
    modo garantisco che in train e in predizione venga applicata la stessa trasformazione.
    
    Modelli supportati:
        - RF (Random Forest): robusto e performante su feature eterogenee;
            non richiede scaling -> la faccio per essere coerente con MLP
            
        - MLP (Multi-Layer Perceptron): modello più complesso, capace di apprendere pattern non lineari;
            richiede scaling
"""

def build_model(model_type: str, n_samples: int):
    
    if model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes = (32, 16), # due hidden layer da 32 e 16 neuroni -> ho una capacità di apprendimento moderata
            activation = "relu", # attivazione ReLU (Rectified Linear Unit) -> per catturare non-linearità complesse
            solver = "adam", # ottimizzatore stocastico adattivo -> Adam gestisce il learning rate da solo
            max_iter = 300,
            random_state = 42
        )
        
    else:
        clf = RandomForestClassifier(
            n_estimators = 200, # numero di alberi
            max_depth = max(3, min(8, n_samples // 2))  # profondità adattiva in base ai campioni disponibili
            min_samples_leaf = 2, # vado a ridurre l'overfitting impedendo foglie con un solo campione
            class_weight = "balanced",
            random_state = 42
        )
        
    return Pipeline([("scaler", MinMaxScaler()), ("clf", clf)]) # Restituisco scaler + modello


# ------------------------------------------------------------------------------------
# ================================== FUNZIONE TRAIN ==================================

"""
    Vado a (ri)addestrare il modello sui feedback raccolti.
    
    Pre-condizione: devo avere almeno due classi presenti (1=like, 0=dislike)
    
    -> dopo ogni nuovo feedback, il modello viene riaddestrato per incorporare la nuova informazione
"""
    
def train_model(state): # qui (state) me lo sono immaginato come diziomnario quando vado a salvare i feedback
    
    if state["user_history"]["label"].nunique() < 2: # senza due classi diverse non vado ad addestrare nulla
        return None
    
    # X = feature numeriche dei brani votati | y = (0/1)
    X = state["user_history"][state["feature_cols"]]
    y = state["user_history"]["label"].astype(int)
    
    # Switch automatico del modello
    model_type = "rf" if len(state["user_history"]) < 20 else "mlp" # uso RF di default e abilito MLP solo dopo 20 voti
    state["model_type"] = model_type
    
    pipeline = build_model(state["model_type"]) # vado a costruire il modello richiesto (RF o MLP) con scaler
    pipeline.fit(X, y) # addestramento supervisionato sui feedback raccolti
    
    state["model"] = pipeline # salvo il modello addestrato
    return pipeline # restituisco il modello addestrato


# ------------------------------------------------------------------------------------
# ================================ FUNZIONE PREDICT ==================================

"""
    Qui seleziono il prossimo brano usando:
        - massima probabilità -> per massima confidenza (exploitation)
        - massima incertezza (p circa 0.5) -> per esplorare meglio (exploration)
 """
    
def select_next_song(state, candidate_df, exploration_rate=0.3):
    # state -> contiene il modello, le feature, storico dell'utente
	# candidate_df -> canzoni non ancora ascoltate
	# exploration_rate = 0.3 -> 30% delle volte esploro, 70% sfrutto

    model = state.get("model")
    features = state["feature_cols"]

    if model is None:
        return candidate_df.sample(1) # Se non ho ancora un modello addestrato, scelgo una canzone a caso (Cold start)

    probs = model.predict_proba(candidate_df[features])[:, 1] # Probabilità di "like" per ogni canzone candidata
    
    # Aggiungo le probabilità al DataFrame delle candidate
    candidate_df = candidate_df.copy()
    candidate_df["like_prob"] = probs

    # Scelgo se esplorare o sfruttare -> il 30% delle volte entra qui
    if np.random.rand() < exploration_rate:
        # Exploration -> modello impara più velocemente
        candidate_df["uncertainty"] = np.abs(candidate_df["like_prob"] - 0.5) # ho incertezza massima quando p è vicino a 0.5
        return candidate_df.sort_values("uncertainty").head(1) # scelgo la canzone su cui il modello è più indeciso

    # Exploitation -> massima confidenza
    return candidate_df.sort_values("like_prob", ascending=False).head(1) # scelgo la canzone con la più alta probabilità di like


# ------------------------------------------------------------------------------------
# ============================ FUNZIONE FEATURE IMPORTANCE ============================

"""
    Stampo le top 4 feature più importanti (solo per RF)
    
        -> aiuta a capire che pattern sta apprendendo il modello
"""
    
def print_feature_importance(state):
    
    model = state.get("model")
    
    if not isinstance(model, Pipeline):
        return
    
    clf = model.named_steps.get("clf")
    
    if not hasattr(clf, "feature_importances_"):
        return
    
    importances = pd.Series(clf.feature_importances_, index=state["feature_cols"]).sort_values(ascending=False)
    top = importances.head(4)
    formatted = ", ".join([f"{feat} ({score:.2f})" for feat, score in top.items()])
    
    print(f"[Insight] Sto dando più peso a: {formatted}")


# ------------------------------------------------------------------------------------
# ================================= FUNZIONE FEEDBACK ================================

"""
    Aggiungiamo al log utente (la history) le feature del brano proposto con l’etichetta
"""

def save_feedback(state: Dict, song: pd.Series, label: int):
    
    record = song[state["meta_cols"] + state["feature_cols"]].copy()
    record["label"] = label
    
    state["user_history"] = pd.concat(
        [state["user_history"],
         pd.DataFrame([record])],
        ignore_index=True
    )