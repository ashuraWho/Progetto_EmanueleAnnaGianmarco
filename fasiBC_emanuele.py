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
            early_stopping=True, # interrompe l'addestramento se non migliora
            n_iter_no_change=15, # numero di iterazioni senza miglioramento per early stopping
            random_state = 42
        )
        
    else:
        clf = RandomForestClassifier(
            n_estimators = 200, # numero di alberi
            max_depth = max(3, min(8, n_samples // 2)), # profondità adattiva in base ai campioni disponibili (min 3, max 8)
            min_samples_leaf = 2, # vado a ridurre l'overfitting impedendo foglie con un solo campione
            class_weight = "balanced", # riduco overfitting se ho pochi voti
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
    
def train_model(state: Dict): # qui (state) me lo sono immaginato come dizionario quando vado a salvare i feedback
    
    if state["user_history"]["vote"].nunique() < 2: # senza due classi diverse non vado ad addestrare nulla
        state["model"] = None
        return None

    # Salvo (se esiste) l'ultima loss dell'MLP per confrontarla dopo il nuovo training
    previous_loss = state.get("last_loss")
    
    # X = feature numeriche dei brani votati | y = (0/1)
    X = state["user_history"][state["feature_cols"]]
    y = state["user_history"]["vote"].astype(int)
    
    # Switch automatico del modello
    model_type = "rf" if len(state["user_history"]) < 30 else "mlp" # uso RF di default e abilito MLP solo dopo 30 voti
    state["model_type"] = model_type
    
    pipeline = build_model(state["model_type"], len(state["user_history"])) # vado a costruire il modello richiesto (RF o MLP) con scaler
    pipeline.fit(X, y) # addestramento supervisionato sui feedback raccolti
    
    
    # Analisi della Loss (solo per MLP)
    """
        La loss rappresenta quanto il modello sta sbagliando durante l'addestramento.
        Una loss in diminuzione indica che il modello sta imparando a fare previsioni migliori.
        Qui salvo la loss attuale per confrontarla con quella precedente e vedere se sta migliorando.
    """
    
    if model_type == "mlp":
        clf = pipeline.named_steps.get("clf")

        # Ogni volta che ri-addestro l’MLP con i tuoi nuovi voti, scorro più epoche interne
        #   -> per ogni epoca memorizzo la loss in loss_curve_
        if hasattr(clf, "loss_curve_") and len(clf.loss_curve_) > 0:
            current_loss = float(clf.loss_curve_[-1])

            # Storico delle loss ad ogni ri-addestramento
            loss_history = state.get("loss_history", [])
            loss_history.append(current_loss)
            state["loss_history"] = loss_history

            # Salvo anche l'ultima loss per confronto nel main
            state["last_loss"] = current_loss
        else:
            # Ho visto che in casi rari loss_curve_ potrebbe non essere popolata
            state["last_loss"] = None
    else:
        # Se sto usando RF azzero eventuali info sulla loss del MLP
        state["last_loss"] = None
    
    state["model"] = pipeline # salvo il modello addestrato
    
    return pipeline # restituisco il modello addestrato


# ------------------------------------------------------------------------------------
# ================================ FUNZIONE PREDICT ==================================

"""
    Qui seleziono il prossimo brano usando:
        - massima probabilità -> per massima confidenza (exploitation)
        - massima incertezza (p circa 0.5) -> per esplorare meglio (exploration)
 """
    
def select_next_song(state: Dict, candidate_df: pd.DataFrame, exploration_rate=0.3):
    # state -> contiene il modello, le feature, storico dell'utente
	# candidate_df -> canzoni non ancora ascoltate
	# exploration_rate = 0.3 -> 30% delle volte esploro, 70% sfrutto
 
    # Non ci sono più canzoni da proporre
    if candidate_df.empty:
        return None

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
        # uncertainty = distanza dalla decision boundary (p = 0.5)
        candidate_df["uncertainty"] = np.abs(candidate_df["like_prob"] - 0.5) # ho incertezza massima quando p è vicino a 0.5 (decision boundary)
        return candidate_df.sort_values("uncertainty").head(1) # scelgo la canzone su cui il modello è più indeciso

    # Exploitation -> massima confidenza
    return candidate_df.sort_values("like_prob", ascending=False).head(1) # scelgo la canzone con la più alta probabilità di like


# ------------------------------------------------------------------------------------
# ============================ FUNZIONE FEATURE IMPORTANCE ============================

"""
    Stampo le top features più importanti (solo per RF)
    
        -> aiuta a capire che pattern sta apprendendo il modello
"""
    
def print_feature_importance(state: Dict, top_k=4):
    
    model = state.get("model")
    
    if not isinstance(model, Pipeline):
        return
    
    clf = model.named_steps.get("clf")
    
    if not hasattr(clf, "feature_importances_"):
        return
    
    importances = pd.Series(clf.feature_importances_, index=state["feature_cols"]).sort_values(ascending=False)
    top = importances.head(top_k)
    formatted = ", ".join([f"{feat} ({score:.2f})" for feat, score in top.items()])
    
    print(f"[Insight] Sto dando più peso a: {formatted}")