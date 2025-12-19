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

        if n_samples < 600:
            hidden = (12,) # architettura più semplice per pochi dati
        else:
            hidden = (24, 12) # architettura più complessa per più dati
      
        clf = MLPClassifier(
            hidden_layer_sizes = hidden, # architettura adattiva in base ai campioni disponibili
            activation = "relu", # attivazione ReLU (Rectified Linear Unit) -> per catturare non-linearità complesse
            solver = "adam", # ottimizzatore stocastico adattivo -> Adam gestisce il learning rate da solo
            alpha = 0.1, # regolarizzazione L2: impedisce pesi eccessivi (evita sbalzi verso generi random)
            learning_rate_init = 0.005, # passo leggermente più piccolo per addestramenti più stabili
            max_iter = 500,
            early_stopping=True, # interrompe l'addestramento se non migliora
            validation_fraction = 0.1, # 10% dei dati per validazione interna
            n_iter_no_change = 20, # numero di iterazioni senza miglioramento per early stopping
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
    
    # Considero per l'addestramento solo i voti informativi:
    #   0 = dislike forte
    #   1 = like forte
    #   3 = forse sì  (like debole)
    #   4 = forse no  (dislike debole)
    # I voti 2 = indifferente vengono ignorati (nessun effetto sul modello).
    history_train = state["user_history"][state["user_history"]["vote"].isin([0, 1, 3, 4])].copy() # filtro solo i voti utili
    
    if history_train.empty:
        state["model"] = None # non ho dati utili per addestrare
        return None

    # Target binario: 1 per like/forse sì, 0 per dislike/forse no
    history_train["target"] = history_train["vote"].isin([1, 3]).astype(int) # converto in 0/1

    # Senza almeno una canzone "piace" e una "non piace", non addestro nulla
    if history_train["target"].nunique() < 2:
        state["model"] = None
        return None

    # Salvo (se esiste) l'ultima loss dell'MLP per confrontarla dopo il nuovo training
    previous_loss = state.get("last_loss")
    
    # X = feature numeriche dei brani votati | y = (0/1) (dopo la mappatura sopra)
    X = history_train[state["feature_cols"]]
    y = history_train["target"].astype(int)

    # Pesi dei campioni:
    #   - like/dislike forti -> peso 1.0
    #   - forse sì/forse no -> peso 0.5
    sample_weight = history_train["vote"].map(
        {
            1: 1.0,  # like forte
            0: 1.0,  # dislike forte
            3: 0.5,  # forse sì
            4: 0.5,  # forse no
        }
    ).astype(float)
    
    # Switch automatico del modello (uso il target binario dopo la mappatura)
    n_like = int(y.sum()) # numero di like (1)
    n_dislike = int(len(y) - n_like) # numero di dislike (0)
    n_samples = len(history_train) # numero totale di campioni

    model_type = "rf" if len(state["user_history"]) < 300 or n_like < 60 or n_dislike < 60 else "mlp" # uso RF di default e abilito MLP solo se ho abbastanza dati
    state["model_type"] = model_type
    
    pipeline = build_model(state["model_type"], len(state["user_history"])) # vado a costruire il modello richiesto (RF o MLP) con scaler
    pipeline.fit(X, y, clf__sample_weight=sample_weight.values) # addestramento supervisionato sui feedback raccolti, con pesi diversi per voti forti/deboli
    
    # ---------
    
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
	# candidate_df -> canzoni non ancora votate
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