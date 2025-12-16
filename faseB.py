# ===================================================================================
# ===================================================================================

# Obiettivo della fase:
# Voglio addestrare un modello di classificazione binaria che stimi
# la probabilità che un utente metta "like" a un brano audio,
#
#                   P(“piace”=1 | feature audio)
#
# usando i feedback raccolti.
# Senza almeno un like e un dislike, l’addestramento viene rimandato.

# ===================================================================================
# ===================================================================================

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------------

"""
    Vado a costruire una Pipeline con scaler MinMax + modello supervisionato, in questo
    modo garantisco che in train e in predizione venga applicata la stessa trasformazione.
    
    Modelli supportati:
        - RF (Random Forest): robusto e performante su feature eterogenee;
            non richiede scaling -> la faccio per essere coerente con MLP
            
            
            
        - MLP (Multi-Layer Perceptron): modello più complesso, capace di apprendere pattern non lineari;
            richiede scaling
            
"""

def build_model(model_type: str):
    
    if model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 16), # due hidden layer da 32 e 16 neuroni -> ho una capacità di apprendimento moderata
            activation="relu", # attivazione ReLU (Rectified Linear Unit) -> per catturare non-linearità complesse
            solver="adam", # ottimizzatore stocastico adattivo -> Adam gestisce il learning rate da solo
            max_iter=200,
            random_state=42
        )
        
    else:
        clf = RandomForestClassifier(
            n_estimators=200, # numero di alberi
            max_depth=None, # per ora metto una profondità libera
            min_samples_leaf=2, # vado a ridurre l'overfitting impedendo foglie con un solo campione
            class_weight="balanced",
            random_state=42
        )
        
    return Pipeline([("scaler", MinMaxScaler()), ("clf", clf)]) # Restituisco scaler + modello

# ------------------------------------------------------------------------------------

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
    
    pipeline = build_model(state["model_type"]) # vado a costruire il modello richiesto (RF o MLP) con scaler
    pipeline.fit(X, y) # addestramento supervisionato sui feedback raccolti
    state["model"] = pipeline # salvo il modello addestrato
    
    return pipeline # restituisco il modello addestrato

# ------------------------------------------------------------------------------------

"""
    Stampo le top 4 feature più importanti (solo per RF)
    -> aiuta a capire che pattern sta apprendendo il modello
"""
    
def print_feature_importance(state):
    
    model = state.get("model")
    
    if not isinstance(model, Pipeline):
        return
    
    rf = model.named_steps.get("clf")
    
    if not hasattr(rf, "feature_importances_"):
        return
    
    importances = pd.Series(rf.feature_importances_, index=state["feature_cols"])
    top = importances.sort_values(ascending=False).head(4)
    formatted = ", ".join([f"{feat} ({score:.2f})" for feat, score in top.items()])
    print(f"[Insight] Sto dando più peso a: {formatted}")