import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict

# ======================= GRAFICO VALENCE vs ENERGY + BOUNDARY =======================

"""
    Mostra il grafico di dispersione Valence vs Energy e traccia la decision boundary del modello:
        - punti verdi: brani con voto 1 (Like)
        - punti rossi : brani con voto 0 (Dislike)
        - sfondo colorato: probabilità di like stimata dal modello (da rosso/basso a verde/alto)
        - curva nera: decision boundary p = 0.5
        
        - uso il modello addestrato per calcolare la probabilità di like
          su una griglia nello spazio (valence, energy)
"""

def plot_valence_energy_boundary(state: Dict):
    
    user_history = state.get("user_history")
    model = state.get("model")
    feature_cols = state.get("feature_cols", [])

    if user_history is None or user_history.empty or model is None:
        print("[Grafico] Non ci sono abbastanza dati o il modello non è addestrato: salto il grafico finale.")
        return

    if "valence" not in feature_cols or "energy" not in feature_cols:
        print("[Grafico] Le feature 'valence' ed 'energy' non sono disponibili: impossibile creare il grafico.")
        return

    # Separo like e dislike
    likes = user_history[user_history["vote"] == 1]
    dislikes = user_history[user_history["vote"] == 0]

    # Range del grafico rispetto ai dati osservati -> lo estendo con un padding per visibilità
    v_min, v_max = user_history["valence"].min(), user_history["valence"].max()
    e_min, e_max = user_history["energy"].min(), user_history["energy"].max()

    padding = 0.05 # padding per i bordi
    v_min, v_max = max(0.0, v_min - padding), min(1.0, v_max + padding)
    e_min, e_max = max(0.0, e_min - padding), min(1.0, e_max + padding)

    # Griglia di punti nello spazio Valence-Energy
    xx, yy = np.meshgrid(
        np.linspace(v_min, v_max, 100), # 100 punti tra v_min e v_max
        np.linspace(e_min, e_max, 100), # 100 punti tra e_min e e_max
    )

    grid = pd.DataFrame({
        "valence": xx.ravel(), # colonna di valence
        "energy": yy.ravel(), # colonna di energy
    })

    # Per le altre feature uso la mediana osservata nello user_history
    for feat in feature_cols:
        if feat in ["valence", "energy"]:
            continue
        median_val = user_history[feat].median()
        grid[feat] = median_val

    # Probabilità di like sul reticolo
    probs = model.predict_proba(grid[feature_cols])[:, 1] # Probabilità di like (classe 1)
    Z = probs.reshape(xx.shape) # Rimodello per adattarlo alla griglia

    plt.figure(figsize=(7, 5))

    # Sfondo colorato con le probabilità
    cs = plt.contourf(xx, yy, Z, levels=20, cmap="RdYlGn", alpha=0.4) # da rosso (basso) a verde (alto)
    plt.colorbar(cs, label="Probabilità di like")

    # Linea di decisione p = 0.5
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5) # decision boundary

    # Punti reali dell'utente
    if not likes.empty:
        plt.scatter(likes["valence"], likes["energy"], c="green", label="Like (1)") # punti verdi
    if not dislikes.empty:
        plt.scatter(dislikes["valence"], dislikes["energy"], c="red", label="Dislike (0)") # punti rossi

    plt.xlabel("Valence (felicità)")
    plt.ylabel("Energy (energia)")
    plt.title("AI DJ - Spazio Valence vs Energy e decision boundary del modello")
    plt.legend()
    plt.tight_layout()
    plt.show()