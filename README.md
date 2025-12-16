# Progetto_EmanueleAnnaGianmarco
Progetto Finale Academy Python &amp; Machine Learning
Partecipanti:
- Emanuele Anzellotti
- Anna Firinu
- Gianmarco Sorrentino

## AI DJ (Recommender interattivo da terminale)

Lo script `ai_dj.py` simula un sistema di raccomandazione che impara dai feedback dell'utente in tempo reale.

### Prerequisiti
- Python 3.9+
- Librerie: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Uso
```bash
python ai_dj.py
```

### Flusso di gioco
- **Riscaldamento**: 5 brani casuali da votare (1=like, 0=dislike, q=esci).
- **Modello attivo**: Random Forest con scaling MinMax sceglie il brano più promettente e aggiorna il modello a ogni feedback.
- **Interpretabilità**: ogni 10 turni stampa le feature più importanti.
- **Uscita**: scatter plot Valence vs Energy colorato per like/dislike.
