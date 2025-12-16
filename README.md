# Progetto_EmanueleAnnaGianmarco
Progetto Finale Academy Python &amp; Machine Learning
Partecipanti:
- Emanuele Anzellotti
- Anna Firinu
- Gianmarco Sorrentino

# AI DJ – Fasi B e C: Machine Learning & Recommender System

## Obiettivo

Le fasi B e C del progetto hanno l’obiettivo di creare un modello di classificazione binaria che impari dai feedback dell’utente e predica la probabilità che un brano venga apprezzato:

$$P(\text{“mi piace”}=1 \mid \text{feature audio})$$

Il sistema è progettato per apprendere in tempo reale, aggiornando il modello dopo ogni voto e suggerendo brani con confidenza crescente o, alternativamente, esplorando quelli più incerti.

## Fase B – Training del modello
1. Input: feedback dell’utente (user_history) con feature audio numeriche e label 0/1.
2. Condizione pre-addestramento: il modello viene costruito solo se ci sono almeno due classi presenti (almeno un like e un dislike).
3. Scelta del modello:
    - Random Forest (RF) -> ideale per dataset piccoli, infatti messo di default.
    - MLP (Multi-Layer Perceptron) -> abilitato dopo 20 voti.
4. Pipeline: tutte le feature vengono scalate con MinMaxScaler per uniformità tra RF e MLP.
5. Training: ogni volta che arriva un nuovo feedback, il modello viene riaddestrato per incorporare la nuova informazione.
6. Output: pipeline addestrata salvata nello state["model"].

## Fase C – Predizione & Active Learning
1. Input: pool di brani non ancora ascoltati (candidate_df) e modello addestrato.
2. Predizione: il modello calcola la probabilità di like per ciascun brano.
3. Exploration/Exploitation:
    - Exploitation (70%) -> scegliere la canzone con probabilità di like più alta.
    - Exploration (30%) -> scegliere la canzone con probabilità più vicina a 0.5, dove il modello è più incerto.
4. Motivazione: proporre brani incerti permette al modello di imparare più velocemente.