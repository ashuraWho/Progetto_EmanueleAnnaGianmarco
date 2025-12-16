import pandas as pd

# Importo le funzioni dal modulo cold_start
from cold_start import load_dataset, cold_start


def main():
    print("\n==============================")
    print("   TEST PROGETTO - AI DJ ")
    print("==============================\n")

    # Percorso del dataset
    dataset_path = "dataset.csv"

    # Caricamento dataset
    print("Caricamento dataset in corso...")
    df = load_dataset(dataset_path)
    print(f"Dataset caricato correttamente ({len(df)} canzoni)\n")

    while True:
        print("\n--- MENU ---")
        print("1. Avvia Cold Start")
        print("0. Esci")

        choice = input("Scelta: ")

        if choice == "1":
            # Avvio Cold Start
            user_history, seen_tracks = cold_start(df, n_songs=5)

            print("\n--- RISULTATO COLD START ---")
            print(user_history)

            print("\nNumero canzoni viste:", len(seen_tracks))

        elif choice == "0":
            print("\nUscita dal programma. A presto! 👋")
            break

        else:
            print("Scelta non valida. Riprova.")

if __name__ == "__main__":
    main()
