"""
Pulizia del dataset: rimozione delle istanze e classi identificate
dall'analisi di coerenza delle mani.

Rimuove:
  - Istanze con errori certi di rilevamento (errori_certi.txt)
  - Istanze con mano errata rispetto al signer (errori_mano_errata.txt)
  - Classi intere da eliminare (classi_da_eliminare.txt)
"""

import os
import shutil

DATASET_DIR = "/Users/antoniowalterdefusco/Documents/Project/Signify/data/processed/Dataset_Keypoints_Train"
DATA_DIR = "/Users/antoniowalterdefusco/Documents/Project/Signify/data/processed"


def carica_lista(filepath):
    """Carica un file di testo e restituisce le righe non vuote e non commentate."""
    righe = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                righe.append(line)
    return righe


def main():
    print("=" * 60)
    print("PULIZIA DATASET")
    print("=" * 60)

    # 1. Elimina classi intere
    classi_file = os.path.join(DATA_DIR, "classi_da_eliminare.txt")
    classi = carica_lista(classi_file)
    print(f"\n[1/3] Eliminazione di {len(classi)} classi intere...")
    for classe in classi:
        class_dir = os.path.join(DATASET_DIR, classe)
        if os.path.exists(class_dir):
            n_files = len(os.listdir(class_dir))
            shutil.rmtree(class_dir)
            print(f"  Eliminata: {classe}/ ({n_files} file)")
        else:
            print(f"  Non trovata: {classe}/")

    # 2. Elimina errori certi
    errori_certi_file = os.path.join(DATA_DIR, "errori_certi.txt")
    righe = carica_lista(errori_certi_file)
    print(f"\n[2/3] Eliminazione di {len(righe)} errori certi...")
    eliminati = 0
    non_trovati = 0
    for riga in righe:
        parti = riga.split('|')
        classe = parti[0]
        filename = parti[1]
        filepath = os.path.join(DATASET_DIR, classe, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            eliminati += 1
        else:
            non_trovati += 1
    print(f"  Eliminati: {eliminati}, Non trovati: {non_trovati}")

    # 3. Elimina errori mano errata
    errori_mano_file = os.path.join(DATA_DIR, "errori_mano_errata.txt")
    righe = carica_lista(errori_mano_file)
    print(f"\n[3/3] Eliminazione di {len(righe)} errori mano errata...")
    eliminati = 0
    non_trovati = 0
    for riga in righe:
        parti = riga.split('|')
        classe = parti[0]
        filename = parti[1]
        filepath = os.path.join(DATASET_DIR, classe, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            eliminati += 1
        else:
            non_trovati += 1
    print(f"  Eliminati: {eliminati}, Non trovati: {non_trovati}")

    # Conteggio finale
    print(f"\n{'=' * 60}")
    print("CONTEGGIO FINALE")
    print(f"{'=' * 60}")
    classi_rimaste = [d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d))]
    totale_file = 0
    classi_vuote = []
    for classe in classi_rimaste:
        n = len([f for f in os.listdir(os.path.join(DATASET_DIR, classe)) if f.endswith('.npy')])
        totale_file += n
        if n == 0:
            classi_vuote.append(classe)

    print(f"Classi rimaste: {len(classi_rimaste)}")
    print(f"File rimasti: {totale_file}")
    if classi_vuote:
        print(f"Classi diventate vuote dopo la pulizia: {len(classi_vuote)}")
        for c in classi_vuote:
            print(f"  - {c}")


if __name__ == "__main__":
    main()
