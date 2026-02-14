"""
Bilanciamento del dataset: rimozione classi sotto-rappresentate
e trimming casuale di quelle sovra-rappresentate.

Regole:
  - Classi con < 12 campioni → eliminate completamente
  - Classi con 12-15 campioni → invariate
  - Classi con > 15 campioni → ridotte a 15 (eliminazione casuale)
"""

import os
import shutil
import random
import numpy as np

DATASET_DIR = "/Users/antoniowalterdefusco/Documents/Project/Signify/data/processed/Dataset_Keypoints_Train"
SOGLIA_MIN = 12
TARGET_MAX = 15

random.seed(42)  # Riproducibilità


def main():
    print("=" * 60)
    print("BILANCIAMENTO DATASET")
    print(f"  Soglia minima: {SOGLIA_MIN} campioni")
    print(f"  Target massimo: {TARGET_MAX} campioni")
    print("=" * 60)

    glosses = sorted([g for g in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, g))])

    classi_rimosse = 0
    campioni_rimossi_sottosoglia = 0
    classi_trimmate = 0
    campioni_rimossi_trim = 0
    classi_ok = 0

    for gloss in glosses:
        gloss_dir = os.path.join(DATASET_DIR, gloss)
        npy_files = [f for f in os.listdir(gloss_dir) if f.endswith('.npy')]
        n = len(npy_files)

        if n < SOGLIA_MIN:
            # Classe sotto-soglia: eliminare completamente
            shutil.rmtree(gloss_dir)
            classi_rimosse += 1
            campioni_rimossi_sottosoglia += n

        elif n > TARGET_MAX:
            # Classe sovra-rappresentata: trim casuale
            da_rimuovere = n - TARGET_MAX
            file_da_eliminare = random.sample(npy_files, da_rimuovere)
            for f in file_da_eliminare:
                os.remove(os.path.join(gloss_dir, f))
            classi_trimmate += 1
            campioni_rimossi_trim += da_rimuovere

        else:
            classi_ok += 1

    # Conteggio finale
    glosses_finali = sorted([g for g in os.listdir(DATASET_DIR)
                             if os.path.isdir(os.path.join(DATASET_DIR, g))])
    totale_file = 0
    counts = []
    for g in glosses_finali:
        n = len([f for f in os.listdir(os.path.join(DATASET_DIR, g)) if f.endswith('.npy')])
        totale_file += n
        counts.append(n)

    counts = np.array(counts)

    print(f"\n{'─' * 60}")
    print("OPERAZIONI EFFETTUATE")
    print(f"{'─' * 60}")
    print(f"  Classi rimosse (< {SOGLIA_MIN} campioni): {classi_rimosse} ({campioni_rimossi_sottosoglia} campioni)")
    print(f"  Classi trimmate (> {TARGET_MAX} → {TARGET_MAX}): {classi_trimmate} ({campioni_rimossi_trim} campioni rimossi)")
    print(f"  Classi invariate: {classi_ok}")

    print(f"\n{'─' * 60}")
    print("DATASET RISULTANTE")
    print(f"{'─' * 60}")
    print(f"  Classi: {len(glosses_finali)}")
    print(f"  Campioni totali: {totale_file}")
    print(f"  Media: {np.mean(counts):.1f}")
    print(f"  Mediana: {np.median(counts):.0f}")
    print(f"  Std: {np.std(counts):.2f}")
    print(f"  Min: {np.min(counts)}")
    print(f"  Max: {np.max(counts)}")

    # Distribuzione finale
    print(f"\n  Distribuzione:")
    for v in range(int(np.min(counts)), int(np.max(counts)) + 1):
        n_classi = sum(1 for c in counts if c == v)
        bar = '█' * (n_classi // 5)
        print(f"    {v:3d} campioni: {n_classi:5d} classi  {bar}")

    print("=" * 60)


if __name__ == "__main__":
    main()
