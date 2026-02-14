"""
Analisi raffinata di coerenza delle mani con informazioni sui signer.

Logica di classificazione degli errori:
  1. Classe ENTRAMBE → istanza con UNA o NESSUNA mano  → ERRORE CERTO
  2. Classe SOLO_DX/SOLO_SX → istanza NESSUNA mano     → ERRORE CERTO
  3. Classe NESSUNA (pattern dominante)                 → CLASSE DA ELIMINARE
  4. Classe SOLO_DX → istanza SOLO_SX (e viceversa)     → Controllare se il signer
     è mancino/destrorso: se coerente col signer → OK, altrimenti ERRORE

La mano dominante di ogni signer viene inferita analizzando tutti i suoi video
nelle classi a una sola mano.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# Percorsi
DATASET_DIR = "/Users/antoniowalterdefusco/Documents/Project/Signify/data/processed/Dataset_Keypoints_Train"
CSV_PATH = "/Users/antoniowalterdefusco/Documents/Project/Signify/data/processed/train.csv"
OUTPUT_DIR = "/Users/antoniowalterdefusco/Documents/Project/Signify/data/processed"


def classifica_mani(filepath):
    """Classifica il pattern di utilizzo delle mani per un file .npy."""
    data = np.load(filepath)
    n_frames = data.shape[0]
    soglia = 0.20

    lh_presenti = sum(1 for frame in data if not np.all(frame[132:195] == 0))
    rh_presenti = sum(1 for frame in data if not np.all(frame[195:258] == 0))

    lh_usata = (lh_presenti / n_frames) >= soglia
    rh_usata = (rh_presenti / n_frames) >= soglia

    if lh_usata and rh_usata:
        return "ENTRAMBE"
    elif rh_usata:
        return "SOLO_DX"
    elif lh_usata:
        return "SOLO_SX"
    else:
        return "NESSUNA"


def carica_mapping_signer():
    """Carica la mappatura video_filename → Participant ID dal CSV estratto."""
    df = pd.read_csv(CSV_PATH)

    mapping = {}
    for _, row in df.iterrows():
        npy_name = row['Video file'].split('.')[0] + '.npy'
        mapping[npy_name] = row['Participant ID']

    print(f"Caricati {len(mapping)} mapping video → signer ({df['Participant ID'].nunique()} signer)")
    return mapping


def main():
    print("=" * 70)
    print("ANALISI RAFFINATA DI COERENZA MANI CON DATI SIGNER")
    print("=" * 70)

    # Fase 1: Carica mapping signer
    print("\n[1/4] Caricamento mapping signer...")
    signer_map = carica_mapping_signer()

    # Fase 2: Classifica tutte le istanze e determina pattern per classe
    print("\n[2/4] Classificazione pattern mani per tutte le istanze...")
    glosses = sorted([g for g in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, g))])

    # Struttura: {gloss: {filename: pattern}}
    classe_istanze = {}
    # Struttura: {gloss: pattern_dominante}
    classe_pattern = {}
    totale_analizzati = 0

    for gloss in glosses:
        gloss_dir = os.path.join(DATASET_DIR, gloss)
        npy_files = [f for f in os.listdir(gloss_dir) if f.endswith('.npy')]

        if not npy_files:
            classe_pattern[gloss] = "VUOTA"
            continue

        istanze = {}
        for npy_file in npy_files:
            filepath = os.path.join(gloss_dir, npy_file)
            try:
                istanze[npy_file] = classifica_mani(filepath)
                totale_analizzati += 1
            except Exception as e:
                print(f"  Errore: {filepath} → {e}")

        classe_istanze[gloss] = istanze
        conteggio = Counter(istanze.values())
        classe_pattern[gloss] = conteggio.most_common(1)[0][0]

        if totale_analizzati % 2000 == 0:
            print(f"  Analizzati {totale_analizzati} file...")

    print(f"  Totale analizzati: {totale_analizzati}")

    # Fase 3: Inferisci mano dominante per signer
    print("\n[3/4] Inferenza mano dominante per signer...")
    # Per ogni signer, conta quante volte usa DX vs SX nelle classi a una sola mano
    signer_hand_counts = defaultdict(lambda: {"DX": 0, "SX": 0})

    for gloss, istanze in classe_istanze.items():
        pat_dominante = classe_pattern[gloss]
        # Usiamo solo le classi dove il pattern è SOLO_DX o SOLO_SX
        if pat_dominante not in ("SOLO_DX", "SOLO_SX"):
            continue

        for npy_file, pattern in istanze.items():
            if pattern not in ("SOLO_DX", "SOLO_SX"):
                continue
            signer = signer_map.get(npy_file, None)
            if signer is None:
                continue
            if pattern == "SOLO_DX":
                signer_hand_counts[signer]["DX"] += 1
            else:
                signer_hand_counts[signer]["SX"] += 1

    # Determina mano dominante
    signer_dominante = {}
    print(f"\n  {'Signer':<10} {'DX':>6} {'SX':>6} {'Dominante':>12}")
    print(f"  {'─' * 36}")
    for signer in sorted(signer_hand_counts.keys()):
        counts = signer_hand_counts[signer]
        dx = counts["DX"]
        sx = counts["SX"]
        if dx + sx == 0:
            dominante = "SCONOSCIUTA"
        elif dx >= sx:
            dominante = "DESTRORSO"
        else:
            dominante = "MANCINO"
        signer_dominante[signer] = dominante
        print(f"  {signer:<10} {dx:>6} {sx:>6} {dominante:>12}")

    # Fase 4: Classifica errori
    print("\n[4/4] Classificazione errori...")

    errori_certi = []        # Da eliminare sicuramente
    errori_mano_errata = []  # Mano sbagliata rispetto al signer
    classi_da_eliminare = [] # Classi con pattern NESSUNA o VUOTA
    ok_mancini = []          # Sembrano anomali ma coerenti col signer mancino

    for gloss, istanze in classe_istanze.items():
        pat_dominante = classe_pattern[gloss]

        # Classi senza mani rilevate → eliminare la classe intera
        if pat_dominante == "NESSUNA":
            classi_da_eliminare.append(gloss)
            continue

        if pat_dominante == "VUOTA":
            classi_da_eliminare.append(gloss)
            continue

        for npy_file, pattern in istanze.items():
            signer = signer_map.get(npy_file, "SCONOSCIUTO")

            # Regola 1: classe ENTRAMBE → istanza con una o nessuna mano = errore
            if pat_dominante == "ENTRAMBE" and pattern != "ENTRAMBE":
                errori_certi.append({
                    'classe': gloss, 'file': npy_file, 'signer': signer,
                    'pattern_istanza': pattern, 'pattern_classe': pat_dominante,
                    'motivo': f"Classe a due mani, istanza con {pattern}"
                })

            # Regola 2: classe a una mano → istanza senza mani = errore
            elif pat_dominante in ("SOLO_DX", "SOLO_SX") and pattern == "NESSUNA":
                errori_certi.append({
                    'classe': gloss, 'file': npy_file, 'signer': signer,
                    'pattern_istanza': pattern, 'pattern_classe': pat_dominante,
                    'motivo': "Nessuna mano rilevata in classe a una mano"
                })

            # Regola 3: classe SOLO_DX → istanza SOLO_SX (o viceversa)
            elif pat_dominante == "SOLO_DX" and pattern == "SOLO_SX":
                dom_signer = signer_dominante.get(signer, "SCONOSCIUTA")
                if dom_signer == "MANCINO":
                    ok_mancini.append({
                        'classe': gloss, 'file': npy_file, 'signer': signer,
                        'motivo': "Signer mancino, usa mano SX (coerente)"
                    })
                else:
                    errori_mano_errata.append({
                        'classe': gloss, 'file': npy_file, 'signer': signer,
                        'pattern_istanza': pattern, 'pattern_classe': pat_dominante,
                        'motivo': f"Signer {dom_signer}, ma usa mano SX in classe DX"
                    })

            elif pat_dominante == "SOLO_SX" and pattern == "SOLO_DX":
                dom_signer = signer_dominante.get(signer, "SCONOSCIUTA")
                if dom_signer == "DESTRORSO":
                    ok_mancini.append({
                        'classe': gloss, 'file': npy_file, 'signer': signer,
                        'motivo': "Signer destrorso, usa mano DX (coerente)"
                    })
                else:
                    errori_mano_errata.append({
                        'classe': gloss, 'file': npy_file, 'signer': signer,
                        'pattern_istanza': pattern, 'pattern_classe': pat_dominante,
                        'motivo': f"Signer {dom_signer}, ma usa mano DX in classe SX"
                    })

    # --- Report finale ---
    print("\n" + "=" * 70)
    print("REPORT FINALE")
    print("=" * 70)

    print(f"\nClassi da eliminare (pattern NESSUNA/VUOTA): {len(classi_da_eliminare)}")
    for c in classi_da_eliminare:
        print(f"  - {c}")

    print(f"\nErrori certi (da eliminare): {len(errori_certi)}")
    print(f"  - Istanze in classe ENTRAMBE senza entrambe le mani: "
          f"{sum(1 for e in errori_certi if 'due mani' in e['motivo'])}")
    print(f"  - Istanze senza mani in classe a una mano: "
          f"{sum(1 for e in errori_certi if 'Nessuna mano' in e['motivo'])}")

    print(f"\nErrori mano errata (mano diversa dal signer): {len(errori_mano_errata)}")

    print(f"\nIstanze OK (signer mancino/destrorso coerente): {len(ok_mancini)}")

    # Riepilogo signer mancini
    mancini = [s for s, d in signer_dominante.items() if d == "MANCINO"]
    print(f"\nSigner mancini identificati: {len(mancini)}")
    for s in mancini:
        c = signer_hand_counts[s]
        print(f"  {s}: DX={c['DX']}, SX={c['SX']}")

    # Salva risultati
    # 1. Errori certi
    with open(os.path.join(OUTPUT_DIR, "errori_certi.txt"), 'w') as f:
        f.write(f"# Errori certi da eliminare: {len(errori_certi)} istanze\n")
        f.write("# Formato: CLASSE|FILE|SIGNER|PATTERN_ISTANZA|PATTERN_CLASSE|MOTIVO\n\n")
        for e in errori_certi:
            f.write(f"{e['classe']}|{e['file']}|{e['signer']}|"
                    f"{e['pattern_istanza']}|{e['pattern_classe']}|{e['motivo']}\n")

    # 2. Errori mano errata
    with open(os.path.join(OUTPUT_DIR, "errori_mano_errata.txt"), 'w') as f:
        f.write(f"# Errori mano errata: {len(errori_mano_errata)} istanze\n")
        f.write("# Formato: CLASSE|FILE|SIGNER|PATTERN_ISTANZA|PATTERN_CLASSE|MOTIVO\n\n")
        for e in errori_mano_errata:
            f.write(f"{e['classe']}|{e['file']}|{e['signer']}|"
                    f"{e['pattern_istanza']}|{e['pattern_classe']}|{e['motivo']}\n")

    # 3. Classi da eliminare
    with open(os.path.join(OUTPUT_DIR, "classi_da_eliminare.txt"), 'w') as f:
        f.write(f"# Classi da eliminare: {len(classi_da_eliminare)}\n\n")
        for c in classi_da_eliminare:
            f.write(f"{c}\n")

    # 4. OK mancini
    with open(os.path.join(OUTPUT_DIR, "ok_mancini.txt"), 'w') as f:
        f.write(f"# Istanze coerenti con signer mancino/destrorso: {len(ok_mancini)}\n\n")
        for e in ok_mancini:
            f.write(f"{e['classe']}|{e['file']}|{e['signer']}|{e['motivo']}\n")

    print(f"\nFile salvati in {OUTPUT_DIR}/:")
    print(f"  - errori_certi.txt ({len(errori_certi)} istanze)")
    print(f"  - errori_mano_errata.txt ({len(errori_mano_errata)} istanze)")
    print(f"  - classi_da_eliminare.txt ({len(classi_da_eliminare)} classi)")
    print(f"  - ok_mancini.txt ({len(ok_mancini)} istanze)")
    print("=" * 70)


if __name__ == "__main__":
    main()
