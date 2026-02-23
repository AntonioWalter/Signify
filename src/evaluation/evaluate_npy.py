"""
Script di valutazione del modello LSTM+Attention sul test set.

Funziona sui file .npy pre-estratti (Dataset_Keypoints_Test/) che devono
essere stati generati prima con extract_landmarks_test.py.

Questo approccio è molto più veloce rispetto a processare i video on-the-fly
perché salta la parte di estrazione landmark con MediaPipe.

La valutazione viene fatta SOLO sulle classi presenti nel nostro training set:
i segni del test set non presenti tra le 2344 classi vengono ignorati.

Uso:
    python -m src.evaluation.evaluate_npy
    python -m src.evaluation.evaluate_npy --test_dir data/processed/Dataset_Keypoints_Test --model checkpoints/best_model.pt

Output:
    - Metriche a schermo (accuracy, F1, precision, recall)
    - File results/evaluation_results.json con tutti i dettagli
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

# Aggiungiamo la root del progetto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.model import LSTMAttention


# ========================================
# CONFIGURAZIONE DEFAULT
# ========================================
DEFAULT_TEST_DIR = "data/processed/Dataset_Keypoints_Test"
DEFAULT_CHECKPOINT = "checkpoints/best_model.pt"
DEFAULT_MAX_FRAMES = 30
RESULTS_DIR = "results"


def load_model(checkpoint_path, device):
    """
    Carica il modello addestrato dal checkpoint.

    Il checkpoint contiene:
    - I pesi del modello addestrati
    - Gli iperparametri usati durante il training
    - Il mapping classi <-> indice (essenziale per interpretare le predizioni)

    Args:
        checkpoint_path: percorso al file .pt del checkpoint
        device: dispositivo di calcolo (cpu o cuda)

    Returns:
        model: modello caricato e pronto per l'inferenza
        checkpoint: dizionario con tutte le info salvate
    """
    print(f"Caricamento modello da: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    hparams = checkpoint['hyperparameters']
    model = LSTMAttention(
        input_size=hparams['input_size'],
        hidden_size=hparams['hidden_size'],
        num_layers=hparams['num_layers'],
        num_classes=checkpoint['num_classes'],
        dropout=hparams['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Modalità valutazione (dropout disattivato)

    print(f"  Classi nel modello: {checkpoint['num_classes']:,}")
    print(f"  Salvato all'epoca:  {checkpoint['epoch']}")
    print(f"  Val accuracy:       {checkpoint['val_acc']:.2f}%")

    return model, checkpoint


def normalize_sequence(sequence, max_frames):
    """
    Normalizza la sequenza a max_frames usando uniform resampling.

    Usa lo stesso metodo del dataset di training: campionamento equidistante
    che preserva l'intero movimento (né troncamento né padding puri).

    Args:
        sequence: array numpy (num_frames, 258)
        max_frames: numero target di frame

    Returns:
        array numpy (max_frames, 258)
    """
    num_frames = sequence.shape[0]
    if num_frames == 0:
        return np.zeros((max_frames, sequence.shape[1]))

    # Indici equidistanti calcolati sul numero di frame originale
    indices = np.linspace(0, num_frames - 1, max_frames).round().astype(int)
    return sequence[indices]


def load_test_samples(test_dir, class_to_idx):
    """
    Carica tutti i campioni dal test set, filtrandoli per le classi
    presenti nel nostro modello.

    Le classi del test set NON presenti nel training vengono ignorate:
    non ha senso valutare il modello su segni che non ha mai visto.

    Args:
        test_dir: percorso alla cartella Dataset_Keypoints_Test/
        class_to_idx: dizionario {nome_classe: indice} del modello

    Returns:
        samples: lista di (percorso_file, indice_classe)
        skipped_classes: set di classi ignorate (non nel training)
    """
    samples = []
    skipped_classes = set()

    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Cartella test non trovata: {test_dir}\n"
            f"Esegui prima: python -m src.data_preparation.extract_landmarks_test"
        )

    # Scansione delle cartelle (una per classe)
    class_dirs = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    found_classes = 0
    for cls_name in class_dirs:
        cls_upper = cls_name.upper().strip()

        # La classe deve essere presente nel training set
        if cls_upper not in class_to_idx:
            skipped_classes.add(cls_upper)
            continue

        found_classes += 1
        label = class_to_idx[cls_upper]
        cls_dir = os.path.join(test_dir, cls_name)

        for filename in os.listdir(cls_dir):
            if filename.endswith('.npy'):
                filepath = os.path.join(cls_dir, filename)
                samples.append((filepath, label))

    return samples, skipped_classes, found_classes


def run_evaluation(model, samples, max_frames, device, batch_size=64):
    """
    Esegue l'inferenza del modello su tutti i campioni del test set.

    Per ogni campione:
    1. Carica il file .npy
    2. Applica uniform resampling a max_frames frame
    3. Fa il forward pass del modello
    4. Registra la predizione

    Args:
        model: modello addestrato in modalità eval
        samples: lista di (percorso_file, etichetta_reale)
        max_frames: numero di frame per sequenza
        device: dispositivo di calcolo
        batch_size: campioni elaborati insieme alla volta

    Returns:
        y_true: lista delle etichette reali
        y_pred: lista delle predizioni del modello
    """
    y_true = []
    y_pred = []

    # Elaboriamo i campioni in batch per efficienza
    for i in tqdm(range(0, len(samples), batch_size), desc="Valutazione"):
        batch_samples = samples[i:i + batch_size]

        batch_sequences = []
        batch_labels = []

        for filepath, label in batch_samples:
            try:
                # Carica e normalizza la sequenza
                sequence = np.load(filepath)
                sequence = normalize_sequence(sequence, max_frames)
                batch_sequences.append(sequence)
                batch_labels.append(label)
            except Exception as e:
                print(f"\n  Errore su {os.path.basename(filepath)}: {e}")
                continue

        if not batch_sequences:
            continue

        # Conversione in tensore e inferenza
        batch_tensor = torch.FloatTensor(np.array(batch_sequences)).to(device)

        with torch.no_grad():
            outputs, _ = model(batch_tensor)
            # Prendiamo la classe con il punteggio più alto (top-1)
            _, predicted = outputs.max(1)

        y_true.extend(batch_labels)
        y_pred.extend(predicted.cpu().numpy().tolist())

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(
        description="Valutazione modello LSTM+Attention sul test set"
    )
    parser.add_argument('--test_dir', default=DEFAULT_TEST_DIR,
                        help=f"Cartella test set .npy (default: {DEFAULT_TEST_DIR})")
    parser.add_argument('--model', default=DEFAULT_CHECKPOINT,
                        help=f"Percorso checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument('--max_frames', type=int, default=DEFAULT_MAX_FRAMES,
                        help=f"Frame per sequenza (default: {DEFAULT_MAX_FRAMES})")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size per l'inferenza (default: 64)")
    args = parser.parse_args()

    print("=" * 60)
    print("VALUTAZIONE MODELLO LSTM+ATTENTION — TEST SET")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ──────────────────────────────────────
    # 1. DISPOSITIVO
    # ──────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDispositivo: {device}")

    # ──────────────────────────────────────
    # 2. CARICAMENTO MODELLO
    # ──────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"\n✗ Checkpoint non trovato: {args.model}")
        sys.exit(1)

    model, checkpoint = load_model(args.model, device)

    # Il mapping classi -> indici salvato durante il training
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = checkpoint['idx_to_class']
    max_frames = checkpoint['hyperparameters']['max_frames']

    # ──────────────────────────────────────
    # 3. CARICAMENTO TEST SET
    # ──────────────────────────────────────
    print(f"\nCaricamento test set da: {args.test_dir}")
    samples, skipped_classes, found_classes = load_test_samples(
        args.test_dir, class_to_idx
    )

    print(f"  Classi nel test set:        {found_classes + len(skipped_classes):,}")
    print(f"  Classi nel modello (usate): {found_classes:,}")
    print(f"  Classi ignorate:            {len(skipped_classes):,}")
    print(f"  Campioni da valutare:       {len(samples):,}")

    if not samples:
        print("\n✗ Nessun campione valido trovato. Verifica il percorso test_dir.")
        sys.exit(1)

    # ──────────────────────────────────────
    # 4. VALUTAZIONE
    # ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    y_true, y_pred = run_evaluation(
        model, samples, max_frames, device, args.batch_size
    )

    # ──────────────────────────────────────
    # 5. CALCOLO METRICHE
    # ──────────────────────────────────────

    # Accuracy (top-1): percentuale di predizioni esatte
    accuracy = accuracy_score(y_true, y_pred)

    # F1, Precision, Recall: usiamo la media pesata per le classi sbilanciate
    # zero_division=0 evita warning per classi senza predizioni
    f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # Top-5 Accuracy: calcolata separatamente con re-inferenza
    # (qui non abbiamo i logits, potremmo aggiungerla in futuro se serve)

    # ──────────────────────────────────────
    # 6. STAMPA RISULTATI
    # ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RISULTATI VALUTAZIONE")
    print(f"{'=' * 60}")
    print(f"  Campioni valutati:  {len(y_true):,}")
    print(f"  Classi valutate:    {found_classes:,}")
    print(f"  {'─' * 50}")
    print(f"  Accuracy:   {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  F1-score:   {f1:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"{'=' * 60}")

    # ──────────────────────────────────────
    # 7. SALVATAGGIO RISULTATI
    # ──────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_dir': args.test_dir,
        'checkpoint': args.model,
        'samples_evaluated': len(y_true),
        'classes_evaluated': found_classes,
        'classes_skipped': len(skipped_classes),
        'metrics': {
            'accuracy': round(accuracy, 6),
            'f1_score': round(f1, 6),
            'precision': round(precision, 6),
            'recall': round(recall, 6),
        },
        'hyperparameters': checkpoint['hyperparameters']
    }

    results_path = os.path.join(RESULTS_DIR, 'test_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Risultati salvati in: {results_path}")

    # Report per classe (solo le top 20 più frequenti nel test)
    report = classification_report(
        y_true, y_pred,
        target_names=[idx_to_class[str(i)] for i in range(len(idx_to_class))
                      if i in set(y_true)],
        labels=list(set(y_true)),
        zero_division=0
    )
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Valutazione del {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report)
    print(f"  Report per classe in: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
