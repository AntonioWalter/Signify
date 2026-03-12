"""
Script di valutazione del modello LSTM+Attention sui dataset REALI di Test e Validation.

Questo script:
1. Carica il modello addestrato dal checkpoint
2. Carica i dataset Test e Val REALI (non uno split interno del train)
3. Allinea le classi usando il mapping class_to_idx salvato nel checkpoint
4. Calcola: Accuracy, Top-5 Accuracy, Classification Report
5. Salva tutti i risultati in results/
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Aggiungiamo la root del progetto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.dataset import SignLanguageDataset
from src.models.model import LSTMAttention
from torch.utils.data import DataLoader


# ========================================
# CONFIGURAZIONE
# ========================================

TRAIN_DIR = "data/processed/Dataset_Keypoints_Train"
TEST_DIR = "data/processed/Dataset_Keypoints_Test"
VAL_DIR = "data/processed/Dataset_Keypoints_Val"
CHECKPOINT_PATH = "models_saved/best_model.pt"
RESULTS_DIR = "results"


def load_model(checkpoint_path, device):
    """
    Carica il modello salvato durante il training.

    Il checkpoint contiene:
    - I pesi del modello (state_dict)
    - Gli iperparametri usati per la creazione del modello
    - Il mapping classi <-> indici
    - Le metriche di validazione al momento del salvataggio

    Args:
        checkpoint_path: percorso al file .pt del checkpoint
        device: dispositivo su cui caricare il modello (cpu/cuda)

    Returns:
        model: modello con i pesi caricati
        checkpoint: dizionario con tutte le informazioni salvate
    """
    print(f"  Caricamento modello da: {checkpoint_path}")

    # Carichiamo il checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Estraiamo gli iperparametri salvati durante il training
    hparams = checkpoint['hyperparameters']

    # Ricreiamo il modello con gli stessi iperparametri
    model = LSTMAttention(
        input_size=hparams['input_size'],
        hidden_size=hparams['hidden_size'],
        num_layers=hparams['num_layers'],
        num_classes=checkpoint['num_classes'],
        dropout=hparams['dropout']
    ).to(device)

    # Carichiamo i pesi addestrati
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  Epoca di salvataggio: {checkpoint['epoch']}")
    print(f"  Val accuracy al salvataggio: {checkpoint['val_acc']:.2f}%")
    print(f"  Classi nel modello: {checkpoint['num_classes']}")

    return model, checkpoint


def create_eval_dataset(data_dir, class_to_idx, max_frames=30):
    """
    Crea un dataset di valutazione usando il mapping classi del checkpoint.

    IMPORTANTE: usiamo lo stesso class_to_idx del training per garantire
    che l'indice 0 corrisponda alla stessa classe sia nel modello che nei dati.
    Le classi presenti nei dati ma NON nel mapping vengono ignorate.

    Args:
        data_dir: percorso alla cartella con i dati (Test o Val)
        class_to_idx: mapping {nome_classe: indice} dal checkpoint
        max_frames: numero di frame per sequenza

    Returns:
        dataset: SignLanguageDataset configurato
        skipped: numero di classi saltate (non presenti nel training)
    """
    dataset = SignLanguageDataset(data_dir, max_frames=max_frames, augment=False)

    # Riallineiamo i sample con il mapping del checkpoint
    # I sample sono tuple (filepath, old_label_idx)
    aligned_samples = []
    skipped_classes = set()

    for filepath, old_idx in dataset.samples:
        class_name = dataset.idx_to_class[old_idx]
        if class_name in class_to_idx:
            new_idx = class_to_idx[class_name]
            aligned_samples.append((filepath, new_idx))
        else:
            skipped_classes.add(class_name)

    dataset.samples = aligned_samples
    dataset.class_to_idx = class_to_idx
    dataset.idx_to_class = {v: k for k, v in class_to_idx.items()}
    dataset.num_classes = len(class_to_idx)

    return dataset, len(skipped_classes)


def evaluate(model, data_loader, device, desc="Valutazione"):
    """
    Valuta il modello su un DataLoader.

    Calcola l'accuracy standard (top-1) e la top-5 accuracy.
    La top-5 accuracy misura quante volte la classe corretta è tra
    le prime 5 predizioni del modello.

    Args:
        model: modello da valutare
        data_loader: DataLoader con i dati
        device: dispositivo di calcolo
        desc: descrizione per la barra di progresso

    Returns:
        results: dizionario con le metriche calcolate
        all_preds: lista di tutte le predizioni
        all_labels: lista di tutte le etichette reali
    """
    model.eval()  # Modalità valutazione

    all_preds = []
    all_labels = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in tqdm(data_loader, desc=desc):
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, _ = model(sequences)

            # Top-1: la classe con il punteggio più alto
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(labels).sum().item()

            # Top-5: controlliamo se la classe corretta è tra le prime 5
            _, top5_pred = outputs.topk(5, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top5_pred[i]:
                    correct_top5 += 1

            total += labels.size(0)

            # Salviamo predizioni e etichette per il report dettagliato
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcolo delle metriche
    accuracy_top1 = 100.0 * correct_top1 / total if total > 0 else 0
    accuracy_top5 = 100.0 * correct_top5 / total if total > 0 else 0

    results = {
        'top1_accuracy': accuracy_top1,
        'top5_accuracy': accuracy_top5,
        'total_samples': total,
        'correct_top1': correct_top1,
        'correct_top5': correct_top5
    }

    return results, all_preds, all_labels


def main():
    """Funzione principale di valutazione."""

    print("=" * 60)
    print("VALUTAZIONE MODELLO SU TEST E VALIDATION REALI")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ──────────────────────────────────────
    # 1. DISPOSITIVO DI CALCOLO
    # ──────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n  Dispositivo: {device}")

    # ──────────────────────────────────────
    # 2. CARICAMENTO MODELLO
    # ──────────────────────────────────────
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n  ✗ Checkpoint non trovato: {CHECKPOINT_PATH}")
        print("    Esegui prima lo script di training (train.py)")
        return

    model, checkpoint = load_model(CHECKPOINT_PATH, device)
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = checkpoint['idx_to_class']
    hparams = checkpoint['hyperparameters']

    # ──────────────────────────────────────
    # 3. CARICAMENTO DATASET REALI
    # ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Caricamento dataset reali...")

    all_results = {}

    for dataset_name, data_dir in [("Test", TEST_DIR), ("Val", VAL_DIR)]:
        if not os.path.exists(data_dir):
            print(f"\n  ⚠ {dataset_name} non trovato: {data_dir}")
            continue

        dataset, skipped = create_eval_dataset(data_dir, class_to_idx, max_frames=hparams['max_frames'])
        loader = DataLoader(
            dataset,
            batch_size=hparams['batch_size'],
            shuffle=False,
            num_workers=hparams.get('num_workers', 2)
        )

        print(f"\n  {dataset_name} Set:")
        print(f"    Campioni: {len(dataset):,}")
        if skipped > 0:
            print(f"    Classi ignorate (non nel train): {skipped}")

        # ──────────────────────────────────
        # 4. VALUTAZIONE
        # ──────────────────────────────────
        results, all_preds, all_labels = evaluate(model, loader, device, desc=f"  {dataset_name}")

        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │ {dataset_name} Set — RISULTATI               │")
        print(f"  ├─────────────────────────────────────┤")
        print(f"  │ Campioni:       {results['total_samples']:>10,}          │")
        print(f"  │ Top-1 Accuracy: {results['top1_accuracy']:>10.2f}%         │")
        print(f"  │ Top-5 Accuracy: {results['top5_accuracy']:>10.2f}%         │")
        print(f"  └─────────────────────────────────────┘")

        all_results[dataset_name.lower()] = results

        # Salvataggio predizioni per analisi successive
        os.makedirs(RESULTS_DIR, exist_ok=True)
        preds_path = os.path.join(RESULTS_DIR, f'predictions_{dataset_name.lower()}.npz')
        np.savez(preds_path, predictions=all_preds, labels=all_labels)

    # ──────────────────────────────────────
    # 5. SALVATAGGIO RISULTATI COMPLESSIVI
    # ──────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    results_to_save = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': CHECKPOINT_PATH,
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_val_acc': checkpoint['val_acc'],
        'num_classes': checkpoint['num_classes'],
        'metrics': all_results,
        'hyperparameters': hparams
    }

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Risultati salvati in: {results_path}")
    print(f"  Predizioni salvate in: {RESULTS_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
