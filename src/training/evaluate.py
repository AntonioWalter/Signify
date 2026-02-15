"""
Script di valutazione del modello LSTM+Attention.

Questo script valuta le prestazioni del modello addestrato sul test set.
Calcola:
- Accuracy e Top-5 accuracy
- Classification report (precision, recall, F1-score)
- Salvataggio dei risultati in un file di log
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

from src.models.dataset import create_data_loaders
from src.models.model import LSTMAttention


# ========================================
# CONFIGURAZIONE
# ========================================

DATA_DIR = "data/processed/Dataset_Keypoints_Train"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
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
    print(f"Caricamento modello da: {checkpoint_path}")

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

    return model, checkpoint


def evaluate(model, test_loader, device):
    """
    Valuta il modello sul test set.

    Calcola l'accuracy standard (top-1) e la top-5 accuracy.
    La top-5 accuracy misura quante volte la classe corretta è tra
    le prime 5 predizioni del modello — utile quando ci sono tante classi simili.

    Args:
        model: modello da valutare
        test_loader: DataLoader con i dati di test
        device: dispositivo di calcolo

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
        for sequences, labels in tqdm(test_loader, desc="Valutazione"):
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
    accuracy_top1 = 100.0 * correct_top1 / total
    accuracy_top5 = 100.0 * correct_top5 / total

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
    print("VALUTAZIONE MODELLO LSTM + ATTENTION")
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
    print(f"\nDispositivo: {device}")

    # ──────────────────────────────────────
    # 2. CARICAMENTO MODELLO
    # ──────────────────────────────────────
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n✗ Checkpoint non trovato: {CHECKPOINT_PATH}")
        print("  Esegui prima lo script di training (train.py)")
        return

    model, checkpoint = load_model(CHECKPOINT_PATH, device)
    idx_to_class = checkpoint['idx_to_class']

    # ──────────────────────────────────────
    # 3. CARICAMENTO TEST SET
    # ──────────────────────────────────────
    hparams = checkpoint['hyperparameters']
    _, _, test_loader, _ = create_data_loaders(
        DATA_DIR,
        batch_size=hparams['batch_size'],
        max_frames=hparams['max_frames']
    )

    # ──────────────────────────────────────
    # 4. VALUTAZIONE
    # ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Valutazione sul test set...")
    results, all_preds, all_labels = evaluate(model, test_loader, device)

    # ──────────────────────────────────────
    # 5. RISULTATI
    # ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RISULTATI")
    print(f"{'=' * 60}")
    print(f"  Campioni di test: {results['total_samples']:,}")
    print(f"  Top-1 Accuracy:   {results['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:   {results['top5_accuracy']:.2f}%")

    # ──────────────────────────────────────
    # 6. SALVATAGGIO RISULTATI
    # ──────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Salvataggio metriche in JSON
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    results_to_save = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': CHECKPOINT_PATH,
        'metrics': results,
        'hyperparameters': checkpoint['hyperparameters']
    }

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n  Risultati salvati in: {results_path}")

    # Salvataggio predizioni (utile per analisi successive)
    preds_path = os.path.join(RESULTS_DIR, 'predictions.npz')
    np.savez(preds_path, predictions=all_preds, labels=all_labels)
    print(f"  Predizioni salvate in: {preds_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
