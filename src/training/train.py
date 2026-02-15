"""
Script di addestramento del modello LSTM+Attention.

Questo script gestisce l'intero processo di training:
1. Caricamento e preparazione del dataset (split train/val/test)
2. Inizializzazione del modello, loss function e optimizer
3. Loop di addestramento con validazione ad ogni epoca
4. Early stopping per evitare overfitting
5. Salvataggio del modello migliore e logging delle metriche
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm

# Aggiungiamo la root del progetto al path per gli import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.dataset import create_data_loaders
from src.models.model import LSTMAttention


# ========================================
# CONFIGURAZIONE
# ========================================

# Percorsi
DATA_DIR = "data/processed/Dataset_Keypoints_Train"
CHECKPOINT_DIR = "checkpoints"

# Iperparametri del modello
INPUT_SIZE = 258        # Feature per frame (landmark)
HIDDEN_SIZE = 128       # Dimensione stato nascosto LSTM
NUM_LAYERS = 2          # Numero di layer LSTM
DROPOUT = 0.3           # Probabilità di dropout

# Iperparametri di training
MAX_FRAMES = 30         # Frame per sequenza
BATCH_SIZE = 64         # Campioni per batch
LEARNING_RATE = 0.001   # Learning rate iniziale
NUM_EPOCHS = 100        # Numero massimo di epoche
PATIENCE = 10           # Epoche senza miglioramento prima di fermarsi


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Esegue una singola epoca di addestramento.

    Per ogni batch:
    1. Sposta i dati sulla GPU (se disponibile)
    2. Calcola le predizioni del modello (forward pass)
    3. Calcola la loss (quanto le predizioni si discostano dalle etichette)
    4. Aggiorna i pesi del modello (backward pass + optimizer step)

    Args:
        model: il modello da addestrare
        train_loader: DataLoader con i dati di training
        criterion: funzione di loss (CrossEntropyLoss)
        optimizer: ottimizzatore (Adam)
        device: dispositivo di calcolo (cpu o cuda)

    Returns:
        avg_loss: loss media sull'intera epoca
        accuracy: percentuale di predizioni corrette
    """
    model.train()  # Modalità training (attiva dropout e batch norm)

    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in tqdm(train_loader, desc="  Training", leave=False):
        # Spostiamo i dati sul dispositivo di calcolo
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Azzeriamo i gradienti del passo precedente
        optimizer.zero_grad()

        # Forward pass: calcoliamo le predizioni
        outputs, _ = model(sequences)  # _ = pesi attention (non servono qui)

        # Calcoliamo la loss
        loss = criterion(outputs, labels)

        # Backward pass: calcoliamo i gradienti
        loss.backward()

        # Aggiornamento dei pesi del modello
        optimizer.step()

        # Accumuliamo le statistiche
        total_loss += loss.item() * sequences.size(0)
        _, predicted = outputs.max(1)     # Prendiamo la classe con il punteggio più alto
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Valuta il modello sul set di validazione.

    Simile al training, ma:
    - Non calcoliamo i gradienti (torch.no_grad())
    - Non aggiorniamo i pesi
    - Il modello è in modalità eval (dropout disattivato)

    Args:
        model: il modello da valutare
        val_loader: DataLoader con i dati di validazione
        criterion: funzione di loss
        device: dispositivo di calcolo

    Returns:
        avg_loss: loss media sulla validazione
        accuracy: percentuale di predizioni corrette
    """
    model.eval()  # Modalità valutazione (disattiva dropout)

    total_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() disabilita il calcolo dei gradienti per risparmiare memoria
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="  Validazione", leave=False):
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs, _ = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * sequences.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    """Funzione principale che orchestra l'intero processo di training."""

    print("=" * 60)
    print("ADDESTRAMENTO MODELLO LSTM + ATTENTION")
    print(f"Avvio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ──────────────────────────────────────
    # 1. DISPOSITIVO DI CALCOLO
    # ──────────────────────────────────────
    # Usiamo la GPU se disponibile, altrimenti la CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nDispositivo: GPU ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nDispositivo: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print(f"\nDispositivo: CPU")

    # ──────────────────────────────────────
    # 2. CARICAMENTO DATASET
    # ──────────────────────────────────────
    print(f"\nCaricamento dataset da: {DATA_DIR}")
    train_loader, val_loader, dataset = create_data_loaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        max_frames=MAX_FRAMES
    )
    num_classes = dataset.num_classes

    # ──────────────────────────────────────
    # 3. INIZIALIZZAZIONE MODELLO
    # ──────────────────────────────────────
    model = LSTMAttention(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(device)

    # Stampa riepilogo del modello
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: LSTM + Attention")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Parametri: {total_params:,}")

    # ──────────────────────────────────────
    # 4. LOSS FUNCTION E OPTIMIZER
    # ──────────────────────────────────────
    # CrossEntropyLoss: standard per la classificazione multi-classe
    criterion = nn.CrossEntropyLoss()

    # Adam: optimizer adattivo, funziona bene nella maggior parte dei casi
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: riduce il learning rate quando la validation loss smette di migliorare
    # Se dopo 5 epoche la loss non migliora, il lr viene moltiplicato per 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         # Monitoriamo la loss (vogliamo che diminuisca)
        factor=0.5,         # Fattore di riduzione del learning rate
        patience=5,         # Epoche da attendere prima di ridurre
        verbose=True        # Stampa quando il lr viene ridotto
    )

    # ──────────────────────────────────────
    # 5. EARLY STOPPING
    # ──────────────────────────────────────
    # Fermiamo il training se la validation loss non migliora per PATIENCE epoche
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Creazione cartella per i checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Log delle metriche per ogni epoca (per i grafici)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # ──────────────────────────────────────
    # 6. LOOP DI ADDESTRAMENTO
    # ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Inizio addestramento ({NUM_EPOCHS} epoche max, patience={PATIENCE})")
    print(f"{'─' * 60}")

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Training su un'epoca
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validazione
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Aggiornamento dello scheduler
        scheduler.step(val_loss)

        # Salvataggio metriche
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Tempo impiegato per l'epoca
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Stampa delle metriche
        print(f"Epoca {epoch:3d}/{NUM_EPOCHS} "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:6.2f}% "
              f"| LR: {current_lr:.6f} "
              f"| {epoch_time:.1f}s")

        # ── Early Stopping ──
        if val_loss < best_val_loss:
            # La validation loss è migliorata: salviamo il modello
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Salvataggio del modello migliore
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'num_classes': num_classes,
                'class_to_idx': dataset.class_to_idx,
                'idx_to_class': dataset.idx_to_class,
                'hyperparameters': {
                    'input_size': INPUT_SIZE,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'max_frames': MAX_FRAMES,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                }
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  ✓ Modello salvato (val_loss migliorata)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\n⚠ Early stopping: la validation loss non migliora "
                      f"da {PATIENCE} epoche")
                break

    # ──────────────────────────────────────
    # 7. RIEPILOGO FINALE
    # ──────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("ADDESTRAMENTO COMPLETATO")
    print(f"{'=' * 60}")
    print(f"  Epoche completate: {epoch}")
    print(f"  Tempo totale: {total_time / 60:.1f} minuti")
    print(f"  Miglior val loss: {best_val_loss:.4f}")
    print(f"  Modello salvato in: {CHECKPOINT_DIR}/best_model.pt")

    # Salvataggio dello storico delle metriche
    history_path = os.path.join(CHECKPOINT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Storico salvato in: {history_path}")

    # Salvataggio del mapping delle classi (utile per l'inferenza)
    mapping_path = os.path.join(CHECKPOINT_DIR, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': dataset.class_to_idx,
            'idx_to_class': dataset.idx_to_class
        }, f, indent=2)
    print(f"  Mapping classi salvato in: {mapping_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
