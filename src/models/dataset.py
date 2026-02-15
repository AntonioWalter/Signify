"""
Dataset PyTorch per il riconoscimento della lingua dei segni.

Questo modulo definisce il dataset che carica le sequenze di landmark
estratte dai video ASL e le prepara per l'addestramento del modello.

Struttura dati attesa:
    Dataset_Keypoints_Train/
    ├── ABOUT1/
    │   ├── video1.npy    (shape: num_frames x 258)
    │   ├── video2.npy
    │   └── ...
    ├── HELLO/
    │   └── ...
    └── ...

Ogni file .npy contiene una matrice dove:
    - le righe sono i frame del video
    - le colonne sono i 258 valori dei landmark (132 pose + 63 mano sx + 63 mano dx)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SignLanguageDataset(Dataset):
    """
    Dataset PyTorch per sequenze di landmark della lingua dei segni.

    Per ogni campione:
    1. Carica il file .npy corrispondente
    2. Normalizza la sequenza a un numero fisso di frame (padding o troncamento)
    3. Applica data augmentation se abilitata (solo in fase di training)
    """

    def __init__(self, data_dir, max_frames=30, augment=False):
        """
        Inizializza il dataset scansionando le cartelle delle classi.

        Args:
            data_dir: percorso alla cartella contenente le sotto-cartelle delle classi
            max_frames: numero fisso di frame a cui normalizzare ogni sequenza
            augment: se True, applica data augmentation (usare solo per il training)
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.augment = augment

        # Scansione delle cartelle: ogni sotto-cartella è una classe (un segno ASL)
        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        # Numero totale di classi (segni diversi)
        self.num_classes = len(self.classes)

        # Mapping: nome della classe -> indice numerico
        # Esempio: {"ABOUT1": 0, "HELLO": 1, ...}
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Mapping inverso: indice numerico -> nome della classe
        # Utile per decodificare le predizioni del modello
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # Lista di tutti i campioni: (percorso_file, indice_classe)
        # Scorriamo tutte le cartelle e raccogliamo ogni file .npy
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for filename in os.listdir(cls_dir):
                if filename.endswith('.npy'):
                    filepath = os.path.join(cls_dir, filename)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((filepath, label))

    def __len__(self):
        """Restituisce il numero totale di campioni nel dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Carica e restituisce un singolo campione.

        Args:
            idx: indice del campione da caricare

        Returns:
            sequence: tensore di shape (max_frames, 258) con la sequenza normalizzata
            label: indice numerico della classe
        """
        filepath, label = self.samples[idx]

        # Carica la sequenza di landmark dal file .npy
        # Shape originale: (num_frames_originali, 258)
        sequence = np.load(filepath)

        # Normalizzazione della lunghezza della sequenza
        # Tutti i video hanno un numero diverso di frame, ma il modello
        # ha bisogno di input a dimensione fissa (max_frames)
        sequence = self._normalize_length(sequence)

        # Data augmentation: applicata solo durante il training
        # per aumentare la varietà dei dati e ridurre l'overfitting
        if self.augment:
            sequence = self._augment(sequence)

        # Conversione in tensore PyTorch (float32 per compatibilità con il modello)
        sequence = torch.FloatTensor(sequence)
        label = torch.LongTensor([label]).squeeze()

        return sequence, label

    def _normalize_length(self, sequence):
        """
        Normalizza la sequenza a esattamente max_frames frame.

        Se il video ha meno frame del necessario, aggiungiamo righe di zeri
        alla fine (zero-padding). Se ne ha di più, tronchiamo gli ultimi frame.

        Args:
            sequence: array numpy di shape (num_frames, 258)

        Returns:
            array numpy di shape (max_frames, 258)
        """
        num_frames, num_features = sequence.shape

        if num_frames >= self.max_frames:
            # Troncamento: prendiamo solo i primi max_frames frame
            return sequence[:self.max_frames]
        else:
            # Padding: aggiungiamo righe di zeri per raggiungere max_frames
            padding = np.zeros((self.max_frames - num_frames, num_features))
            return np.vstack([sequence, padding])

    def _augment(self, sequence):
        """
        Applica trasformazioni casuali alla sequenza per aumentare la varietà dei dati.

        Tecniche utilizzate:
        1. Rumore gaussiano: aggiunge piccole perturbazioni casuali ai valori
        2. Scaling: moltiplica i valori per un fattore casuale vicino a 1

        Args:
            sequence: array numpy di shape (max_frames, 258)

        Returns:
            array numpy con le trasformazioni applicate
        """
        # Rumore gaussiano con probabilità del 50%
        # Simula piccole variazioni naturali nel tracciamento dei landmark
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.005, sequence.shape)
            sequence = sequence + noise

        # Scaling casuale con probabilità del 50%
        # Simula variazioni nella distanza dalla telecamera
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            sequence = sequence * scale

        return sequence


def create_data_loaders(data_dir, batch_size=32, max_frames=30, val_size=0.15):
    """
    Crea i DataLoader per training e validazione.

    Il dataset viene suddiviso in due parti:
    - Training (85%): usato per addestrare il modello
    - Validazione (15%): usato per monitorare le prestazioni durante il training

    Il test viene effettuato su un dataset separato (non gestito qui).

    La suddivisione è stratificata: ogni classe mantiene le stesse proporzioni
    in entrambi i sottoinsiemi.

    Args:
        data_dir: percorso alla cartella del dataset
        batch_size: numero di campioni per batch
        max_frames: numero fisso di frame per sequenza
        val_size: proporzione del dataset per la validazione (default: 15%)

    Returns:
        train_loader: DataLoader per il training
        val_loader: DataLoader per la validazione
        dataset: oggetto SignLanguageDataset con le informazioni sulle classi
    """
    # Creiamo il dataset completo (senza augmentation per ora)
    full_dataset = SignLanguageDataset(data_dir, max_frames=max_frames, augment=False)

    # Estraiamo le etichette per lo split stratificato
    labels = [label for _, label in full_dataset.samples]

    # Split stratificato: separiamo training e validazione
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=val_size,
        stratify=labels,
        random_state=42  # Per riproducibilità
    )

    # Creiamo i due dataset con i rispettivi sottoinsiemi di indici
    # Il training set ha l'augmentation abilitata, la validazione no
    train_dataset = torch.utils.data.Subset(
        SignLanguageDataset(data_dir, max_frames=max_frames, augment=True),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"Split del dataset:")
    print(f"  Training:    {len(train_dataset):,} campioni")
    print(f"  Validazione: {len(val_dataset):,} campioni")
    print(f"  Classi:      {full_dataset.num_classes:,}")

    # Creiamo i DataLoader che gestiscono il batching e lo shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # Mescoliamo i dati ad ogni epoca (solo training)
        num_workers=2,       # Processi paralleli per il caricamento
        pin_memory=True      # Ottimizzazione per il trasferimento CPU -> GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, full_dataset


# Blocco di test: eseguilo per verificare che il dataset funzioni correttamente
if __name__ == "__main__":
    DATA_DIR = "data/processed/Dataset_Keypoints_Train"

    print("=" * 60)
    print("TEST DEL DATASET")
    print("=" * 60)

    # Test 1: Creazione del dataset
    dataset = SignLanguageDataset(DATA_DIR)
    print(f"\nCampioni totali: {len(dataset)}")
    print(f"Classi: {dataset.num_classes}")
    print(f"Prime 5 classi: {dataset.classes[:5]}")

    # Test 2: Caricamento di un campione
    sequence, label = dataset[0]
    print(f"\nCampione di esempio:")
    print(f"  Shape sequenza: {sequence.shape}")     # Atteso: (30, 258)
    print(f"  Tipo: {sequence.dtype}")                # Atteso: float32
    print(f"  Label: {label.item()} ({dataset.idx_to_class[label.item()]})")

    # Test 3: Creazione dei DataLoader
    print(f"\n{'─' * 60}")
    train_loader, val_loader, _ = create_data_loaders(DATA_DIR)

    # Test 4: Verifica di un batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch di esempio:")
    print(f"  Input shape: {batch_x.shape}")   # Atteso: (32, 30, 258)
    print(f"  Labels shape: {batch_y.shape}")  # Atteso: (32,)
    print("=" * 60)

