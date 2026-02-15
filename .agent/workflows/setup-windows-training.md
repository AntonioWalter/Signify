---
description: Setup del progetto su Windows e avvio del training del modello LSTM+Attention
---

# Setup Signify su Windows - Guida completa

## Contesto
Questo progetto è stato sviluppato su Mac. I file `.npy` del dataset (5.3GB) sono sulla pennetta KINGSTON nella cartella `Dataset_Keypoints_Train`. Il codice è su GitHub.

## Passi

### 1. Clona la repository
// turbo
```bash
git clone https://github.com/AntonioWalter/Signify.git
cd Signify
```

### 2. Crea il virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Installa PyTorch con supporto CUDA (per la RTX 2070)
**IMPORTANTE**: questo comando installa la versione GPU, non quella CPU.
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Installa le altre dipendenze
// turbo
```bash
pip install numpy scikit-learn tqdm matplotlib
```

### 5. Copia il dataset dalla pennetta
Copia manualmente la cartella `Dataset_Keypoints_Train` dalla pennetta KINGSTON nella posizione:
```
Signify/data/processed/Dataset_Keypoints_Train/
```
Assicurati che la struttura sia:
```
data/processed/Dataset_Keypoints_Train/
├── 1DOLLAR/
│   ├── file1.npy
│   └── ...
├── ABOUT1/
│   └── ...
└── ... (2344 cartelle totali)
```

### 6. Verifica che il dataset funzioni
// turbo
```bash
python -m src.models.dataset
```
Output atteso:
- Campioni totali: ~32.435
- Classi: 2.344
- Shape sequenza: (30, 258)

### 7. Verifica che il modello funzioni
// turbo
```bash
python -m src.models.model
```
Output atteso:
- Output shape: (4, 2344)
- Attention shape: (4, 30)
- Parametri: ~1.160.873

### 8. Avvia il training
```bash
python -m src.training.train
```
Il training:
- Usa automaticamente la GPU (RTX 2070) se disponibile
- Fa early stopping dopo 10 epoche senza miglioramento
- Salva il modello migliore in `checkpoints/best_model.pt`
- Salva le metriche in `checkpoints/training_history.json`

## Note
- Il modello ha ~1.2M parametri, molto leggero per la RTX 2070
- Batch size: 64, Learning rate: 0.001
- Gli iperparametri si cambiano in cima a `src/training/train.py`
- Se vuoi cambiare il numero di frame per sequenza, modifica MAX_FRAMES in `train.py` e il parametro max_frames in `dataset.py`
