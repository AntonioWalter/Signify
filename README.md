# Signify - ASL Translation Model

Signify è un modello di Deep Learning basato su un'architettura **LSTM + Attention** per il riconoscimento della Lingua dei Segni Americana (ASL) a partire da dati video processati tramite MediaPipe. Il focus principale del progetto è l'estrazione ultraveloce di landmark spaziali e la modellazione temporale per riconoscere 2.344 segni unici.

## 🗂 Struttura della Repository
La codebase è stata rigorosamente organizzata per isolare la data pipeline dai pesi del modello:
```
Signify/
├── docs/                 # Documentazione del progetto e file di log defintivi
├── src/
│   ├── data_preparation/ # Script per estrarre Keypoints via MediaPipe e filtrare le classi
│   ├── models/           # Architettura (model.py) e DataLoader (dataset.py)
│   ├── training/         # Script per l'addestramento e iterazione
│   └── evaluation/       # Script per la validazione sul Test/Val set reale
├── models_saved/         # (GitIgnored) Pesi del modello (.pt)
├── data/                 # (GitIgnored) Input raw e output .npy
├── results/              # (GitIgnored) File JSON di Evaluation
├── requirements.txt      # Le librerie strette minime per la riproduzione
└── README.md
```

## 🚀 Setup dell'Ambiente
Per garantire che il codice di estrazione sfrutti la CPU in modo ottimale (specialmente su architettura Apple Silicon), è essenziale usare l'ambiente Python isolato e l'API *Legacy* di MediaPipe contenuta nelle dipendenze.

```bash
# Crea un ambiente virtuale (consigliato Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate

# Installa rigorosamente le versioni nel txt (importante: mediapipe==0.10.14)
pip install -r requirements.txt
```

## 📊 Dati ed Estrazione (Data Pipeline)
Se possiedi i video raw (es. ASL Citizen Dataset), questi vanno posizionati in `data/raw/ASL_Citizen/videos/`.

L'intero processo di data preparation esplorativa (estrazione keypoint, pulizia e validazione preliminare) è gestito tramite **Jupyter Notebooks**, i quali offrono un'interfaccia interattiva ideale per il preprocessing.

1. Avvia Jupyter Notebook:
   ```bash
   pip install jupyter
   jupyter notebook notebooks/
   ```
2. **Estrazione Train, Val e Test set:** Esegui i notebook `Extract_Landmarks.ipynb`, `Extract_Landmarks_Test.ipynb` e `Extract_Landmarks_Val.ipynb` in sequenza.
3. **Allineamento Classi:** (Cruciale per rimuovere dai set di validazione segni rari mai visti dal Train). Esegui `Filter_Classes.ipynb`.

## 🧠 Addestramento e Valutazione (Training Pipeline)
Il training viene fatto in mini-batch sfruttando l'architettura `LSTMAttention` (2 livelli + Attention custom).

1. **Addestramento:** Lancia l'addestramento da zero. Genera e salva in automatico `models_saved/best_model.pt`.
   ```bash
   python -m src.training.train
   ```
2. **Valutazione Finale:** Calcola *Top-1* e *Top-5* Accuracy sui set di validazione reali allineati.
   ```bash
   python -m src.training.evaluate
   ```

---
*Per i dettagli architetturali dell'Attention Layer o lo stack tecnologico referenziarsi alla tesi nei documenti PDF.*
