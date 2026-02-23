"""
Modello LSTM con meccanismo di Attention per la classificazione dei segni ASL.

Architettura:
    1. LSTM bidirezionale: elabora la sequenza di landmark frame per frame,
       producendo un vettore di output per ciascun frame
    2. Attention Layer: assegna un peso di importanza a ogni frame e calcola
       un vettore di contesto come media pesata degli output
    3. Classificatore: mappa il vettore di contesto in una delle classi ASL

Il meccanismo di attention permette al modello di concentrarsi sui frame
più informativi della sequenza, migliorando sia le prestazioni che
l'interpretabilità del modello.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Layer di Attention per sequenze temporali.

    Riceve gli output dell'LSTM (un vettore per ogni frame) e calcola
    un punteggio di importanza per ciascun frame. I punteggi vengono
    normalizzati con softmax e usati per calcolare la media pesata
    degli output, producendo un singolo vettore di contesto.

    In pratica, il modello impara automaticamente quali frame del video
    sono più rilevanti per riconoscere il segno.
    """

    def __init__(self, hidden_size):
        """
        Args:
            hidden_size: dimensione dei vettori di output dell'LSTM
        """
        super(AttentionLayer, self).__init__()

        # Rete a due strati che calcola il punteggio di attenzione per ogni frame
        # Input: vettore di un frame (hidden_size) -> punteggio scalare (1)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # Riduzione dimensionalità
            nn.Tanh(),                                  # Attivazione non lineare
            nn.Linear(hidden_size // 2, 1)              # Punteggio finale (scalare)
        )

    def forward(self, lstm_outputs):
        """
        Calcola il vettore di contesto pesato con attention.

        Args:
            lstm_outputs: output dell'LSTM, shape (batch_size, num_frames, hidden_size)

        Returns:
            context: vettore di contesto, shape (batch_size, hidden_size)
            weights: pesi di attention, shape (batch_size, num_frames)
                     (utili per la visualizzazione e l'interpretabilità)
        """
        # Calcoliamo un punteggio per ogni frame di ogni sequenza nel batch
        # Shape: (batch_size, num_frames, 1)
        scores = self.attention(lstm_outputs)

        # Rimuoviamo l'ultima dimensione per avere (batch_size, num_frames)
        scores = scores.squeeze(-1)

        # Normalizziamo i punteggi con softmax: i pesi sommano a 1
        # I frame con punteggio alto riceveranno un peso maggiore
        weights = F.softmax(scores, dim=1)

        # Calcoliamo il vettore di contesto come media pesata
        # weights: (batch_size, num_frames) -> (batch_size, num_frames, 1) per il broadcasting
        # lstm_outputs: (batch_size, num_frames, hidden_size)
        # Risultato: (batch_size, hidden_size)
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)

        return context, weights


class LSTMAttention(nn.Module):
    """
    Modello completo LSTM + Attention per la classificazione dei segni ASL.

    Flusso dei dati:
        Sequenza (30 x 258) -> LSTM -> Attention -> Classificatore -> Predizione

    L'LSTM è bidirezionale: legge la sequenza sia in avanti che all'indietro,
    catturando il contesto da entrambe le direzioni temporali.
    """

    def __init__(self, input_size=258, hidden_size=128, num_layers=2,
                 num_classes=2344, dropout=0.3):
        """
        Args:
            input_size: numero di feature per frame (258 landmark)
            hidden_size: dimensione dello stato nascosto dell'LSTM
            num_layers: numero di strati LSTM impilati
            num_classes: numero di classi (segni ASL) da classificare
            dropout: probabilità di dropout per la regolarizzazione
        """
        super(LSTMAttention, self).__init__()

        # Salviamo i parametri per poterli consultare dopo
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM bidirezionale
        # - batch_first=True: il batch è la prima dimensione (batch, frames, features)
        # - bidirectional=True: legge la sequenza in entrambe le direzioni
        # - dropout: applicato tra i layer LSTM (attivo solo se num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # L'output dell'LSTM bidirezionale ha dimensione hidden_size * 2
        # perché concatena le uscite della direzione avanti e indietro
        lstm_output_size = hidden_size * 2

        # Layer di Attention: calcola i pesi di importanza per ogni frame
        self.attention = AttentionLayer(lstm_output_size)
        
        # Projection Head Avanzata (Super LSTM v2)
        # Aumentiamo la profondità per migliorare la separabilità delle 2344 classi
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size * 2),  # Expand capacity
            nn.BatchNorm1d(hidden_size * 2),               # Normalize activations
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),       # Compress features
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)            # Final classification
        )

    def forward(self, x):
        """
        Forward pass del modello.

        Args:
            x: sequenza di input, shape (batch_size, max_frames, input_size)
               Esempio: (32, 30, 258) per un batch di 32 sequenze

        Returns:
            output: logits per ogni classe, shape (batch_size, num_classes)
            attention_weights: pesi di attention, shape (batch_size, max_frames)
        """
        # Passo 1: LSTM elabora la sequenza frame per frame
        # lstm_out contiene l'output per ogni frame: (batch, frames, hidden*2)
        lstm_out, _ = self.lstm(x)

        # Passo 2: Attention seleziona i frame più importanti
        # context è il "riassunto pesato" della sequenza: (batch, hidden*2)
        context, attention_weights = self.attention(lstm_out)

        # Passo 3: Il classificatore produce le predizioni finali
        # output sono i logits (punteggi grezzi) per ogni classe: (batch, num_classes)
        output = self.classifier(context)

        return output, attention_weights



class PositionalEncoding(nn.Module):
    """
    Inietta informazioni sulla posizione temporale nel modello Transformer.
    Poiché il Transformer non ha ricorrenza, non sa "dove" si trova un frame
    nella sequenza (inizio, centro, fine) senza questo encoding.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creazione della matrice di encoding una volta sola
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Termini sinusoidali a frequenze diverse
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Aggiungiamo la dimensione batch: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Buffer: stato non addestrabile che viene salvato col modello
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input embedding, shape (batch_size, seq_len, d_model)
        """
        # Aggiungiamo l'encoding posizionale all'input
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SignTransformer(nn.Module):
    """
    Modello Transformer per il riconoscimento dei segni (SignFormer).
    
    Architettura:
    1. Input Projection: Mappa landmarks (258) -> Model Dimension (es. 256)
    2. Positional Encoding: Aggiunge info temporali
    3. Transformer Encoder: N layer di Self-Attention e Feed-Forward
    4. Classification Head: Mappa l'output (medio o [CLS]) -> Classi
    """
    def __init__(self, input_dim=258, model_dim=256, num_classes=2344, 
                 num_heads=8, num_layers=4, dropout=0.1, max_len=100):
        super(SignTransformer, self).__init__()
        
        self.model_dim = model_dim
        
        # 1. Proiezione lineare dell'input (Feature Embedding)
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_len, dropout=dropout)
        
        # 3. Transformer Encoder Layer
        # batch_first=True: input shape (batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: input sequence (batch_size, seq_len, input_dim)
        """
        # Proiezione input: (batch, seq, 258) -> (batch, seq, 256)
        x = self.input_projection(x)
        
        # Scaling richiesto dall'architettura originale del Transformer ("Attention is All You Need")
        x = x * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32).to(x.device))
        
        # Aggiunta encoding posizionale
        x = self.pos_encoder(x)
        
        # Passaggio nel Transformer Encoder
        # Output: (batch, seq, model_dim)
        x = self.transformer_encoder(x)
        
        # Global Average Pooling: facciamo la media su tutti i frame temporali
        # Un'alternativa è usare solo l'ultimo frame o un token [CLS] speciale, 
        # ma GAP funziona bene per l'action recognition.
        # (batch, seq, model_dim) -> (batch, model_dim)
        context = x.mean(dim=1)
        
        # Classificazione
        output = self.classifier(context)
        
        return output, None  # Ritorniamo None per compatibilità con l'interfaccia train (no attention weights espliciti per ora)


# Blocco di test: verifica che il modello funzioni correttamente
if __name__ == "__main__":
    print("=" * 60)
    print("TEST DEL MODELLO (SignTransformer)")
    print("=" * 60)

    # Parametri del modello
    INPUT_DIM = 258      # Numero di landmark per frame
    MODEL_DIM = 256      # Dimensione interna del modello
    NUM_CLASSES = 2344   # Numero di segni ASL
    BATCH_SIZE = 4       # Dimensione del batch di test
    MAX_FRAMES = 30      # Numero di frame per sequenza
    NUM_HEADS = 8        # Testine di attention
    NUM_LAYERS = 4       # Numero di layer encoder

    # Creazione del modello
    model = SignTransformer(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )

    # Input di test: batch di sequenze casuali
    test_input = torch.randn(BATCH_SIZE, MAX_FRAMES, INPUT_DIM)
    print(f"\nInput shape: {test_input.shape}")

    # Forward pass
    output, _ = model(test_input)
    print(f"Output shape: {output.shape}")              # Atteso: (4, 2344)

    # Conta dei parametri del modello
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParametri totali: {total_params:,}")
    print("=" * 60)
