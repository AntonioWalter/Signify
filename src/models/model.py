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

        # Classificatore finale: mappa il vettore di contesto nelle classi
        # Uso un piccolo MLP con dropout per evitare overfitting
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
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


# Blocco di test: verifica che il modello funzioni correttamente
if __name__ == "__main__":
    print("=" * 60)
    print("TEST DEL MODELLO")
    print("=" * 60)

    # Parametri del modello
    INPUT_SIZE = 258     # Numero di landmark per frame
    HIDDEN_SIZE = 128    # Dimensione stato nascosto LSTM
    NUM_CLASSES = 2344   # Numero di segni ASL
    BATCH_SIZE = 4       # Dimensione del batch di test
    MAX_FRAMES = 30      # Numero di frame per sequenza

    # Creazione del modello
    model = LSTMAttention(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    )

    # Input di test: batch di sequenze casuali
    test_input = torch.randn(BATCH_SIZE, MAX_FRAMES, INPUT_SIZE)
    print(f"\nInput shape: {test_input.shape}")

    # Forward pass
    output, attention_weights = model(test_input)
    print(f"Output shape: {output.shape}")              # Atteso: (4, 2344)
    print(f"Attention shape: {attention_weights.shape}")  # Atteso: (4, 30)

    # Verifica che i pesi di attention sommino a 1 per ogni campione
    attn_sums = attention_weights.sum(dim=1)
    print(f"Somma pesi attention: {attn_sums}")          # Atteso: ~1.0 per ogni campione

    # Conta dei parametri del modello
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParametri totali: {total_params:,}")
    print(f"Parametri addestrabili: {trainable_params:,}")
    print("=" * 60)
