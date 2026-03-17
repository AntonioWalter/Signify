/**
 * nlp.js
 * Modulo per il raffinamento linguistico delle predizioni tramite BERT.
 * Implementa l'algoritmo "Matrix of Solutions" (N-Best Rescoring)
 * sfruttando DistilBERT (via Transformers.js) per la coerenza grammaticale.
 */

// Cache per il pipeline di BERT
let filler = null;
let Transformers = null;

/**
 * Carica dinamicamente Transformers.js e il modello DistilBERT.
 * L'utilizzo del dynamic import evita che l'intero bundle fallisca se Transformers.js
 * ha problemi di inizializzazione nel browser.
 */
export async function loadBert() {
  if (filler) return filler;
  try {
    console.log('Caricamento Transformers.js dal browser...');
    if (!Transformers) {
        // Caricamento asincrono per evitare di bloccare il thread principale (boot istantaneo)
        Transformers = await import('@xenova/transformers');
        
        // Configuriamo l'ambiente Transformers.js per la massima robustezza
        Transformers.env.allowLocalModels = false;
        Transformers.env.useBrowserCache = true;
        // Ottimizzazione: disabilitiamo il worker WASM multithread se ort-web è già attivo
        // Questo previene conflitti con il modello LSTM delle mani
        if (Transformers.env.backends?.onnx?.wasm) {
            Transformers.env.backends.onnx.wasm.numThreads = 1;
        }
    }
    
    console.log('Caricamento DistilBERT (fill-mask)...');
    filler = await Transformers.pipeline('fill-mask', 'Xenova/distilbert-base-uncased');
    return filler;
  } catch (error) {
    console.error('Errore nel caricamento di BERT/transformers:', error);
    // In caso di errore, resettiamo Transformers per riprovare al prossimo giro
    Transformers = null;
    return null;
  }
}

/**
 * Implementazione dell'algoritmo "Matrix Rescoring" (2-pass) dal notebook.
 * @param {Array} phraseData - Array di oggetti { top5: [ {word, p}, ... ] } per ogni segno.
 * @returns {Array} - Array di stringhe (parole corrette).
 */
export async function matrixRescoringBert(phraseData, passes = 2) {
  const model = await loadBert();
  if (!model || phraseData.length === 0) {
    // Fallback alla visione pura (Top-1)
    return phraseData.map(d => d.top5[0].word);
  }

  // Passaggio 0: Inizializziamo con il miglior candidato visivo (Top-1)
  let currentBestPhrase = phraseData.map(d => d.top5[0].word.toLowerCase());

  for (let p = 0; p < passes; p++) {
    const newBestPhrase = [];

    for (let i = 0; i < phraseData.length; i++) {
        const candidates = phraseData[i].top5.map(c => c.word.toLowerCase());
        
        // Creiamo il contesto con il [MASK] nella posizione corrente
        const context = [...currentBestPhrase];
        context[i] = '[MASK]';
        const sentence = context.join(' ');

        try {
            // Chiamiamo BERT per predire i candidati per il [MASK]
            const preds = await model(sentence, { top_k: 100 });
            
            // Creiamo un dizionario di score da BERT
            const bertScores = {};
            preds.forEach(pr => {
                const token = pr.token_str.trim().toLowerCase();
                bertScores[token] = pr.score;
            });

            let bestWord = candidates[0];
            let maxTotalScore = -1.0;

            for (const cand of candidates) {
                const cleanCand = cleanWord(cand).toLowerCase();
                let s = bertScores[cleanCand] || 0.0;
                
                // Bias visivo minimo (come nel notebook)
                if (cand === candidates[0]) s += 0.005;

                if (s > maxTotalScore) {
                    maxTotalScore = s;
                    bestWord = cand;
                }
            }
            newBestPhrase.push(bestWord);
        } catch (e) {
            console.warn('BERT error for step', i, e);
            newBestPhrase.push(candidates[0]);
        }
    }
    currentBestPhrase = [...newBestPhrase];
  }

  return currentBestPhrase;
}

/**
 * Funzione helper per ripulire i nomi delle classi (rimuove numeri finali)
 */
export function cleanWord(word) {
  if (!word) return "";
  // Rimuove eventuali numeri alla fine del nome classe (es. "APPLE1" -> "APPLE")
  return word.replace(/[0-9]/g, '');
}

/**
 * Classe per gestire la coda di segni prima di lanciare BERT.
 * BERT viene applicato sull'intera frase una volta accumulati i segni.
 */
export class SignMatrixBuffer {
    constructor() {
        this.buffer = []; // Array di top5
    }

    addStep(topCandidates) {
        this.buffer.push({ top5: topCandidates });
    }

    getRawPhrase() {
        return this.buffer.map(d => d.top5[0].word);
    }

    async getCorrectedPhrase() {
        if (this.buffer.length === 0) return [];
        return await matrixRescoringBert(this.buffer);
    }

    reset() {
        this.buffer = [];
    }
    
    get length() {
        return this.buffer.length;
    }
}
