// Utilizziamo il bundle globale caricato tramite CDN in index.html
// @ts-ignore
const ort = window.ort;

let session = null
let vocab = null

export async function loadModel(onProgress) {
  try {
    onProgress?.('Verifica ambiente...')
    if (!window.ort) {
      throw new Error('ONNX Runtime non caricato correttamente. Controlla la connessione internet.')
    }
    console.log('ORT Version:', window.ort.version)

    onProgress?.('Caricamento vocabolario...')
    const resVocab = await fetch(`${import.meta.env.BASE_URL}models/vocab.json`)
    if (!resVocab.ok) throw new Error('Errore nel caricamento del vocabolario')
    vocab = await resVocab.json()

    onProgress?.('Scaricamento modello (49MB)...')
    window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'
    window.ort.env.wasm.numThreads = 1 // Forza single-thread per problemi di SharedArrayBuffer su GitHub Pages
    
    // Proviamo a bypassare la cache per assicurarci di avere il file corretto
    const resModel = await fetch(`${import.meta.env.BASE_URL}models/signify_lstm.onnx`, { cache: 'no-store' })
    if (!resModel.ok) throw new Error('File ONNX non trovato in public/models/')
    
    const modelBuffer = await resModel.arrayBuffer()
    console.log('Model Buffer Size:', modelBuffer.byteLength, 'bytes')
    
    if (modelBuffer.byteLength < 1000000) {
      throw new Error(`File modello troppo piccolo (${modelBuffer.byteLength} bytes). Possibile download corrotto.`)
    }

    onProgress?.('Inizializzazione AI (WebGL/WASM)...')
    // Tentativo primario con WASM (più stabile su tutti i PC e laptop)
    try {
      session = await window.ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      })
      console.log('Backend WASM inizializzato con successo')
    } catch (e) {
      console.warn('WASM Fallito, provo WebGL...', e)
      onProgress?.('WASM fallito, provo WebGL...')
      session = await window.ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['webgl']
      })
      console.log('Backend WebGL inizializzato')
    }
    
    onProgress?.('Sistema Pronto!')
    return true
  } catch (err) {
    console.error('Errore Caricamento:', err)
    let msg = err.message
    if (msg.includes('irVersion')) msg = "Errore integrità modello (irVersion). Prova a svuotare la cache del browser."
    throw new Error(msg)
  }
}

export async function predict(landmarkBuffer) {
  if (!session || !vocab || landmarkBuffer.length < 30) return null

  // Take last 30 frames
  const window = landmarkBuffer.slice(-30)
  const flat = window.flat()
  const tensor = new ort.Tensor('float32', Float32Array.from(flat), [1, 30, 258])

  const result = await session.run({ landmarks: tensor })
  const logits = result.logits.data // Float32Array of 2344 values

  // Softmax + Top-5
  const max = Math.max(...logits)
  const exp = Array.from(logits).map(v => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  const probs = exp.map(v => v / sum)

  const indexed = probs.map((p, i) => ({ word: vocab[String(i)] || `cls_${i}`, p }))
  indexed.sort((a, b) => b.p - a.p)

  return indexed.slice(0, 5)
}

export function getVocabWords() {
  if (!vocab) return []
  return Object.values(vocab).filter(w => w && w.length > 2)
}
