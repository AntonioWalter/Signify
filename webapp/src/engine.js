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

    console.log('[Engine] Scaricamento modello (49MB)...')
    window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'
    
    // Il file viene cachato dal browser per accessi successivi istantanei
    const resModel = await fetch(`${import.meta.env.BASE_URL}models/signify_lstm.onnx`)
    if (!resModel.ok) throw new Error('File ONNX non trovato in public/models/')
    
    // Implementazione del download reader per tracciare la percentuale
    const contentLength = resModel.headers.get('content-length')
    const total = contentLength ? parseInt(contentLength, 10) : 49000000 // stima 49MB
    let loaded = 0
    
    const reader = resModel.body.getReader()
    const chunks = []
    
    while(true) {
      const {done, value} = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.byteLength
      const percent = Math.round((loaded / total) * 100)
      console.log(`[Engine] Download: ${percent}% (${(loaded / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB)`)
      onProgress?.(`Scaricamento modello: ${percent}%`, percent)
    }
    
    console.log('[Engine] Costruzione buffer finale...')
    // Uniamo i chunks
    const modelBuffer = new Uint8Array(loaded)
    let position = 0
    for(let chunk of chunks) {
      modelBuffer.set(chunk, position)
      position += chunk.length
    }
    
    console.log('[Engine] Model Buffer Size:', modelBuffer.byteLength, 'bytes')
    
    if (modelBuffer.byteLength < 1000000) {
      throw new Error(`File modello troppo piccolo (${modelBuffer.byteLength} bytes). Possibile download corrotto.`)
    }

    onProgress?.('Inizializzazione AI (WebGL/WASM)...')
    // Tentativo primario con provider multipli (WebGL preferito, WASM come fallback nativo)
    // Non forziamo numThreads o ottimizzazioni aggressive per evitare blocchi o "SharedArrayBuffer" issues
    try {
      session = await window.ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['webgl', 'wasm']
      })
      console.log('Backend AI (WebGL/WASM) inizializzato con successo')
    } catch (e) {
      console.warn('Inizializzazione fallita, uso WASM di base...', e)
      onProgress?.('WebGL/WASM falliti, ripiego su WASM base...')
      session = await window.ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm']
      })
      console.log('Backend WASM di ripiego inizializzato')
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
