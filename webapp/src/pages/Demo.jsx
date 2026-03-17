import React, { useState, useEffect, useRef, useCallback, useContext } from 'react'
import { predict } from '../engine.js'
import { cleanWord, loadBert, matrixRescoringBert } from '../nlp.js'
import { AppContext } from '../App.jsx'
import SkeletonRenderer from '../components/SkeletonRenderer.jsx'
import { Play, RotateCcw, Sparkles, BrainCircuit, Loader2, ChevronRight, Clapperboard } from 'lucide-react'

// ─── Frasi pre-definite con segni verificati ──────────────────────────────────
const DEMO_PHRASES = [
  { label: '🚗 La ragazza guida l\'auto', signs: ['girl', 'drive', 'car'], emoji: '🚗' },
  { label: '💃 Il ragazzo balla',        signs: ['boy', 'dance'], emoji: '💃' },
  { label: '🐕 Il cane salta',          signs: ['dog1', 'jump'], emoji: '🐕' },
  { label: '🐱 Il gatto beve acqua',     signs: ['cat3', 'drink2', 'water'], emoji: '🐱' },
  { label: '⚽ Il bambino gioca',      signs: ['boy', 'play'], emoji: '⚽' },
]

// Durata approssimativa di ogni segno in ms (30 frame × 40ms cadenza)
const SIGN_DURATION_MS = 30 * 40  // 1200ms per default

export default function Demo() {
  const { translate } = useContext(AppContext)

  // Selezione frase
  const [selectedPhrase, setSelectedPhrase] = useState(null)

  // Stato riproduzione
  const [playState, setPlayState] = useState('idle') // idle | playing | analyzing | done
  const [currentSignIdx, setCurrentSignIdx] = useState(-1)

  // Predizioni accumulate (una per segno)
  const [predictions, setPredictions] = useState([])

  // Frase BERT finale
  const [bertPhrase, setBertPhrase] = useState([])
  const [bertLoading, setBertLoading] = useState(false)

  // Stato caricamento landmark corrente (passa nome segno a SkeletonRenderer)
  const [currentSignName, setCurrentSignName] = useState(null)

  // Timer e continuazione
  const timerRef = useRef(null)
  const playingRef = useRef(false)

  // ─── Cleanup timer ────────────────────────────────────────────────────────
  useEffect(() => () => clearTimeout(timerRef.current), [])

  // ─── Avvio riproduzione ────────────────────────────────────────────────────
  const startDemo = useCallback(async () => {
    if (!selectedPhrase) return
    setPlayState('playing')
    setPredictions([])
    setBertPhrase([])
    setCurrentSignIdx(0)
    setCurrentSignName(selectedPhrase.signs[0])
    playingRef.current = true
  }, [selectedPhrase])

  // ─── Step per ogni segno ──────────────────────────────────────────────────
  const advanceSign = useCallback(async (signIdx, phrase, allPreds) => {
    if (!playingRef.current) return

    const signName = phrase.signs[signIdx]

    // Carichiamo i landmark del segno e li classifichiamo
    let preds = null
    try {
      const res = await fetch(`./landmarks/${signName}.json`)
      const frames = await res.json()
      const WINDOW = 30
      // Campionamento a 30 frame (replicando np.linspace come in Python)
      // indices = np.linspace(0, num_frames - 1, 30).round().astype(int)
      const buf = []
      for (let i = 0; i < WINDOW; i++) {
        // Calcolo indice proporzionale come np.linspace
        const p = i / (WINDOW - 1)
        const exactIdx = p * (frames.length - 1)
        let idx = Math.round(exactIdx)
        
        // Safety bounds
        if (idx < 0) idx = 0
        if (idx >= frames.length) idx = frames.length - 1
        
        buf.push(frames[idx])
      }
      
      preds = await predict(buf)
    } catch (e) {
      console.warn('Errore predizione segno', signName, e)
    }

    const newPreds = [...allPreds, { sign: signName, preds }]
    setPredictions(newPreds)

    const nextIdx = signIdx + 1

    if (nextIdx < phrase.signs.length) {
      // Passiamo al segno successivo dopo la durata dell'animazione
      timerRef.current = setTimeout(() => {
        if (!playingRef.current) return
        setCurrentSignIdx(nextIdx)
        setCurrentSignName(phrase.signs[nextIdx])
        advanceSign(nextIdx, phrase, newPreds)
      }, SIGN_DURATION_MS)
    } else {
      // Ultimo segno: BERT rescoring
      timerRef.current = setTimeout(async () => {
        if (!playingRef.current) return
        setPlayState('analyzing')
        setBertLoading(true)
        try {
          const phraseData = newPreds
            .filter(p => p.preds)
            .map(p => ({ top5: p.preds }))
          if (phraseData.length > 0) {
            const refined = await matrixRescoringBert(phraseData)
            setBertPhrase(refined)
          }
        } catch (e) {
          console.error('BERT Demo Error:', e)
        } finally {
          setBertLoading(false)
          setPlayState('done')
          setCurrentSignName(null)
        }
      }, SIGN_DURATION_MS)
    }
  }, [])

  // ─── Trigger avanzamento quando parte al primo segno ─────────────────────
  useEffect(() => {
    if (playState === 'playing' && currentSignIdx === 0 && selectedPhrase) {
      advanceSign(0, selectedPhrase, [])
    }
  }, [playState, currentSignIdx, selectedPhrase, advanceSign])

  // ─── Reset ────────────────────────────────────────────────────────────────
  const resetDemo = () => {
    playingRef.current = false
    clearTimeout(timerRef.current)
    setPlayState('idle')
    setCurrentSignIdx(-1)
    setCurrentSignName(null)
    setPredictions([])
    setBertPhrase([])
  }

  const getTranslated = (word) => {
    if (!word) return '---'
    return translate(cleanWord(word)).toUpperCase()
  }

  const isPlaying = playState === 'playing' || playState === 'analyzing'

  return (
    <div style={{ width: '100%', maxWidth: '1280px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{ textAlign: 'center', paddingTop: '8px' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <Clapperboard size={28} color="var(--accent-primary)" />
          <h2 style={{ fontSize: '28px', fontWeight: 900, letterSpacing: '-1px', margin: 0 }}>
            Live Demo – Linguaggio dei Segni
          </h2>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '15px', margin: 0 }}>
          Scegli una frase: il manichino la eseguirà e il modello AI la tradurrà in tempo reale
        </p>
      </div>

      {/* ── Selezione Frasi ─────────────────────────────────────────────────── */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <p className="card-title" style={{ margin: '0 0 16px' }}>1 · Seleziona una frase</p>
        <div className="responsive-stack" style={{ flexDirection: 'row', flexWrap: 'wrap', gap: '12px' }}>
          {DEMO_PHRASES.map((phrase, i) => (
            <button
              key={i}
              className="demo-phrase-chip"
              style={{
                background: selectedPhrase === phrase ? 'var(--accent-primary)' : 'var(--bg-card)',
                color: selectedPhrase === phrase ? '#fff' : 'var(--text-primary)',
                border: selectedPhrase === phrase ? 'none' : '1px solid var(--border-glass-strong)',
                boxShadow: selectedPhrase === phrase ? '0 4px 16px rgba(139,92,246,0.4)' : 'none',
              }}
              onClick={() => { resetDemo(); setSelectedPhrase(phrase) }}
              disabled={isPlaying}
            >
              {phrase.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Area Principale ─────────────────────────────────────────────────── */}
      {selectedPhrase && (
        <div className="responsive-grid" style={{ gridTemplateColumns: '1fr 1fr', alignItems: 'start' }}>
          
          {/* Colonna Sinistra: Manichino */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <p className="card-title" style={{ margin: 0 }}>2 · Animazione Segno</p>
              <div style={{ display: 'flex', gap: '8px' }}>
                {/* Progress indicatori */}
                {selectedPhrase.signs.map((s, i) => (
                  <div
                    key={i}
                    style={{
                      width: '10px', height: '10px', borderRadius: '50%',
                      background: i < currentSignIdx ? 'var(--success)'
                               : i === currentSignIdx && isPlaying ? 'var(--accent-primary)'
                               : 'var(--border-glass-strong)',
                      boxShadow: i === currentSignIdx && isPlaying ? '0 0 8px var(--accent-primary)' : 'none',
                      transition: 'all 0.3s ease'
                    }}
                  />
                ))}
              </div>
            </div>

            {/* Skeleton Canvas */}
            <div className="demo-canvas-wrapper" style={{
              border: isPlaying ? '2px solid var(--accent-primary)' : '2px solid var(--border-glass)',
              boxShadow: isPlaying ? '0 0 20px rgba(139,92,246,0.3)' : 'none',
              transition: 'all 0.4s ease',
            }}>
              {currentSignName
                ? <SkeletonRenderer signName={currentSignName} />
                : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '280px', gap: '12px', color: 'var(--text-muted)' }}>
                    {playState === 'done'
                      ? <><div style={{ fontSize: '48px' }}>✅</div><p style={{ fontSize: '14px' }}>Demo completata!</p></>
                      : <><Clapperboard size={48} opacity={0.4} /><p style={{ fontSize: '14px' }}>Avvia la demo per vedere il manichino</p></>
                    }
                  </div>
                )
              }
            </div>

            {/* Segno corrente */}
            {currentSignIdx >= 0 && currentSignIdx < selectedPhrase.signs.length && (
              <div style={{ textAlign: 'center', marginTop: '12px' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px' }}>
                  Segno {currentSignIdx + 1}/{selectedPhrase.signs.length}:
                </span>
                <span style={{ fontSize: '20px', fontWeight: 800, marginLeft: '8px', color: 'var(--accent-primary)' }}>
                  {getTranslated(selectedPhrase.signs[currentSignIdx])}
                </span>
              </div>
            )}

            {/* Controlli */}
            <div style={{ display: 'flex', gap: '10px', marginTop: '16px', justifyContent: 'center' }}>
              {!isPlaying && playState !== 'done' && (
                <button className="ctrl-btn primary" style={{ padding: '10px 24px' }} onClick={startDemo}>
                  <Play size={16} /> Avvia Demo
                </button>
              )}
              {isPlaying && (
                <button className="ctrl-btn" style={{ padding: '10px 24px', opacity: 0.6 }} disabled>
                  <Loader2 size={16} className="animate-spin" /> 
                  {playState === 'analyzing' ? 'BERT Rescoring...' : 'In esecuzione...'}
                </button>
              )}
              {(isPlaying || playState === 'done') && (
                <button className="ctrl-btn danger" style={{ padding: '10px 20px' }} onClick={resetDemo}>
                  <RotateCcw size={16} /> Reset
                </button>
              )}
            </div>
          </div>

          {/* Colonna Destra: Predizioni */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            
            {/* Predizioni per segno */}
            <div className="glass-card" style={{ padding: '20px' }}>
              <p className="card-title" style={{ margin: '0 0 16px' }}>3 · Predizioni AI per Segno</p>
              {predictions.length === 0 && (
                <p style={{ color: 'var(--text-muted)', fontSize: '14px', textAlign: 'center', padding: '20px 0' }}>
                  Le predizioni appariranno durante la riproduzione…
                </p>
              )}
              {predictions.map((item, idx) => (
                <div key={idx} style={{
                  padding: '12px 14px', marginBottom: '10px',
                  background: 'rgba(0,0,0,0.1)', borderRadius: '12px',
                  border: '1px solid var(--border-glass)',
                  animation: 'wordPop 0.3s ease'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>
                      Segno {idx + 1}
                    </span>
                    <span style={{ fontSize: '16px', fontWeight: 800, color: 'var(--accent-primary)' }}>
                      {item.preds ? getTranslated(item.preds[0].word) : '???'}
                    </span>
                  </div>
                  {item.preds && (
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                      {item.preds.slice(0, 3).map((p, pi) => (
                        <span key={pi} style={{
                          fontSize: '11px', padding: '2px 8px', borderRadius: '8px',
                          background: pi === 0 ? 'rgba(139,92,246,0.2)' : 'var(--border-glass)',
                          color: pi === 0 ? 'var(--accent-primary)' : 'var(--text-secondary)',
                          fontWeight: 600
                        }}>
                          {getTranslated(p.word)} {(p.p * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Frase BERT */}
            <div className="glass-card" style={{ padding: '20px' }}>
              <p className="card-title" style={{ margin: '0 0 12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sparkles size={14} color="var(--accent-secondary)" /> 4 · Frase Corretta da BERT
              </p>
              {bertLoading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-secondary)', fontSize: '14px' }}>
                  <Loader2 size={16} className="animate-spin" /> BERT sta elaborando la frase…
                </div>
              )}
              {!bertLoading && bertPhrase.length === 0 && (
                <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                  La frase corretta apparirà al termine della demo…
                </p>
              )}
              {!bertLoading && bertPhrase.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {bertPhrase.map((w, i) => (
                    <span key={i} className="phrase-word" style={{ fontSize: '16px' }}>
                      {getTranslated(w)}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Info segni nella frase */}
            {selectedPhrase && (
              <div className="glass-card" style={{ padding: '16px' }}>
                <p className="card-title" style={{ margin: '0 0 10px' }}>Segni in questa frase</p>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {selectedPhrase.signs.map((s, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      {i > 0 && <ChevronRight size={14} color="var(--text-muted)" />}
                      <span style={{
                        padding: '4px 12px', borderRadius: '20px', fontSize: '13px', fontWeight: 700,
                        background: i === currentSignIdx && isPlaying ? 'var(--accent-primary)' : 'var(--border-glass)',
                        color: i === currentSignIdx && isPlaying ? '#fff' : 'var(--text-secondary)',
                        transition: 'all 0.3s ease'
                      }}>
                        {getTranslated(s)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Placeholder se nessuna frase selezionata ──────────────────────── */}
      {!selectedPhrase && (
        <div className="glass-card" style={{ padding: '60px', textAlign: 'center' }}>
          <Clapperboard size={64} opacity={0.2} style={{ marginBottom: '20px' }} />
          <p style={{ color: 'var(--text-muted)', fontSize: '16px' }}>
            Seleziona una frase qui sopra per avviare la demo
          </p>
        </div>
      )}
    </div>
  )
}
