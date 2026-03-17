import React, { useRef, useState, useCallback, useEffect, useContext } from 'react'
import { predict } from '../engine.js'
import { useMediapipe } from '../useMediapipe.js'
import { AppContext } from '../App.jsx'
import { SignMatrixBuffer, cleanWord, loadBert } from '../nlp.js'
import { Camera, PlaySquare, Square, MessageSquare, Trash2, Bot, ListOrdered, Activity, Hash, Layers, Frame, Zap, BrainCircuit, Sparkles, Loader2, RefreshCcw } from 'lucide-react'

const WINDOW_SIZE = 30
const MOTION_START_THRESHOLD = 0.015   
const MOTION_STOP_THRESHOLD  = 0.006 // Soglia per silenzio assoluto
const VALLEY_THRESHOLD       = 0.4   // Trigger se motion scende sotto il 40% del picco recente
const FRAMES_TO_STOP = 8             // Ridotto da 12 per maggiore reattività
const MAX_CAPTURE_FRAMES = 80        // Forza break dopo ~2.6 secondi (30fps)

function handMotion(prevFrame, currFrame) {
  if (!prevFrame || !currFrame) return 0
  let sum = 0
  for (let i = 132; i < 258; i++) {
    const d = (currFrame[i] || 0) - (prevFrame[i] || 0)
    sum += d * d
  }
  return Math.sqrt(sum / 126)
}

export default function Translator() {
  const { translate } = useContext(AppContext)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  
  // Segmentation Refs
  const prevFrameRef    = useRef(null)
  const captureRef      = useRef([])       
  const isRecordingRef  = useRef(false)    
  const quietFramesRef  = useRef(0)        
  const matrixRef       = useRef(new SignMatrixBuffer())
  const isBertBusyRef   = useRef(false)
  const motionPeakRef   = useRef(0)        // Traccia il picco di movimento della sessione corrente

  // State
  const [active, setActive] = useState(false)
  const [sourceType, setSourceType] = useState('webcam') // 'webcam' or 'file'
  const [videoUrl, setVideoUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle, recording, analyzing, bert
  const [motionLevel, setMotionLevel] = useState(0)
  const [top5, setTop5] = useState([])
  const [phrase, setPhrase] = useState([])
  const [frameCount, setFrameCount] = useState(0)
  const [bertLoading, setBertLoading] = useState(false)
  const [bertReady, setBertReady] = useState(false)

  // Pre-load BERT
  useEffect(() => {
    setBertLoading(true)
    loadBert().then(() => {
        setBertReady(true)
        setBertLoading(false)
    })
  }, [])

  useEffect(() => {
    if (!active) {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop())
        videoRef.current.srcObject = null
      }
      isRecordingRef.current = false
      captureRef.current = []
      setStatus('idle')
      return
    }

    if (sourceType === 'webcam') {
      navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
        .then(stream => {
          if (videoRef.current) {
            // Reset src in case we were playing a file before
            videoRef.current.pause()
            videoRef.current.removeAttribute('src')
            videoRef.current.load()
            videoRef.current.srcObject = stream
            videoRef.current.play().catch(e => console.warn('Webcam play blocked:', e))
          }
        })
        .catch(err => console.error('MediaDevices Error:', err))
    } else if (sourceType === 'file' && videoUrl) {
      if (videoRef.current) {
        // Reset srcObject in case we were using webcam before
        if (videoRef.current.srcObject) {
          videoRef.current.srcObject.getTracks().forEach(t => t.stop())
          videoRef.current.srcObject = null
        }
        videoRef.current.pause()
        videoRef.current.src = videoUrl
        videoRef.current.loop = true
        videoRef.current.load()
        videoRef.current.play().catch(e => console.warn('Video file play blocked:', e))
      }
    }
  }, [active, sourceType, videoUrl])

  // Gestione URL Video (Blob) e pulizia memoria
  useEffect(() => {
    return () => {
       if (videoUrl) {
         console.log('Revoca URL blob:', videoUrl)
         URL.revokeObjectURL(videoUrl)
       }
    }
  }, [videoUrl])

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setVideoUrl(url)
      setSourceType('file')
      setActive(true)
      // Resettiamo l'input per permettere di ricaricare lo stesso file
      e.target.value = ''
    }
  }

  const classifyAuto = useCallback(async (frames) => {
    setStatus('analyzing')

    // Campionamento forzato a WINDOW_SIZE (30 frame)
    // Garantiamo che l'output sia SEMPRE di 30 frame, anche se l'input è più corto
    const buf = Array.from({ length: WINDOW_SIZE }, (_, i) => {
        const idx = Math.min(Math.floor(i * frames.length / WINDOW_SIZE), frames.length - 1)
        return frames[idx]
    })

    const preds = await predict(buf)
    if (preds) {
      setTop5(preds)
      
      // Aggiungiamo i candidati alla matrice
      matrixRef.current.addStep(preds)
      
      // Lanciamo il ricalcolo BERT sulla frase intera (solo se non siamo già occupati)
      if (!isBertBusyRef.current) {
        isBertBusyRef.current = true
        setStatus('bert')
        try {
          const refinedPhrase = await matrixRef.current.getCorrectedPhrase()
          setPhrase(refinedPhrase)
        } catch (err) {
          console.error('BERT Rescoring Error:', err)
        } finally {
          isBertBusyRef.current = false
          setStatus('idle')
        }
      }
    } else {
      setStatus('idle')
    }
  }, [])

  const onFrame = useCallback((features) => {
    setFrameCount(f => f + 1)
    
    // Calcolo movimento istantaneo
    const motion = handMotion(prevFrameRef.current, features)
    prevFrameRef.current = features
    setMotionLevel(motion)

    // Logica di segmentazione automatica migliorata
    if (!isRecordingRef.current) {
      if (motion > MOTION_START_THRESHOLD) {
        isRecordingRef.current = true
        quietFramesRef.current = 0
        motionPeakRef.current = motion
        captureRef.current = [features]
        setStatus('recording')
      }
    } else {
      captureRef.current.push(features)
      
      // Aggiorniamo il picco di movimento della sessione attuale
      if (motion > motionPeakRef.current) motionPeakRef.current = motion

      // LOGICA DI STOP 1: Silenzio assoluto (classico)
      const isQuiet = motion < MOTION_STOP_THRESHOLD
      
      // LOGICA DI STOP 2: Motion Valley (per esperti che fluiscono)
      // Se il movimento scende significativamente rispetto al picco di questo segno
      const isValley = motion < (motionPeakRef.current * VALLEY_THRESHOLD) && motionPeakRef.current > (MOTION_START_THRESHOLD * 2)

      if (isQuiet || isValley) {
        quietFramesRef.current += 1
        // Se siamo in una "valle" o in "silenzio" per abbastanza frame, terminiamo il segno
        if (quietFramesRef.current >= FRAMES_TO_STOP) {
          isRecordingRef.current = false
          const captured = [...captureRef.current]
          captureRef.current = []
          motionPeakRef.current = 0
          classifyAuto(captured)
        }
      } else {
        quietFramesRef.current = 0
      }

      // LOGICA DI STOP 3: Hard Break (Limite temporale)
      if (captureRef.current.length >= MAX_CAPTURE_FRAMES) {
         isRecordingRef.current = false
         const captured = [...captureRef.current]
         captureRef.current = []
         motionPeakRef.current = 0
         classifyAuto(captured)
      }
    }
  }, [classifyAuto])

  useMediapipe({ videoRef, canvasRef, onFrame, enabled: active })

  const clearPhrase = () => {
    matrixRef.current.reset()
    setPhrase([])
    setTop5([])
  }

  const getCleanTranslatedWord = (word) => {
    if (!word) return '---'
    return translate(cleanWord(word)).toUpperCase()
  }

  return (
    <div style={{ width: '100%', maxWidth: '1200px', display: 'flex', flexDirection: 'column' }}>
      
      <div className="translator-layout responsive-stack">
        
        {/* BERT Status Bar */}
        {(bertLoading || !bertReady) && (
          <div className="glass-card" style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', gap: '12px', borderLeft: '4px solid #f59e0b' }}>
            <Loader2 className="animate-spin" size={20} color="#f59e0b" />
            <span style={{ fontSize: '14px', fontWeight: 600 }}>Inizializzazione BERT Modello Linguistico... (Potrebbe richiedere un istante la prima volta)</span>
          </div>
        )}

        {/* Top Row: Webcam and Current Sign */}
        <div className="responsive-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
          
          {/* Left: Webcam */}
          <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <p className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Camera size={16} /> Traduzione Live
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                 <div className="meter-bg" style={{ width: '60px', height: '4px' }}>
                    <div className="meter-fill" style={{ width: `${Math.min(motionLevel/MOTION_START_THRESHOLD*100, 100)}%`, background: motionLevel > MOTION_START_THRESHOLD ? '#ef4444' : '#10b981'}} />
                 </div>
                 <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 700 }}>{status.toUpperCase()}</span>
              </div>
            </div>

            <div className="webcam-container" style={{ flex: 1, minHeight: 0, padding: 0, overflow: 'hidden', position: 'relative' }}>
              <video 
                ref={videoRef} 
                playsInline 
                muted 
                style={{ 
                  display: active ? 'block' : 'none',
                  // Specchia solo la webcam, non i video caricati
                  transform: sourceType === 'webcam' ? 'scaleX(-1)' : 'none'
                }} 
              />
              <canvas ref={canvasRef} width={640} height={480} style={{ display: active ? 'block' : 'none' }} />
              {!active && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '16px', padding: '40px', color: 'var(--text-muted)' }}>
                  <div className="abstract-glow" style={{ opacity: 0.1, transform: 'scale(1.5)' }} />
                  <Camera size={64} opacity={0.5} style={{ zIndex: 1 }} />
                  <p style={{ textAlign: 'center', fontSize: '14px', zIndex: 1 }}>Avvia la webcam o carica un video per iniziare</p>
                </div>
              )}
              <div className="webcam-overlay" style={{ padding: '12px' }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <div className="status-badge" style={{ borderColor: status === 'recording' ? '#ef4444' : undefined }}>
                    <div className={`status-dot ${active ? 'active' : ''}`} style={{ background: status === 'recording' ? '#ef4444' : undefined }} />
                    {active ? (status === 'recording' ? 'REC' : sourceType === 'webcam' ? 'LIVE' : 'FILE') : 'OFF'}
                  </div>
                  {sourceType === 'file' && active && (
                    <button className="ctrl-btn" style={{ padding: '6px 12px', fontSize: '12px' }} onClick={() => { setSourceType('webcam'); setVideoUrl(null); setActive(false); }}>
                      <RefreshCcw size={14} /> Reset Sorgente
                    </button>
                  )}
                </div>
                
                <div style={{ display: 'flex', gap: '8px' }}>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    style={{ display: 'none' }} 
                    accept="video/*" 
                    onChange={handleFileChange} 
                  />
                  <button className="ctrl-btn" style={{ padding: '6px 12px', fontSize: '12px' }} onClick={() => fileInputRef.current?.click()}>
                    <PlaySquare size={14} /> Carica Video
                  </button>
                  <button className={`ctrl-btn ${active ? 'danger' : 'primary'}`} style={{ padding: '6px 14px', fontSize: '13px' }} onClick={() => { if(!active) setSourceType('webcam'); setActive(v => !v); }}>
                    {active ? <><Square size={14} /> Stop</> : <><PlaySquare size={14} /> Avvia Webcam</>}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Current Sign & Phrase */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            <div className="glass-card current-sign" style={{ padding: '20px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <p className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center', marginBottom: '12px' }}>
                <BrainCircuit size={16} /> Output Visivo (Top-1)
              </p>
              <div style={{ textAlign: 'center' }}>
                <span className="sign-label" style={{ fontSize: '42px', minHeight: '60px' }}>
                  {status === 'analyzing' ? '...' : (top5.length > 0 ? getCleanTranslatedWord(top5[0].word) : '---')}
                </span>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' }}>
                  {status === 'recording' ? 'Registrazione in corso...' : 'In attesa di movimento'}
                </p>
              </div>
              {top5.length > 0 && (
                <div className="confidence-bar-wrap" style={{ marginTop: '12px' }}>
                  <div className="confidence-bar-bg">
                    <div className="confidence-bar" style={{ width: `${(top5[0].p * 100).toFixed(1)}%` }} />
                  </div>
                  <p className="confidence-label">Confidenza: {(top5[0].p * 100).toFixed(1)}%</p>
                </div>
              )}
            </div>

            <div className="glass-card" style={{ padding: '20px' }}>
              <p className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <Sparkles size={16} color="var(--accent-secondary)" /> Frase Corretta da BERT
              </p>
              <div className="phrase-output" style={{ minHeight: '80px', padding: '16px', marginBottom: '12px', position: 'relative' }}>
                {status === 'bert' && (
                    <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.2)', backdropFilter: 'blur(2px)', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '12px', zIndex: 10 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white', fontSize: '13px' }}>
                            <Loader2 className="animate-spin" size={16} /> BERT Rescoring...
                        </div>
                    </div>
                )}
                {phrase.length > 0
                  ? phrase.map((w, i) => <span key={i} className="phrase-word" style={{ fontSize: '15px', padding: '5px 12px' }}>{getCleanTranslatedWord(w)}</span>)
                  : <span className="phrase-placeholder">Esegui i segni. La "Matrix of Solutions" BERT correggerà la frase...</span>
                }
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button className="ctrl-btn danger" style={{ padding: '8px 16px', fontSize: '13px' }} onClick={clearPhrase}>
                  <Trash2 size={14} /> Reset Frase
                </button>
              </div>
            </div>

          </div>
        </div>

        {/* Bottom Row: Top 5 Candidates horizontally */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <p className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <ListOrdered size={16} /> Matrice dei Candidati (N-Best)
          </p>
          <div className="candidates-list" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
            {top5.length > 0 ? top5.map((c, i) => (
              <div key={i} className="candidate-item" style={{ padding: '10px 14px', borderLeft: i === 0 ? '4px solid var(--accent-primary)' : undefined }}>
                <span className="candidate-rank" style={{ width: '20px', fontSize: '12px' }}>#{i + 1}</span>
                <span className="candidate-word" style={{ fontSize: '14px', fontWeight: 600 }}>{getCleanTranslatedWord(c.word)}</span>
                <span className="candidate-score" style={{ fontSize: '11px' }}>{(c.p * 100).toFixed(0)}%</span>
              </div>
            )) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', padding: '20px 0', gridColumn: '1 / -1' }}>
                Muovi le mani per avviare il riconoscimento automatico
              </p>
            )}
          </div>
        </div>

      </div>

      <div className="score-strip">
        <div className="score-row">
          <div className="score-item">
            <Sparkles size={24} className="icon" color="var(--accent-secondary)" />
            <p className="score-value">{bertReady ? 'BERT ON' : 'BERT OFF'}</p>
            <p className="score-key">NLP Engine</p>
          </div>
          <div className="score-item">
            <Hash size={24} className="icon" />
            <p className="score-value">{phrase.length}</p>
            <p className="score-key">Segni In Frase</p>
          </div>
          <div className="score-item">
            <Frame size={24} className="icon" />
            <p className="score-value">{frameCount}</p>
            <p className="score-key">Frame Processati</p>
          </div>
        </div>
      </div>

    </div>
  )
}
