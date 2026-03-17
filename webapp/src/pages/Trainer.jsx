import React, { useState, useRef, useCallback, useEffect, useContext } from 'react'
import { predict, getVocabWords } from '../engine.js'
import { useMediapipe } from '../useMediapipe.js'
import SkeletonRenderer from '../components/SkeletonRenderer'
import CURATED_SIGNS from '../curated_signs.json'
import { AppContext } from '../App.jsx'
import { Camera, PlaySquare, Square, Target, Activity, SkipForward, Youtube, CheckCircle, XCircle, Trophy, Target as TargetIcon, Percent } from 'lucide-react'

const WINDOW_SIZE = 30
const TOP_N_MATCH = 5                  // Il segno è corretto se è tra i primi N risultati
const MOTION_START_THRESHOLD = 0.012   // Soglia per considerare che le mani si stanno muovendo
const MOTION_STOP_THRESHOLD  = 0.005   // Soglia per considerare che il gesto è finito
const FRAMES_TO_STOP = 12             // Quanti frame quieti prima di classificare (~0.4s)
const MIN_GESTURE_FRAMES = 12         // Minimo di frame necessari per classificare
const COOLDOWN_MS = 2500              // Pausa dopo classificazione prima di ascoltare di nuovo

function getCuratedWords() {
  const vocab = new Set(getVocabWords().map(w => w.toUpperCase()))
  const intersection = CURATED_SIGNS.filter(s => vocab.has(s))
  return intersection.length > 0 ? intersection : CURATED_SIGNS
}

function getRandomWord(exclude = null) {
  const words = getCuratedWords().filter(w => w !== exclude)
  if (words.length === 0) return 'WASHHANDS'
  return words[Math.floor(Math.random() * words.length)].toUpperCase()
}

function handMotion(prevFrame, currFrame) {
  if (!prevFrame || !currFrame) return 0
  let sum = 0
  for (let i = 132; i < 258; i++) {
    const d = (currFrame[i] || 0) - (prevFrame[i] || 0)
    sum += d * d
  }
  return Math.sqrt(sum / 126)
}

export default function Trainer() {
  const { translate } = useContext(AppContext)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  // Motion segmentation state 
  const prevFrameRef    = useRef(null)
  const captureRef      = useRef([])       
  const isRecordingRef  = useRef(false)    
  const quietFramesRef  = useRef(0)        
  const cooldownUntil   = useRef(0)        

  // UI state
  const [active, setActive]         = useState(false)
  const [targetWord, setTargetWord] = useState(() => getRandomWord())
  const [prediction, setPrediction] = useState(null)
  const [score, setScore]           = useState({ correct: 0, total: 0 })
  const [isSignFound, setIsSignFound] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [captureStatus, setCaptureStatus] = useState('idle') 
  const [motionLevel, setMotionLevel] = useState(0)

  useEffect(() => {
    if (active) {
      navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then(stream => { if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play() } })
    } else {
      if (videoRef.current?.srcObject) videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      isRecordingRef.current = false
      captureRef.current = []
      setPrediction(null)
      setIsSignFound(false)
      setCaptureStatus('idle')
    }
  }, [active])

  const classifyGesture = useCallback(async (frames) => {
    if (frames.length < MIN_GESTURE_FRAMES) return
    setCaptureStatus('analyzing')
    setIsAnalyzing(true)

    const buf = frames.length <= WINDOW_SIZE
      ? frames
      : Array.from({ length: WINDOW_SIZE }, (_, i) =>
          frames[Math.min(Math.floor(i * frames.length / WINDOW_SIZE), frames.length - 1)]
        )

    const preds = await predict(buf)
    setIsAnalyzing(false)

    if (preds) {
      setPrediction(preds)
      const isMatch = preds.slice(0, TOP_N_MATCH).some(p => p.word.toUpperCase() === targetWord)
      if (isMatch) {
        setIsSignFound(true)
        setScore(s => ({ correct: s.correct + 1, total: s.total + 1 }))
        setCaptureStatus('idle')
      } else {
        cooldownUntil.current = performance.now() + COOLDOWN_MS
        setCaptureStatus('cooldown')
        setTimeout(() => setCaptureStatus('idle'), COOLDOWN_MS)
      }
    }
  }, [targetWord])

  const onFrame = useCallback((features) => {
    if (isSignFound) return

    const motion = handMotion(prevFrameRef.current, features)
    prevFrameRef.current = features
    setMotionLevel(motion)

    const now = performance.now()
    if (now < cooldownUntil.current) return

    if (!isRecordingRef.current) {
      if (motion > MOTION_START_THRESHOLD) {
        isRecordingRef.current = true
        quietFramesRef.current = 0
        captureRef.current = [features]
        setCaptureStatus('recording')
      }
    } else {
      captureRef.current.push(features)
      if (motion < MOTION_STOP_THRESHOLD) {
        quietFramesRef.current += 1
        if (quietFramesRef.current >= FRAMES_TO_STOP) {
          isRecordingRef.current = false
          quietFramesRef.current = 0
          const captured = [...captureRef.current]
          captureRef.current = []
          classifyGesture(captured)
        }
      } else {
        quietFramesRef.current = 0
      }
    }
  }, [isSignFound, classifyGesture])

  useMediapipe({ videoRef, canvasRef, onFrame, enabled: active })

  const skipTarget = () => {
    setScore(s => ({ ...s, total: s.total + 1 }))
    nextTarget()
  }

  const nextTarget = () => {
    setTargetWord(getRandomWord(targetWord))
    setPrediction(null)
    setIsSignFound(false)
    setIsAnalyzing(false)
    isRecordingRef.current = false
    captureRef.current = []
    cooldownUntil.current = 0
    setCaptureStatus('idle')
  }

  const accuracy = score.total > 0 ? Math.round((score.correct / score.total) * 100) : 0
  const isCorrect = prediction?.slice(0, TOP_N_MATCH).some(p => p.word.toUpperCase() === targetWord)
  
  const getCleanTranslatedWord = (word) => {
    if (!word) return '---'
    return translate(word.replace(/[0-9]/g, '')).toUpperCase()
  }

  const cleanTargetWord = getCleanTranslatedWord(targetWord)

  const statusInfo = {
    idle:      { icon: <Activity className="status-icon" color="var(--text-muted)" size={48} />, text: 'Inizia il gesto',   color: 'var(--text-muted)' },
    recording: { icon: <Activity className="status-icon" color="#ef4444" size={48} />,  text: 'Registrando...',    color: '#ef4444' },
    analyzing: { icon: <Activity className="status-icon" color="#f59e0b" size={48} />, text: 'Analisi in corso…', color: '#f59e0b' },
    cooldown:  { icon: <Activity className="status-icon" color="var(--text-secondary)" size={48} />, text: 'Riprova...',         color: 'var(--text-secondary)' },
  }[captureStatus] || { icon: <Activity size={48} />, text: '', color: 'white' }

  return (
    <div className="trainer-layout">
      {/* Modal Success Overlay */}
      {isSignFound && (
        <div className="success-modal-overlay">
          <div className="success-modal">
            <div className="icon-wrapper">
              <CheckCircle size={80} />
            </div>
            <h3>{translate('sign_guessed')}</h3>
            <p>{translate('great_job')}</p>
            <button className="next-btn" onClick={nextTarget} style={{ marginTop: '30px' }}>{translate('next_sign')} ⏭</button>
          </div>
        </div>
      )}

      <div className="trainer-header" style={{ marginBottom: '8px' }}>
        <h2 style={{ fontSize: '28px', marginBottom: '8px' }}>{translate('interactive_instructor')}</h2>
        <p style={{ fontSize: '14px' }}>{translate('trainer_subtitle')}</p>
      </div>

      {/* Row 1: Videos Side-by-Side */}
      <div className="responsive-grid" style={{ gridTemplateColumns: '1fr 1fr', marginBottom: '20px' }}>
        
        {/* Left: target + mannequin */}
        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <p className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Target size={16} /> {translate('target_sign')}
            </p>
            <div className="target-label" style={{ margin: 0, fontSize: '28px', lineHeight: 1 }}>{cleanTargetWord}</div>
          </div>

          <div className="demo-video-container" style={{ flex: 1, minHeight: 0 }}>
            <SkeletonRenderer signName={targetWord} />
          </div>

          <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <a href={`https://www.youtube.com/results?search_query=ASL+sign+${targetWord.replace(/[0-9]/g, '')}`} target="_blank" rel="noreferrer" className="youtube-fallback-link" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Youtube size={16} /> YouTube
            </a>
            <button className="ctrl-btn" style={{ fontSize: '12px', padding: '6px 14px' }} onClick={skipTarget}>
              <SkipForward size={14} /> {translate('skip')}
            </button>
          </div>
        </div>

        {/* Right: webcam */}
        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
          <p className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <Camera size={16} /> {translate('your_camera')}
          </p>
          
          <div className="webcam-container" style={{ flex: 1, minHeight: 0, padding: 0, overflow: 'hidden', position: 'relative' }}>
            <video ref={videoRef} playsInline muted style={{ display: active ? 'block' : 'none' }} />
            <canvas ref={canvasRef} width={640} height={480} style={{ display: active ? 'block' : 'none' }} />
            {!active && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '12px', color: 'var(--text-muted)' }}>
                <Camera size={40} opacity={0.5} />
                <p style={{ fontSize: '13px' }}>{translate('start_prompt')}</p>
              </div>
            )}
            <div className="webcam-overlay" style={{ padding: '12px' }}>
              <div className="status-badge" style={{ borderColor: captureStatus === 'recording' ? 'rgba(239,68,68,0.6)' : undefined }}>
                <div className={`status-dot ${active ? 'active' : ''}`} style={{ background: captureStatus === 'recording' ? '#ef4444' : undefined }} />
                {active ? (captureStatus === 'recording' ? translate('rec') : translate('live')) : translate('off')}
              </div>
              <button className={`ctrl-btn ${active ? 'danger' : 'primary'}`} style={{ padding: '6px 14px', fontSize: '13px' }} onClick={() => setActive(v => !v)}>
                {active ? <><Square size={14} /> {translate('stop')}</> : <><PlaySquare size={14} /> {translate('start')}</>}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Status, Feedback, Top 5 */}
      <div className="responsive-grid trainer-feedback-row" style={{ gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr) minmax(0,1.5fr)' }}>
        
        {/* Status */}
        <div className="glass-card" style={{ padding: '20px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <p className="card-title" style={{ marginBottom: '8px' }}>🎬 {translate('detection')}</p>
          <div style={{ margin: '8px 0', transition: 'all 0.3s', display: 'flex', justifyContent: 'center' }}>
            {statusInfo.icon}
          </div>
          <p style={{ fontSize: '15px', fontWeight: 700, color: statusInfo.color, marginBottom: '12px', lineHeight: 1.2 }}>
            {statusInfo.text}
          </p>
          {active && !isSignFound && (
            <div style={{ width: '100%' }}>
              <div className="meter-bg" style={{ height: '6px' }}>
                <div className="meter-fill" style={{
                  width: `${Math.min(motionLevel / MOTION_START_THRESHOLD * 100, 100)}%`,
                  background: motionLevel > MOTION_START_THRESHOLD ? '#ef4444' : motionLevel > MOTION_STOP_THRESHOLD ? '#f59e0b' : '#10b981'
                }} />
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
                {motionLevel > MOTION_START_THRESHOLD ? `🟥 ${translate('recording')}` : motionLevel > MOTION_STOP_THRESHOLD ? `🟡 ...` : `🟢 ${translate('waiting_motion')}`}
              </p>
            </div>
          )}
        </div>

        {/* Feedback */}
        <div className="glass-card" style={{ 
          padding: '20px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
          borderColor: (prediction && !isSignFound) ? (isCorrect ? 'rgba(16,185,129,0.5)' : 'rgba(239,68,68,0.4)') : undefined
        }}>
          <p className="card-title" style={{ marginBottom: '8px' }}>💡 {translate('output')}</p>
          {prediction && !isSignFound ? (
            <>
              <div style={{ color: isCorrect ? 'var(--success)' : 'var(--error)', marginBottom: '8px' }}>
                {isCorrect ? <CheckCircle size={36} /> : <XCircle size={36} />}
              </div>
              <p className={`feedback-text ${isCorrect ? 'success' : 'fail'}`} style={{ fontSize: '17px', margin: 0, lineHeight: 1.2 }}>
                {isCorrect ? translate('correct') : `${translate('detected_label')} ${getCleanTranslatedWord(prediction[0].word)}`}
              </p>
              {!isCorrect && (
                <p style={{ fontSize: '12px', marginTop: '6px', color: 'var(--text-secondary)', lineHeight: 1.2 }}>
                  {translate('trainer_retry_prompt')}
                </p>
              )}
            </>
          ) : (
            <>
              <div style={{ color: 'var(--text-muted)' }}><Activity size={36} /></div>
              <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginTop: '12px' }}>{translate('start_gesture')}</p>
            </>
          )}
        </div>

        {/* Top 5 candidates */}
        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
          <p className="card-title" style={{ marginBottom: '12px' }}>📊 {translate('top_5_candidates')}</p>
          <div className="candidates-list" style={{ gap: '6px', overflowY: 'auto' }}>
            {prediction && !isSignFound ? prediction.slice(0, 5).map((c, i) => (
              <div key={i} className="candidate-item" style={{ padding: '6px 10px', ...(c.word.toUpperCase() === targetWord ? { borderColor: 'rgba(16,185,129,0.5)', background: 'rgba(16,185,129,0.1)' } : {})}}>
                <span className="candidate-rank" style={{ width: '20px', fontSize: '12px' }}>#{i + 1}</span>
                <span className="candidate-word" style={{ fontSize: '13px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{getCleanTranslatedWord(c.word)}</span>
                <span className="candidate-score" style={{ fontSize: '11px', padding: '2px 6px' }}>{(c.p * 100).toFixed(0)}%</span>
              </div>
            )) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '12px', textAlign: 'center', padding: '30px 0', margin: 'auto' }}>{translate('no_recent_m')}</p>
            )}
          </div>
        </div>

      </div>

      {/* Bottom Full-Width Stats */}
      <div className="score-strip">
        <div className="score-row">
          <div className="score-item">
            <Trophy size={24} className="icon" style={{ color: 'var(--accent-success)' }} />
            <p className="score-value">{score.correct}</p>
            <p className="score-key">{translate('signs_correct')}</p>
          </div>
          <div className="score-item">
            <TargetIcon size={24} className="icon" />
            <p className="score-value">{score.total}</p>
            <p className="score-key">{translate('signs_attempted')}</p>
          </div>
          <div className="score-item">
            <Percent size={24} className="icon" style={{ color: accuracy >= 70 ? 'var(--accent-success)' : '#f59e0b' }} />
            <p className="score-value">{accuracy}%</p>
            <p className="score-key">{translate('accuracy')}</p>
          </div>
        </div>
      </div>

    </div>
  )
}


