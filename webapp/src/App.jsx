import React, { useState, useEffect, createContext } from 'react'
import Home from './pages/Home.jsx'
import Translator from './pages/Translator.jsx'
import Trainer from './pages/Trainer.jsx'
import Demo from './pages/Demo.jsx'
import { loadModel } from './engine.js'
import './index.css'
import { Moon, Sun, Languages, RefreshCcw, GraduationCap, Home as HomeIcon, Clapperboard } from 'lucide-react'

export const AppContext = createContext(null)

export default function App() {
  const [page, setPage] = useState('home')
  const [modelStatus, setModelStatus] = useState('loading')
  const [loadingMsg, setLoadingMsg] = useState('Inizializzazione...')
  const [errorMsg, setErrorMsg] = useState('')

  const [theme, setTheme] = useState('dark')
  const [vocabIt, setVocabIt] = useState({})
  
  const UI_STRINGS = {
    it: {
      home: "Home",
      translator: "Traduttore",
      trainer: "Istruttore",
      demo: "Live Demo",
      loading: "Caricamento...",
      online: "Online",
      initializing: "Inizializzazione...",
      error_loading: "Errore nel caricamento del modello",
      unknown_error: "Errore sconosciuto",
      live_translation: "Traduzione Live",
      bert_init: "Inizializzazione BERT Modello Linguistico...",
      start_prompt: "Avvia la webcam o carica un video per iniziare",
      reset_source: "Reset Sorgente",
      load_video: "Carica Video",
      stop: "Stop",
      start_webcam: "Avvia Webcam",
      visual_output: "Output Visivo (Top-1)",
      recording: "Registrazione in corso...",
      waiting_motion: "In attesa di movimento",
      confidence: "Confidenza",
      bert_phrase: "Frase Corretta da BERT",
      reset_phrase: "Reset Frase",
      candidate_matrix: "Matrice dei Candidati (N-Best)",
      move_hands_prompt: "Muovi le mani per avviare il riconoscimento automatico",
      signs_in_phrase: "Segni In Frase",
      processed_frames: "Frame Processati",
      bert_rescoring: "BERT Rescoring...",
      interactive_instructor: "Istruttore Interattivo",
      trainer_subtitle: "Guarda il manichino ed esegui il gesto — la registrazione parte automaticamente quando le mani si muovono.",
      target_sign: "Segno da Eseguire",
      skip: "Salta",
      your_camera: "La Tua Telecamera",
      start: "Avvia",
      detection: "Rilevamento",
      start_gesture: "Inizia il gesto",
      analyzing: "Analisi in corso...",
      retry: "Riprova...",
      rec: "REC",
      live: "Live",
      off: "Off",
      sign_guessed: "Segno Indovinato!",
      great_job: "Ottimo lavoro, hai completato correttamente il gesto.",
      next_sign: "PROSSIMO SEGNO",
      output: "Output",
      correct: "Corretto!",
      detected_label: "Rilevato:",
      trainer_retry_prompt: "Riprova — gesto più lento e preciso",
      top_5_candidates: "Top 5 Candidati",
      no_recent_m: "Nessuna misurazione recente",
      signs_correct: "Corretti",
      signs_attempted: "Tentati",
      accuracy: "Accuratezza",
      demo_subtitle: "Scegli una frase: il manichino la eseguirà e il modello AI la tradurrà in tempo reale",
      select_phrase: "1 · Seleziona una frase",
      animation_sign: "2 · Animazione Segno",
      sign_rank: "Segno",
      ai_predictions_sign: "3 · Predizioni AI per Segno",
      bert_demo_phrase: "4 · Frase Corretta da BERT",
      signs_in_this_phrase: "Segni in questa frase",
      select_phrase_prompt: "Seleziona una frase qui sopra per avviare la demo"
    },
    en: {
      home: "Home",
      translator: "Translator",
      trainer: "Trainer",
      demo: "Live Demo",
      loading: "Loading...",
      online: "Online",
      initializing: "Initializing...",
      error_loading: "Error loading model",
      unknown_error: "Unknown error",
      live_translation: "Live Translation",
      bert_init: "Initializing BERT Language Model...",
      start_prompt: "Start webcam or upload a video to begin",
      reset_source: "Reset Source",
      load_video: "Upload Video",
      stop: "Stop",
      start_webcam: "Start Webcam",
      visual_output: "Visual Output (Top-1)",
      recording: "Recording in progress...",
      waiting_motion: "Waiting for motion",
      confidence: "Confidence",
      bert_phrase: "BERT Corrected Phrase",
      reset_phrase: "Reset Phrase",
      candidate_matrix: "Candidate Matrix (N-Best)",
      move_hands_prompt: "Move your hands to start auto-recognition",
      signs_in_phrase: "Signs In Phrase",
      processed_frames: "Processed Frames",
      bert_rescoring: "BERT Rescoring...",
      interactive_instructor: "Interactive Instructor",
      trainer_subtitle: "Watch the mannequin and perform the gesture — recording starts automatically when hands move.",
      target_sign: "Target Sign",
      skip: "Skip",
      your_camera: "Your Camera",
      start: "Start",
      detection: "Detection",
      start_gesture: "Start the gesture",
      analyzing: "Analyzing...",
      retry: "Retry...",
      rec: "REC",
      live: "Live",
      off: "Off",
      sign_guessed: "Sign Guessed!",
      great_job: "Great job, you completed the gesture correctly.",
      next_sign: "NEXT SIGN",
      output: "Output",
      correct: "Correct!",
      detected_label: "Detected:",
      trainer_retry_prompt: "Retry — slower and more precise gesture",
      top_5_candidates: "Top 5 Candidates",
      no_recent_m: "No recent measurements",
      signs_correct: "Correct",
      signs_attempted: "Attempted",
      accuracy: "Accuracy",
      demo_subtitle: "Choose a phrase: the mannequin will perform it and the AI model will translate it in real time",
      select_phrase: "1 · Select a phrase",
      animation_sign: "2 · Sign Animation",
      sign_rank: "Sign",
      ai_predictions_sign: "3 · AI Predictions per Sign",
      bert_demo_phrase: "4 · BERT Corrected Phrase",
      signs_in_this_phrase: "Signs in this phrase",
      select_phrase_prompt: "Select a phrase above to start the demo"
    }
  }

  useEffect(() => {
    // Apply theme
    document.documentElement.className = theme
  }, [theme])

  useEffect(() => {
    // Load model & italian vocab
    Promise.all([
      loadModel((msg) => setLoadingMsg(msg)),
      fetch(`${import.meta.env.BASE_URL}models/vocab.json`).then(r => r.json()).catch(() => ({})),
      fetch(`${import.meta.env.BASE_URL}models/vocab_it.json`).then(r => r.json()).catch(() => ({}))
    ])
      .then(([_, enVocab, itVocab]) => {
        const mapping = {}
        for (const [key, enWord] of Object.entries(enVocab)) {
          if (itVocab[key]) {
            mapping[enWord.toLowerCase()] = itVocab[key]
          }
        }
        setVocabIt(mapping)
        setModelStatus('ready')
      })
      .catch(err => { 
        console.error(err); 
        setModelStatus('error');
        setErrorMsg(err.message || 'Errore sconosciuto');
      })
  }, [])

  const translate = (key) => {
    if (!key) return ''
    // 1. Check UI strings first
    const ui = UI_STRINGS[lang][key.toLowerCase()]
    if (ui) return ui
    
    // 2. Fallback to vocab if not en
    if (lang === 'en') return key
    return vocabIt[key.toLowerCase()] || key
  }

  return (
    <AppContext.Provider value={{ theme, setTheme, lang, setLang, translate }}>
      <div className="app">
        <header className="header">
          <div className="logo" onClick={() => setPage('home')} style={{ cursor: 'pointer' }}>
            <img src="logosignify.png" alt="Signify Logo" className="logo-icon" />
          </div>
          
          <nav className="nav">
            <button className={`nav-btn ${page === 'home' ? 'active' : ''}`} onClick={() => setPage('home')} style={{display:'flex', alignItems:'center', gap:'6px'}}>
              <HomeIcon size={16} /> {translate('home')}
            </button>
            <button className={`nav-btn ${page === 'translator' ? 'active' : ''}`} onClick={() => setPage('translator')} style={{display:'flex', alignItems:'center', gap:'6px'}}>
              <RefreshCcw size={16} /> {translate('translator')}
            </button>
            <button className={`nav-btn ${page === 'trainer' ? 'active' : ''}`} onClick={() => setPage('trainer')} style={{display:'flex', alignItems:'center', gap:'6px'}}>
              <GraduationCap size={16} /> {translate('trainer')}
            </button>
            <button className={`nav-btn ${page === 'demo' ? 'active' : ''}`} onClick={() => setPage('demo')} style={{display:'flex', alignItems:'center', gap:'6px'}}>
              <Clapperboard size={16} /> {translate('demo')}
            </button>
          </nav>
          
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button className="nav-btn" onClick={() => setLang(l => l === 'it' ? 'en' : 'it')} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px' }}>
              <Languages size={18} /> {lang.toUpperCase()}
            </button>
            <button className="nav-btn" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} style={{ display: 'flex', alignItems: 'center', padding: '6px 10px' }}>
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            
            <div className="status-badge" style={{ fontSize: '12px', marginLeft: '10px' }}>
              <div className={`status-dot ${modelStatus === 'ready' ? 'active' : ''}`} />
              {modelStatus === 'ready' ? translate('online') : translate('loading')}
            </div>
          </div>
        </header>

        <main className="main">
          {modelStatus === 'loading' && (
            <div className="loading-screen">
              <div className="spinner" />
              <p className="loading-text">{loadingMsg}</p>
            </div>
          )}
          {modelStatus === 'error' && (
            <div className="loading-screen">
              <div style={{ fontSize: '48px' }}>⚠️</div>
              <p style={{ color: 'var(--accent-error)', fontSize: '16px' }}>{translate('error_loading')}</p>
              <p style={{ color: 'var(--text-white)', fontSize: '14px', background: 'rgba(255,0,0,0.1)', padding: '10px', borderRadius: '8px', margin: '10px 0' }}>
                <code>{errorMsg}</code>
              </p>
            </div>
          )}
          {modelStatus === 'ready' && page === 'home' && <Home onNavigate={setPage} />}
          {modelStatus === 'ready' && page === 'translator' && <Translator />}
          {modelStatus === 'ready' && page === 'trainer' && <Trainer />}
          {modelStatus === 'ready' && page === 'demo' && <Demo />}
        </main>
      </div>
    </AppContext.Provider>
  )
}
