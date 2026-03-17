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
  const [lang, setLang] = useState('it') // Default to italian
  const [vocabIt, setVocabIt] = useState({})

  useEffect(() => {
    // Apply theme
    document.documentElement.className = theme
  }, [theme])

  useEffect(() => {
    // Load model & italian vocab
    Promise.all([
      loadModel((msg) => setLoadingMsg(msg)),
      fetch('./models/vocab.json').then(r => r.json()).catch(() => ({})),
      fetch('./models/vocab_it.json').then(r => r.json()).catch(() => ({}))
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

  const translate = (word) => {
    if (!word) return ''
    if (lang === 'en') return word
    return vocabIt[word.toLowerCase()] || word
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
              <HomeIcon size={16} /> Home
            </button>
            <button className={`nav-btn ${page === 'translator' ? 'active' : ''}`} onClick={() => setPage('translator')} style={{display:'flex', alignItems:'center', gap:'6px'}}>
              <RefreshCcw size={16} /> Traduttore
            </button>
            <button className={`nav-btn ${page === 'trainer' ? 'active' : ''}`} onClick={() => setPage('trainer')} style={{display:'flex', alignItems:'center', gap:'6px'}}>
              <GraduationCap size={16} /> Istruttore
            </button>
            <button className={`nav-btn ${page === 'demo' ? 'active' : ''}`} onClick={() => setPage('demo')} style={{display:'flex', alignItems:'center', gap:'6px'}}>
              <Clapperboard size={16} /> Live Demo
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
              {modelStatus === 'ready' ? 'Online' : 'Loading'}
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
              <p style={{ color: 'var(--accent-error)', fontSize: '16px' }}>Errore nel caricamento del modello</p>
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
