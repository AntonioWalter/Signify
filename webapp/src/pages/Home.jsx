import React, { useContext } from 'react'
import { ArrowRight, RefreshCcw, GraduationCap, Cpu, Zap, Globe, Github, Linkedin } from 'lucide-react'
import { AppContext } from '../App.jsx'

const TEXTS = {
  it: {
    heroTitle1: 'Abbattiamo le barriere della ',
    heroTitleGradient: 'Comunicazione',
    heroSubtitle: "Signify utilizza l'intelligenza artificiale avanzata per tradurre la Lingua dei Segni in tempo reale, rendendo il mondo più inclusivo e interconnesso.",
    ctaTranslator: 'Prova il Traduttore',
    ctaTrainer: 'Impara i Segni',
    rtTitle: 'Real-Time',
    rtDesc: 'Traduzione simultanea in tempo reale dei segni acquisiti.',
    nnTitle: 'Reti Neurali',
    nnDesc: 'Modello basato su reti neurali per catturare le sfumature del movimento.',
    vocabTitle: '2300+ Segni',
    vocabDesc: 'Supporto ad un vasto vocabolario con traduzione istantanea in Italiano.',
    toolsTitle: 'Esplora gli Strumenti',
    toolTranslatorTitle: 'Traduttore Live',
    toolTranslatorDesc: 'Inquadra la webcam e comunica liberamente. Il modello tradurrà i tuoi gesti in testo istantaneo.',
    toolTrainerTitle: 'Istruttore Interattivo',
    toolTrainerDesc: 'Esercitati con il manichino 3D. Seguiremo i tuoi movimenti e ti daremo feedback in tempo reale.',
    datasetTitle: `Il Dataset e l'Intelligenza dietro Signify`,
    datasetBody: `Il cuore pulsante del progetto è un modello di Deep Learning basato su Reti Neurali. Un ringraziamento speciale va agli autori del dataset originale ASL Citizen (Microsoft / Aweigh et al., Kaggle), che ha fornito migliaia di video annotati per oltre 2000 segni. Ho utilizzato MediaPipe per estrarre i landmark e trasformarli in sequenze 3D. Per rendere Signify fruibile in Italia, ho tradotto l'intero vocabolario in Italiano.`,
    footerText: `© 2026 Antonio Walter De Fusco — Signify Project`,
  },
  en: {
    heroTitle1: 'Breaking the barriers of ',
    heroTitleGradient: 'Communication',
    heroSubtitle: 'Signify uses advanced artificial intelligence to translate Sign Language in real time, making the world more inclusive and interconnected.',
    ctaTranslator: 'Try the Translator',
    ctaTrainer: 'Learn Signs',
    rtTitle: 'Real-Time',
    rtDesc: 'Simultaneous real-time translation of captured signs.',
    nnTitle: 'Neural Networks',
    nnDesc: 'A neural network-based model to capture the nuances of movement.',
    vocabTitle: '2300+ Signs',
    vocabDesc: 'Support for a large vocabulary with instant Italian translation.',
    toolsTitle: 'Explore the Tools',
    toolTranslatorTitle: 'Live Translator',
    toolTranslatorDesc: 'Point your webcam and communicate freely. The model will translate your gestures into instant text.',
    toolTrainerTitle: 'Interactive Instructor',
    toolTrainerDesc: 'Practice with the 3D mannequin. We will track your movements and give you real-time feedback.',
    datasetTitle: 'The Dataset and Intelligence behind Signify',
    datasetBody: 'At the core of the project is a Deep Learning model based on Neural Networks. Special thanks to the authors of the original ASL Citizen dataset (Microsoft / Aweigh et al., Kaggle), which provided thousands of annotated videos for over 2000 signs. MediaPipe was used to extract landmarks and transform them into 3D sequences. To make Signify usable in Italy, the entire vocabulary was translated into Italian.',
    footerText: '© 2026 Antonio Walter De Fusco — Signify Project',
  }
}

export default function Home({ onNavigate }) {
  const { lang } = useContext(AppContext)
  const t = TEXTS[lang] || TEXTS.it

  return (
    <div className="home-layout">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1 className="hero-title">
            {t.heroTitle1}<span className="text-gradient">{t.heroTitleGradient}</span>
          </h1>
          <p className="hero-subtitle">{t.heroSubtitle}</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <button className="ctrl-btn primary hero-btn" onClick={() => onNavigate('translator')}>
              {t.ctaTranslator} <ArrowRight size={18} />
            </button>
            <button className="ctrl-btn hero-btn-outline" onClick={() => onNavigate('trainer')}>
              {t.ctaTrainer} <GraduationCap size={18} />
            </button>
          </div>
        </div>
      </section>

      {/* Stats/Highlight Section */}
      <div className="highlight-grid">
        <div className="highlight-item">
          <Zap className="highlight-icon" color="var(--accent-secondary)" />
          <div>
            <h3>{t.rtTitle}</h3>
            <p>{t.rtDesc}</p>
          </div>
        </div>
        <div className="highlight-item">
          <Cpu className="highlight-icon" color="var(--accent-primary)" />
          <div>
            <h3>{t.nnTitle}</h3>
            <p>{t.nnDesc}</p>
          </div>
        </div>
        <div className="highlight-item">
          <Globe className="highlight-icon" color="var(--success)" />
          <div>
            <h3>{t.vocabTitle}</h3>
            <p>{t.vocabDesc}</p>
          </div>
        </div>
      </div>

      {/* Tool Cards */}
      <section className="tools-preview">
        <h2 className="section-title">{t.toolsTitle}</h2>
        <div className="tool-cards">
          <div className="tool-card group" onClick={() => onNavigate('translator')}>
            <div className="tool-card-icon">
              <RefreshCcw size={32} />
            </div>
            <h3>{t.toolTranslatorTitle}</h3>
            <p>{t.toolTranslatorDesc}</p>
            <div className="tool-card-arrow">
              <ArrowRight size={20} />
            </div>
          </div>

          <div className="tool-card group" onClick={() => onNavigate('trainer')}>
            <div className="tool-card-icon">
              <GraduationCap size={32} />
            </div>
            <h3>{t.toolTrainerTitle}</h3>
            <p>{t.toolTrainerDesc}</p>
            <div className="tool-card-arrow">
              <ArrowRight size={20} />
            </div>
          </div>
        </div>
      </section>

      {/* Tech Section */}
      <section className="tech-section">
        <div className="glass-card tech-card">
          <div className="tech-info">
            <h2>{t.datasetTitle}</h2>
            <p>{t.datasetBody}</p>
            <div className="tech-tags">
              <span className="tech-tag">Reti Neurali</span>
              <span className="tech-tag">ASL Citizen Dataset</span>
              <span className="tech-tag">MediaPipe</span>
              <span className="tech-tag">ONNX</span>
            </div>
          </div>
          <div className="tech-visual">
            <img
              src="sign_language_user.png"
              alt="Person using sign language at a laptop"
              style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '12px' }}
            />
          </div>
        </div>
      </section>

      <footer className="home-footer">
        <p>{t.footerText}</p>
        <div className="footer-links" style={{ display: 'flex', flexDirection: 'row', gap: '20px', alignItems: 'center' }}>
          <a href="https://github.com/AntonioWalter/Signify" target="_blank" rel="noreferrer">
            <Github size={18}/> GitHub
          </a>
          <a href="https://www.linkedin.com/in/antonio-walter-de-fusco/" target="_blank" rel="noreferrer">
            <Linkedin size={18}/> LinkedIn
          </a>
        </div>
      </footer>
    </div>
  )
}
