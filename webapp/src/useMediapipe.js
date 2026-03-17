// Hook per la gestione Mediapipe Holistic in tempo reale
// Usa @mediapipe/holistic che è IDENTICO al Python mp.solutions.holistic.Holistic
// Garantisce la stessa estrazione delle 258 feature usate nel training del modello
import { useEffect, useRef, useCallback } from 'react'
// La classe Holistic viene caricata globalmente via CDN in index.html

const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Braccia
  [11, 23], [12, 24], [23, 24], // Tronco
]

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Pollice
  [0, 5], [5, 6], [6, 7], [7, 8],       // Indice
  [5, 9], [9, 10], [10, 11], [11, 12],  // Medio
  [9, 13], [13, 14], [14, 15], [15, 16],// Anulare
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17] // Mignolo
]

/**
 * Estrae le 258 feature ESATTAMENTE come fa il training Python.
 * Pose (33*4) + Left Hand (21*3) + Right Hand (21*3) = 258
 * 
 * IMPORTANTE: left_hand e right_hand in Holistic sono dal punto di vista
 * del soggetto (non specchiato), esattamente come in mp.solutions.holistic.
 */
function extractFeatures(results) {
  const features = []

  // 1. POSE (33 landmark * 4 = 132 valori): x, y, z, visibility
  if (results.poseLandmarks) {
    for (const lm of results.poseLandmarks) {
      features.push(lm.x, lm.y, lm.z, lm.visibility ?? 0)
    }
  } else {
    features.push(...new Array(33 * 4).fill(0))
  }

  // 2. LEFT HAND (21 landmark * 3 = 63 valori): x, y, z
  if (results.leftHandLandmarks) {
    for (const lm of results.leftHandLandmarks) {
      features.push(lm.x, lm.y, lm.z)
    }
  } else {
    features.push(...new Array(21 * 3).fill(0))
  }

  // 3. RIGHT HAND (21 landmark * 3 = 63 valori): x, y, z
  if (results.rightHandLandmarks) {
    for (const lm of results.rightHandLandmarks) {
      features.push(lm.x, lm.y, lm.z)
    }
  } else {
    features.push(...new Array(21 * 3).fill(0))
  }

  return features  // length = 258
}

export function useMediapipe({ videoRef, canvasRef, onFrame, enabled }) {
  const holisticRef = useRef(null)
  const animFrameRef = useRef(null)
  const latestResultsRef = useRef(null)
  const offscreenRef = useRef(null) // Canvas per la normalizzazione della luminosità

  // Inizializzazione Holistic (uguale a Python)
  useEffect(() => {
    // Utilizziamo il costruttore caricato globalmente via script tag in index.html
    // @ts-ignore
    const HolisticConstructor = window.Holistic
    if (!HolisticConstructor) {
        console.error('Mediapipe Holistic non trovato! Verifica la connessione o lo script in index.html')
        return
    }
    const holistic = new HolisticConstructor({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`
    })

    holistic.setOptions({
      modelComplexity: 1,          // Uguale a Python: model_complexity=1
      smoothLandmarks: true,       // Uguale a Python: smooth_landmarks=True
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.5, // Uguale a Python: min_detection_confidence=0.5
      minTrackingConfidence: 0.5   // Uguale a Python: min_tracking_confidence=0.5
    })

    holistic.onResults((results) => {
      latestResultsRef.current = results
    })

    // Canvas offscreen per normalizzazione luminosità
    const offscreen = document.createElement('canvas')
    offscreen.width = 640
    offscreen.height = 480
    offscreenRef.current = offscreen

    holisticRef.current = holistic
    console.log('✅ Mediapipe Holistic initialized (matches Python training)')

    return () => {
      holistic.close()
      holisticRef.current = null
    }
  }, [])

  // Loop di elaborazione frame
  useEffect(() => {
    if (!enabled) {
      cancelAnimationFrame(animFrameRef.current)
      return
    }

    const processFrame = async () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      const holistic = holisticRef.current
      const offscreen = offscreenRef.current

      if (!video || !canvas || !holistic || !offscreen || video.readyState < 2 || video.paused || video.ended || video.seeking) {
        animFrameRef.current = requestAnimationFrame(processFrame)
        return
      }

      // ─── Preprocessing: normalizzazione luminosità ─────────────────────────
      // Disegniamo il video su un canvas temporaneo con filtri CSS per correggere
      // il controluce e le condizioni di scarsa illuminazione tipiche di una webcam
      // puntata verso una finestra.
      const octx = offscreen.getContext('2d')
      octx.filter = 'brightness(1.4) contrast(1.2)'
      octx.drawImage(video, 0, 0, offscreen.width, offscreen.height)
      octx.filter = 'none'

      // Invia il frame preprocessato (luminosità aumentata) al modello Holistic
      await holistic.send({ image: offscreen })

      // Disegna i risultati
      const results = latestResultsRef.current
      if (results) {
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        const drawLandmarks = (landmarks, connections, color, dotColor, size = 3) => {
          if (!landmarks || landmarks.length === 0) return

          // Connessioni
          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.beginPath()
          for (const [i, j] of connections) {
            const p1 = landmarks[i], p2 = landmarks[j]
            if (p1 && p2) {
              ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height)
              ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height)
            }
          }
          ctx.stroke()

          // Punti
          ctx.fillStyle = dotColor
          for (const p of landmarks) {
            if ((p.visibility ?? 1) < 0.2) continue
            ctx.beginPath()
            ctx.arc(p.x * canvas.width, p.y * canvas.height, size, 0, 2 * Math.PI)
            ctx.fill()
          }
        }

        // Corpo (blu)
        if (results.poseLandmarks) {
          drawLandmarks(results.poseLandmarks, POSE_CONNECTIONS, 'rgba(59,130,246,0.4)', '#3b82f6', 2)
        }

        // Mano Sinistra (verde)
        if (results.leftHandLandmarks) {
          drawLandmarks(results.leftHandLandmarks, HAND_CONNECTIONS, 'rgba(16,185,129,0.6)', '#10b981', 5)
        }

        // Mano Destra (viola)
        if (results.rightHandLandmarks) {
          drawLandmarks(results.rightHandLandmarks, HAND_CONNECTIONS, 'rgba(139,92,246,0.6)', '#8b5cf6', 5)
        }

        // Estrai features e invia al modello LSTM
        const features = extractFeatures(results)
        onFrame?.(features)
      }

      animFrameRef.current = requestAnimationFrame(processFrame)
    }

    animFrameRef.current = requestAnimationFrame(processFrame)
    return () => cancelAnimationFrame(animFrameRef.current)
  }, [enabled, videoRef, canvasRef, onFrame])
}
