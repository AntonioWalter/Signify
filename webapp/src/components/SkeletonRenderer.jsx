import React, { useEffect, useRef, useState } from 'react'

// ─── Connessioni anatomiche del corpo ────────────────────────────────────────
const BODY_LINKS = [
  [11, 12], [11, 13], [13, 15],    // Spalla sx, braccio sx
  [12, 14], [14, 16],               // Braccio dx
  [11, 23], [12, 24], [23, 24]      // Torso
]

const HAND_CHAINS = [
  [0, 1, 2, 3, 4],    // Pollice
  [0, 5, 6, 7, 8],    // Indice
  [5, 9, 10, 11, 12], // Medio
  [9, 13, 14, 15, 16],// Anulare
  [13, 17, 18, 19, 20],// Mignolo
  [0, 5, 9, 13, 17]   // Palmo
]

// ─── Estrae un landmark (x, y, z) da un frame flat-array ─────────────────────
function getLM(frame, offset, index, size) {
  const b = offset + index * size
  return { x: frame[b] ?? 0, y: frame[b + 1] ?? 0, z: frame[b + 2] ?? 0 }
}

// ─── Frame di picco: quello con il maggior movimento delle mani ──────────────
function findPeakFrame(frames) {
  let maxMot = 0, peak = 0
  for (let i = 1; i < frames.length; i++) {
    let mot = 0
    for (let j = 132; j < 258; j++) {
      const d = (frames[i][j] ?? 0) - (frames[i - 1][j] ?? 0)
      mot += d * d
    }
    if (mot > maxMot) { maxMot = mot; peak = i }
  }
  return peak
}

// ─── Disegna lo scheletro completo in un contesto canvas ─────────────────────
function drawFrame(ctx, frame, W, H, alpha = 1) {
  if (!frame || frame.length < 258) return
  ctx.globalAlpha = alpha

  const pose = Array.from({ length: 33 }, (_, i) => {
    const p = getLM(frame, 0, i, 4)
    return { x: p.x * W, y: p.y * H, z: p.z }
  })

  // ── Torso (riempito) ──────────────────────────────────────────────────────
  const [s1, s2, h1, h2] = [pose[11], pose[12], pose[23], pose[24]]
  ctx.beginPath()
  ctx.moveTo(s1.x, s1.y)
  ctx.lineTo(s2.x, s2.y)
  ctx.lineTo(h2.x, h2.y)
  ctx.lineTo(h1.x, h1.y)
  ctx.closePath()
  ctx.fillStyle = `rgba(31,41,55,${0.7 * alpha})`
  ctx.fill()

  // ── Arti ─────────────────────────────────────────────────────────────────
  ctx.lineCap = 'round'
  BODY_LINKS.forEach(([a, b]) => {
    const isArm = (a === 11 && b === 13) || (a === 13 && b === 15) ||
                  (a === 12 && b === 14) || (a === 14 && b === 16)
    // Outer tube
    ctx.beginPath()
    ctx.moveTo(pose[a].x, pose[a].y)
    ctx.lineTo(pose[b].x, pose[b].y)
    ctx.strokeStyle = isArm ? `rgba(129,140,248,${alpha})` : `rgba(79,70,229,${alpha})`
    ctx.lineWidth = isArm ? 10 : 7
    ctx.stroke()
    // Inner highlight
    ctx.beginPath()
    ctx.moveTo(pose[a].x, pose[a].y)
    ctx.lineTo(pose[b].x, pose[b].y)
    ctx.strokeStyle = `rgba(224,231,255,${alpha * 0.5})`
    ctx.lineWidth = 2
    ctx.stroke()
  })

  // ── Testa ─────────────────────────────────────────────────────────────────
  const nose = pose[0]
  const headGrad = ctx.createRadialGradient(nose.x, nose.y - 4, 4, nose.x, nose.y, 22)
  headGrad.addColorStop(0, `rgba(75,85,99,${alpha})`)
  headGrad.addColorStop(1, `rgba(31,41,55,${alpha})`)
  ctx.beginPath()
  ctx.arc(nose.x, nose.y, 22, 0, 2 * Math.PI)
  ctx.fillStyle = headGrad
  ctx.fill()
  ctx.strokeStyle = `rgba(99,102,241,${alpha})`
  ctx.lineWidth = 2
  ctx.stroke()

  // ── Mani ─────────────────────────────────────────────────────────────────
  const drawHand = (offset, primaryColor, highlightColor) => {
    const hand = Array.from({ length: 21 }, (_, i) => {
      const p = getLM(frame, offset, i, 3)
      return { x: p.x * W, y: p.y * H, z: p.z }
    })

    HAND_CHAINS.forEach(chain => {
      ctx.beginPath()
      ctx.moveTo(hand[chain[0]].x, hand[chain[0]].y)
      chain.slice(1).forEach(i => ctx.lineTo(hand[i].x, hand[i].y))
      ctx.strokeStyle = primaryColor
      ctx.lineWidth = 4
      ctx.stroke()
    })

    // Punti con dimensione basata su z (profondità)
    hand.forEach((p, i) => {
      const radius = Math.max(2, 5 - (p.z + 0.5) * 3)  // Più vicino = più grande
      ctx.beginPath()
      ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI)
      // Punta delle dita più luminosa
      ctx.fillStyle = i % 4 === 0 ? highlightColor : `rgba(255,255,255,${alpha * 0.8})`
      ctx.fill()
    })
  }

  const hasLeft  = frame.slice(132, 195).some(v => Math.abs(v) > 0.005)
  const hasRight = frame.slice(195, 258).some(v => Math.abs(v) > 0.005)

  if (hasLeft)  drawHand(132, `rgba(16,185,129,${alpha})`,  `rgba(110,231,183,${alpha})`)
  if (hasRight) drawHand(195, `rgba(139,92,246,${alpha})`,  `rgba(196,181,253,${alpha})`)

  ctx.globalAlpha = 1
}

// ─── Frecce di velocità ───────────────────────────────────────────────────────
function drawVelocityArrows(ctx, prevFrame, currFrame, W, H) {
  if (!prevFrame || !currFrame) return
  [[132, 16], [195, 16]].forEach(([offset, fingerIdx]) => {
    const pp = getLM(prevFrame, offset, fingerIdx, 3)
    const cp = getLM(currFrame, offset, fingerIdx, 3)
    const dx = (cp.x - pp.x) * W, dy = (cp.y - pp.y) * H
    const dist = Math.sqrt(dx * dx + dy * dy)
    if (dist < 4) return   // Troppo piccolo per essere visibile

    const x1 = cp.x * W, y1 = cp.y * H
    const angle = Math.atan2(dy, dx)
    const len = Math.min(dist * 3, 30)
    const x2 = x1 + Math.cos(angle) * len, y2 = y1 + Math.sin(angle) * len

    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.strokeStyle = offset === 132 ? 'rgba(52,211,153,0.8)' : 'rgba(167,139,250,0.8)'
    ctx.lineWidth = 2
    ctx.stroke()

    // Punta freccia
    const aw = 6
    ctx.beginPath()
    ctx.moveTo(x2, y2)
    ctx.lineTo(x2 - Math.cos(angle - 0.4) * aw, y2 - Math.sin(angle - 0.4) * aw)
    ctx.lineTo(x2 - Math.cos(angle + 0.4) * aw, y2 - Math.sin(angle + 0.4) * aw)
    ctx.closePath()
    ctx.fillStyle = offset === 132 ? 'rgba(52,211,153,0.9)' : 'rgba(167,139,250,0.9)'
    ctx.fill()
  })
}

// ─── Componente principale ────────────────────────────────────────────────────
export default function SkeletonRenderer({ signName }) {
  const canvasRef = useRef(null)
  const [landmarks, setLandmarks] = useState(null)
  const [error, setError] = useState(false)
  const frameRef = useRef(0)
  const timerRef = useRef(null)
  const peakRef  = useRef(0)

  useEffect(() => {
    if (!signName) return
    setError(false)
    setLandmarks(null)
    fetch(`./landmarks/${signName.toLowerCase().trim()}.json`)
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(data => {
        setLandmarks(data)
        frameRef.current = 0
        peakRef.current = findPeakFrame(data)
      })
      .catch(() => setError(true))
  }, [signName])

  useEffect(() => {
    if (!landmarks) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    let stopped = false

    const draw = () => {
      if (stopped) return
      const n = landmarks.length
      const idx = frameRef.current % n
      const frame = landmarks[idx]
      const prevFrame = landmarks[(idx - 1 + n) % n]

      ctx.clearRect(0, 0, W, H)

      // ── Frame corrente ──────────────────────────────────────────────────
      drawFrame(ctx, frame, W, H, 1)

      // ── Frecce di velocità ──────────────────────────────────────────────
      drawVelocityArrows(ctx, prevFrame, frame, W, H)

      frameRef.current++

      // Animazione fluida e costante (nessuna pausa e meno blur)
      timerRef.current = setTimeout(() => { if (!stopped) requestAnimationFrame(draw) }, 40)
    }

    requestAnimationFrame(draw)
    return () => { stopped = true; clearTimeout(timerRef.current) }
  }, [landmarks])

  if (error) return (
    <div className="skeleton-placeholder" style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:'100%', gap:'8px', color:'#475569' }}>
      <div style={{ fontSize: '28px' }}>🤲</div>
      <p style={{ fontSize: '12px' }}>Demo non disponibile</p>
    </div>
  )

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={300}
      className="demo-canvas"
      style={{ width: '100%', height: '100%', background: 'transparent' }}
    />
  )
}


