import { useEffect, useRef, useState, useCallback } from 'react'

const BAR_POSITION = 0.25
const FREQ_SMOOTH_RADIUS = 6
const EMA_ALPHA = 0.12
const CONTRAST_GAMMA = 2.2
const CONTRAST_BOOST = 1.8
const SPEC_HEIGHT_RATIO = 0.5

function App() {
  const canvasRef = useRef(null)
  const audioRef = useRef(null)
  const animFrameRef = useRef(null)
  const renderRef = useRef(null)
  const [duration, setDuration] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [audioSrc, setAudioSrc] = useState(null)
  const [files, setFiles] = useState([])
  const [currentFile, setCurrentFile] = useState(null)

  const loadFile = useCallback((file) => {
    setLoaded(false)
    setPlaying(false)
    setCurrentFile(file)
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    setAudioSrc(`/api/audio/${file}`)

    const metaPromise = fetch(`/api/metadata/${file}`).then(r => r.json())
    const imgPromise = new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = reject
      img.src = `/api/spectrogram/${file}`
    })

    Promise.all([metaPromise, imgPromise]).then(([meta, img]) => {
      const tmp = document.createElement('canvas')
      tmp.width = img.width
      tmp.height = img.height
      const tmpCtx = tmp.getContext('2d')
      tmpCtx.drawImage(img, 0, 0)
      const imageData = tmpCtx.getImageData(0, 0, img.width, img.height)
      const px = imageData.data

      const sw = img.width
      const sh = img.height
      const chR = new Float32Array(sh * sw)
      const chG = new Float32Array(sh * sw)
      const chB = new Float32Array(sh * sw)
      for (let i = 0; i < sh * sw; i++) {
        const off = i * 4
        chR[i] = px[off] / 255
        chG[i] = px[off + 1] / 255
        chB[i] = px[off + 2] / 255
      }

      renderRef.current = {
        img, chR, chG, chB, sw, sh,
        emaR: new Float32Array(sh),
        emaG: new Float32Array(sh),
        emaB: new Float32Array(sh),
        smR: new Float32Array(sh),
        smG: new Float32Array(sh),
        smB: new Float32Array(sh),
        prevCol: -1,
        scaledCanvas: null,
        scaledW: 0,
        scaledH: 0,
      }
      setDuration(meta.duration)
      setLoaded(true)
    })
  }, [])

  useEffect(() => {
    fetch('/api/sounds')
      .then(r => r.json())
      .then(data => {
        if (data.files && data.files.length > 0) {
          setFiles(data.files)
          loadFile(data.files[0])
        }
      })
  }, [loadFile])

  useEffect(() => {
    if (!loaded) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      if (renderRef.current) {
        renderRef.current.scaledCanvas = null
      }
    }
    resize()
    window.addEventListener('resize', resize)

    const draw = () => {
      const R = renderRef.current
      const cw = canvas.width
      const ch = canvas.height
      const audio = audioRef.current

      if (!audio || !R || duration <= 0) {
        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, cw, ch)
        animFrameRef.current = requestAnimationFrame(draw)
        return
      }

      const specH = Math.round(ch * SPEC_HEIGHT_RATIO)
      const specTop = Math.round((ch - specH) / 2)

      if (!R.scaledCanvas || R.scaledW !== cw || R.scaledH !== specH) {
        const sc = document.createElement('canvas')
        sc.width = cw
        sc.height = specH
        const sctx = sc.getContext('2d')
        sctx.imageSmoothingEnabled = true
        sctx.drawImage(R.img, 0, 0, cw, specH)
        R.scaledCanvas = sc
        R.scaledW = cw
        R.scaledH = specH
      }

      const progress = Math.min(audio.currentTime / duration, 1)
      const barX = Math.floor(cw * BAR_POSITION)
      const currentCol = Math.min(Math.floor(progress * R.sw), R.sw - 1)

      const { chR, chG, chB, sw, sh, emaR, emaG, emaB, smR, smG, smB } = R
      if (currentCol >= 0) {
        if (R.prevCol < 0 || currentCol < R.prevCol) {
          for (let sy = 0; sy < sh; sy++) {
            const idx = sy * sw + currentCol
            emaR[sy] = chR[idx]
            emaG[sy] = chG[idx]
            emaB[sy] = chB[idx]
          }
        } else {
          for (let col = R.prevCol + 1; col <= currentCol; col++) {
            for (let sy = 0; sy < sh; sy++) {
              const idx = sy * sw + col
              emaR[sy] += (chR[idx] - emaR[sy]) * EMA_ALPHA
              emaG[sy] += (chG[idx] - emaG[sy]) * EMA_ALPHA
              emaB[sy] += (chB[idx] - emaB[sy]) * EMA_ALPHA
            }
          }
        }
        R.prevCol = currentCol
      }

      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, cw, ch)

      const scrollX = Math.round(progress * cw)
      ctx.save()
      ctx.beginPath()
      ctx.rect(barX + 1, specTop, cw - barX, specH)
      ctx.clip()
      ctx.drawImage(R.scaledCanvas, barX - scrollX, specTop)
      ctx.restore()

      if (currentCol >= 0) {
        for (let sy = 0; sy < sh; sy++) {
          let sR = 0, sG = 0, sB = 0
          const lo = Math.max(0, sy - FREQ_SMOOTH_RADIUS)
          const hi = Math.min(sh - 1, sy + FREQ_SMOOTH_RADIUS)
          const n = hi - lo + 1
          for (let k = lo; k <= hi; k++) {
            sR += emaR[k]
            sG += emaG[k]
            sB += emaB[k]
          }
          smR[sy] = sR / n
          smG[sy] = sG / n
          smB[sy] = sB / n
        }

        let bandStart = specTop
        let prevKey = ''
        let prevColor = ''
        const specBottom = specTop + specH
        for (let y = specTop; y <= specBottom; y++) {
          let key = '0,0,0'
          let color = 'rgb(0,0,0)'
          if (y < specBottom) {
            const specRow = Math.min(Math.floor((y - specTop) / specH * sh), sh - 1)
            const rr = smR[specRow], gg = smG[specRow], bb = smB[specRow]
            const lum = rr * 0.299 + gg * 0.587 + bb * 0.114
            const adj = Math.min(1, Math.pow(lum, CONTRAST_GAMMA) * CONTRAST_BOOST)
            const scale = lum > 0.001 ? adj / lum : 0
            const fr = Math.min(255, Math.round(rr * scale * 255))
            const fg = Math.min(255, Math.round(gg * scale * 255))
            const fb = Math.min(255, Math.round(bb * scale * 255))
            key = `${fr},${fg},${fb}`
            color = `rgb(${fr},${fg},${fb})`
          }
          if (key !== prevKey || y === specBottom) {
            if (prevKey && y > bandStart) {
              ctx.fillStyle = prevColor
              ctx.fillRect(0, bandStart, barX, y - bandStart)
            }
            bandStart = y
            prevKey = key
            prevColor = color
          }
        }
      }

      ctx.globalAlpha = 0.2
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 10
      ctx.beginPath()
      ctx.moveTo(barX, specTop)
      ctx.lineTo(barX, specTop + specH)
      ctx.stroke()

      ctx.globalAlpha = 0.9
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(barX, specTop)
      ctx.lineTo(barX, specTop + specH)
      ctx.stroke()

      ctx.globalAlpha = 1.0

      animFrameRef.current = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      window.removeEventListener('resize', resize)
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    }
  }, [loaded, duration])

  const handleClick = () => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) {
      audio.pause()
      setPlaying(false)
    } else {
      audio.play()
      setPlaying(true)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.value
    if (file && file !== currentFile) {
      loadFile(file)
    }
  }

  return (
    <>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{ display: 'block', cursor: 'pointer' }}
      />
      {audioSrc && (
        <audio
          ref={audioRef}
          src={audioSrc}
          preload="auto"
          onEnded={() => setPlaying(false)}
        />
      )}
      {files.length > 1 && (
        <select
          value={currentFile || ''}
          onChange={handleFileChange}
          style={{
            position: 'fixed',
            bottom: 16,
            right: 16,
            background: 'rgba(30, 30, 30, 0.6)',
            color: '#555',
            border: '1px solid #333',
            borderRadius: 4,
            padding: '4px 8px',
            fontSize: 12,
            outline: 'none',
            cursor: 'pointer',
            fontFamily: 'system-ui, sans-serif',
          }}
        >
          {files.map(f => (
            <option key={f} value={f}>{f.replace(/\.[^.]+$/, '')}</option>
          ))}
        </select>
      )}
    </>
  )
}

export default App
