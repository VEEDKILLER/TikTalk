import { useState, useEffect } from 'react'
import CategorySelector from './components/CategorySelector.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import { useAudioRecorder } from './hooks/useAudioRecorder.js'
import * as api from './api/tiktalk.js'

const PHASE = {
  SETUP: 'setup',
  GENERATING: 'generating',
  READY: 'ready',
  RECORDING: 'recording',
  EVALUATING: 'evaluating',
  RESULTS: 'results',
}

function Spinner({ label }) {
  return (
    <div className="flex flex-col items-center gap-3 py-8">
      <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
      {label && <p className="text-sm text-gray-400 text-center">{label}</p>}
    </div>
  )
}

function formatDuration(s) {
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

function Waveform({ active }) {
  return (
    <div className="flex items-end gap-0.5 h-6">
      {[...Array(16)].map((_, i) => (
        <div
          key={i}
          className={`w-1 rounded-full transition-all ${active ? 'bg-red-400' : 'bg-gray-200'}`}
          style={active
            ? { height: `${6 + Math.sin(i * 0.8) * 8 + (i % 3) * 4}px`, animationDelay: `${i * 60}ms` }
            : { height: '4px' }
          }
        />
      ))}
    </div>
  )
}

// ── Left panel: image ─────────────────────────────────────────────────────────
function ImagePanel({ phase, imageBase64, groundTruth, vlmLoading, vlmError }) {
  if (phase === PHASE.SETUP) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-64 bg-gradient-to-br from-blue-50 to-sky-100 rounded-2xl border-2 border-dashed border-blue-200 text-center p-8">
        <span className="text-5xl mb-3">🖼️</span>
        <p className="text-gray-400 text-sm">Your practice image<br />will appear here</p>
      </div>
    )
  }

  if (phase === PHASE.GENERATING) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-64 bg-blue-50 rounded-2xl">
        <Spinner label="Generating picture… 10–20 seconds" />
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Image */}
      <div className="relative rounded-2xl overflow-hidden bg-gray-100 shadow-md">
        <img
          src={`data:image/png;base64,${imageBase64}`}
          alt="Practice scene"
          className="w-full object-cover"
        />
        {/* VLM status badge */}
        {vlmLoading && (
          <div className="absolute top-2 right-2 bg-white/90 backdrop-blur rounded-full px-3 py-1 text-xs text-gray-500 flex items-center gap-1.5 shadow">
            <div className="w-2.5 h-2.5 border-2 border-gray-300 border-t-blue-400 rounded-full animate-spin" />
            Analysing…
          </div>
        )}
        {!vlmLoading && groundTruth && (
          <div className="absolute top-2 right-2 bg-green-500/90 backdrop-blur rounded-full px-3 py-1 text-xs text-white shadow">
            ✓ Ready
          </div>
        )}
        {vlmError && (
          <div className="absolute top-2 right-2 bg-red-500/90 backdrop-blur rounded-full px-3 py-1 text-xs text-white shadow">
            ⚠ Analysis failed
          </div>
        )}
      </div>

      {/* Teaching keywords */}
      {groundTruth?.teaching_focus_words?.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 mb-1.5">Key words to use:</p>
          <div className="flex flex-wrap gap-1.5">
            {groundTruth.teaching_focus_words.map((w) => (
              <span key={w} className="px-2.5 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
                {w}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Right panel: record + results ─────────────────────────────────────────────
function InteractionPanel({
  phase, recorder, vlmLoading, vlmError, result,
  onStart, onStop, onSubmit, onRetry, onNewImage,
}) {
  const canSubmit = !!recorder.audioBlob && !vlmLoading

  // ── RESULTS ─────────────────────────────────────────────────────────────────
  if (phase === PHASE.RESULTS && result) {
    return (
      <ResultPanel
        result={result}
        onRetry={onRetry}
        onNewImage={onNewImage}
      />
    )
  }

  // ── EVALUATING ───────────────────────────────────────────────────────────────
  if (phase === PHASE.EVALUATING) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <Spinner label="Scoring your response…" />
      </div>
    )
  }

  // ── SETUP placeholder ────────────────────────────────────────────────────────
  if (phase === PHASE.SETUP || phase === PHASE.GENERATING) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-64 text-center p-8">
        <span className="text-5xl mb-3">🎤</span>
        <p className="text-gray-400 text-sm">Recording and score<br />will appear here</p>
      </div>
    )
  }

  // ── READY / RECORDING ────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Instruction */}
      <div className="bg-blue-50 rounded-2xl px-4 py-3 border border-blue-100">
        <p className="text-sm font-semibold text-blue-700">Look at the picture and describe what you see!</p>
        <p className="text-xs text-blue-400 mt-0.5">Try to mention the people, actions, and place.</p>
      </div>

      {/* Recording state */}
      {phase === PHASE.RECORDING ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between bg-red-50 rounded-2xl px-4 py-3 border-2 border-red-200">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse" />
              <span className="font-mono font-bold text-red-600 text-lg">
                {formatDuration(recorder.duration)}
              </span>
            </div>
            <Waveform active />
          </div>
          <button
            className="btn-outline w-full border-red-300 text-red-600 hover:bg-red-50"
            onClick={onStop}
          >
            ⏹ Stop Recording
          </button>
        </div>
      ) : !recorder.audioBlob ? (
        /* No recording yet */
        <button
          className="btn-danger w-full py-4 text-base flex items-center justify-center gap-2"
          onClick={onStart}
          disabled={!!recorder.error}
        >
          🎤 Start Recording
        </button>
      ) : (
        /* Has recording */
        <div className="space-y-3">
          <div className="bg-gray-50 rounded-2xl p-3 border border-gray-100">
            <p className="text-xs text-gray-400 mb-1.5">
              Your recording · {formatDuration(recorder.duration)}
            </p>
            <audio
              src={URL.createObjectURL(recorder.audioBlob)}
              controls
              className="w-full"
            />
          </div>
          <div className="flex gap-2">
            <button className="btn-outline flex-1" onClick={recorder.reset}>
              🔄 Re-record
            </button>
            <button
              className="btn-success flex-1"
              onClick={onSubmit}
              disabled={!canSubmit}
            >
              {vlmLoading ? '⏳ Wait…' : '✓ Submit'}
            </button>
          </div>
          {vlmLoading && (
            <p className="text-xs text-center text-gray-400">
              Image analysis still in progress, please wait…
            </p>
          )}
        </div>
      )}

      {recorder.error && (
        <p className="text-xs text-red-500 text-center">{recorder.error}</p>
      )}
      {vlmError && (
        <p className="text-xs text-red-500 text-center">{vlmError}</p>
      )}
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [phase, setPhase] = useState(PHASE.SETUP)
  const [options, setOptions] = useState({ categories: [], characters: [], actions: {} })
  const [category, setCategory] = useState('')
  const [character, setCharacter] = useState('boy')
  const [action, setAction] = useState(null)  // null = "Surprise me" (random scene)
  const [imageBase64, setImageBase64] = useState(null)
  const [imageId, setImageId] = useState(null)
  const [groundTruth, setGroundTruth] = useState(null)
  const [vlmLoading, setVlmLoading] = useState(false)
  const [vlmError, setVlmError] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const recorder = useAudioRecorder()

  useEffect(() => {
    api.getCategories()
      .then((data) => {
        setOptions(data)
        setCategory(data.categories[0] || 'outdoor')
        setCharacter(data.characters[0] || 'boy')
      })
      .catch(() => {
        setOptions({
          categories: ['daily_life', 'school', 'outdoor', 'community', 'nature', 'festivals', 'helping', 'transportation'],
          characters: ['boy', 'girl', 'child', 'mother', 'father', 'grandmother', 'grandfather'],
          actions: {},
        })
        setCategory('outdoor')
      })
  }, [])

  // Changing the scene clears any previously chosen action (actions are
  // category-specific), falling back to "Surprise me".
  function handleCategory(cat) {
    setCategory(cat)
    setAction(null)
  }

  async function handleGenerate() {
    if (!category) return
    setError(null)
    setPhase(PHASE.GENERATING)
    setGroundTruth(null)
    setVlmError(null)
    recorder.reset()
    setResult(null)

    try {
      const data = await api.generateImage({ category, character, action })
      setImageId(data.image_id)
      setImageBase64(data.image_base64)
      setPhase(PHASE.READY)

      // VLM runs in background while student views image
      setVlmLoading(true)
      api.getGroundTruth(data.image_id)
        .then((vlm) => { setGroundTruth(vlm.ground_truth); setVlmLoading(false) })
        .catch((err) => { setVlmError('Analysis failed: ' + err.message); setVlmLoading(false) })
    } catch (err) {
      setError('Image generation failed: ' + err.message)
      setPhase(PHASE.SETUP)
    }
  }

  async function handleStart() {
    setError(null)
    await recorder.start()
    setPhase(PHASE.RECORDING)
  }

  function handleStop() {
    recorder.stop()
    setPhase(PHASE.READY)
  }

  async function handleSubmit() {
    if (!recorder.audioBlob || !imageId) return
    setError(null)
    setPhase(PHASE.EVALUATING)
    try {
      const data = await api.evaluate(recorder.audioBlob, imageId)
      setResult(data)
      setPhase(PHASE.RESULTS)
    } catch (err) {
      setError('Evaluation failed: ' + err.message)
      setPhase(PHASE.READY)
    }
  }

  function handleRetry() {
    recorder.reset()
    setResult(null)
    setError(null)
    setPhase(PHASE.READY)
  }

  function handleNewImage() {
    recorder.reset()
    setResult(null)
    setImageBase64(null)
    setImageId(null)
    setGroundTruth(null)
    setError(null)
    setPhase(PHASE.SETUP)
  }

  const showTwoColumn = phase !== PHASE.SETUP

  return (
    <div className="min-h-screen bg-sky-50 flex flex-col">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="bg-white shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center gap-3">
          <span className="text-2xl">🎤</span>
          <div>
            <h1 className="font-extrabold text-lg text-blue-700 leading-tight">TikTalk</h1>
            <p className="text-xs text-gray-400">English Speaking Practice</p>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-6 flex flex-col gap-5">

        {/* ── Setup card (full width, only in SETUP phase) ─────────────────── */}
        {phase === PHASE.SETUP && (
          <div className="card">
            <div className="mb-5">
              <h2 className="text-lg font-extrabold text-gray-800">Choose a scene to practise</h2>
              <p className="text-sm text-gray-400 mt-0.5">Select a topic and a character, then generate your practice image.</p>
            </div>
            <CategorySelector
              categories={options.categories}
              characters={options.characters}
              actions={options.actions}
              selected={category}
              character={character}
              action={action}
              onCategory={handleCategory}
              onCharacter={setCharacter}
              onAction={setAction}
            />
            <div className="mt-6 border-t border-gray-100 pt-5">
              <button
                className="btn-primary w-full text-base py-3 flex items-center justify-center gap-2"
                onClick={handleGenerate}
                disabled={!category}
              >
                <span>Generate Practice Image</span>
                <span className="text-lg">→</span>
              </button>
              {!category && (
                <p className="text-xs text-center text-gray-400 mt-2">Pick a scene above to continue</p>
              )}
            </div>
          </div>
        )}

        {/* ── Compact scene strip (after SETUP) ────────────────────────────── */}
        {phase !== PHASE.SETUP && (
          <div className="card py-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <span className="font-medium capitalize">{category.replace('_', ' ')}</span>
              <span className="text-gray-300">·</span>
              <span>{character}</span>
              {action && (
                <>
                  <span className="text-gray-300">·</span>
                  <span className="italic">{action}</span>
                </>
              )}
            </div>
            {phase !== PHASE.EVALUATING && (
              <button
                onClick={handleNewImage}
                className="text-sm text-blue-500 hover:underline font-medium"
              >
                ← New Image
              </button>
            )}
          </div>
        )}

        {/* ── Two-column layout ─────────────────────────────────────────────── */}
        {showTwoColumn && (
          <div className="grid grid-cols-2 gap-5 flex-1">

            {/* Left: image */}
            <div className="card">
              <ImagePanel
                phase={phase}
                imageBase64={imageBase64}
                groundTruth={groundTruth}
                vlmLoading={vlmLoading}
                vlmError={vlmError}
              />
            </div>

            {/* Right: record + results */}
            <div className="card overflow-y-auto">
              <InteractionPanel
                phase={phase}
                recorder={recorder}
                vlmLoading={vlmLoading}
                vlmError={vlmError}
                result={result}
                onStart={handleStart}
                onStop={handleStop}
                onSubmit={handleSubmit}
                onRetry={handleRetry}
                onNewImage={handleNewImage}
              />
            </div>

          </div>
        )}

        {/* Global error */}
        {error && (
          <div className="card border-2 border-red-200 bg-red-50">
            <p className="text-sm text-red-600">⚠️ {error}</p>
          </div>
        )}

      </main>
    </div>
  )
}
