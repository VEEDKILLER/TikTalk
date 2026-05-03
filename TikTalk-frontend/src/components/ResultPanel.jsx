function Stars({ score }) {
  const filled = Math.round((score / 100) * 5)
  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`text-xl ${i <= filled ? 'text-yellow-400' : 'text-gray-200'}`}>
          ★
        </span>
      ))}
    </div>
  )
}

function DimensionRow({ label, score, color }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-28 text-sm font-medium text-gray-600 shrink-0">{label}</span>
      <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${score}%` }}
        />
      </div>
      <div className="flex items-center gap-1.5 w-24 shrink-0">
        <Stars score={score} />
      </div>
      <span className="w-10 text-sm font-bold text-right text-gray-700">{Math.round(score)}</span>
    </div>
  )
}

const ENCOURAGEMENT = {
  high: { emoji: '🌟', text: 'Excellent work! You did a great job!', bg: 'bg-yellow-50 border-yellow-200' },
  medium: { emoji: '👍', text: 'Good effort! Keep practising!', bg: 'bg-blue-50 border-blue-200' },
  low: { emoji: '💪', text: "Don't give up! Try again!", bg: 'bg-purple-50 border-purple-200' },
}

export default function ResultPanel({ result, onRetry, onNewImage }) {
  const { scores, transcript, feedback, risk_flags, score_details } = result

  const enc = ENCOURAGEMENT[feedback.encouragement_level] || ENCOURAGEMENT.medium

  return (
    <div className="space-y-5 animate-[fadeIn_0.4s_ease-in]">

      {/* Total score */}
      <div className="card text-center">
        <p className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-1">Your Score</p>
        <div className="flex items-center justify-center gap-3 mb-2">
          <span className="text-6xl font-extrabold text-blue-600">{Math.round(scores.total)}</span>
          <span className="text-2xl text-gray-400 font-light">/100</span>
        </div>
        <div className="flex justify-center">
          <Stars score={scores.total} />
        </div>
      </div>

      {/* Encouragement banner */}
      <div className={`card border-2 flex items-center gap-3 ${enc.bg}`}>
        <span className="text-3xl">{enc.emoji}</span>
        <p className="font-semibold text-gray-700">{enc.text}</p>
      </div>

      {/* Risk flags */}
      {risk_flags.length > 0 && (
        <div className="card border-2 border-orange-200 bg-orange-50">
          <p className="font-semibold text-orange-700 mb-1">⚠️ Notice</p>
          {risk_flags.includes('silence') && (
            <p className="text-sm text-orange-600">Your recording was very short. Try speaking more!</p>
          )}
          {risk_flags.includes('off_topic') && (
            <p className="text-sm text-orange-600">Your response might be off-topic. Try to describe the picture.</p>
          )}
          {risk_flags.includes('low_asr_confidence') && (
            <p className="text-sm text-orange-600">We had trouble understanding you. Speak clearly and closer to the microphone.</p>
          )}
        </div>
      )}

      {/* Dimension scores */}
      <div className="card space-y-3">
        <p className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-1">Breakdown</p>
        <DimensionRow label="Content"       score={scores.semantic}      color="bg-blue-400" />
        <DimensionRow label="Grammar"       score={scores.grammar}       color="bg-green-400" />
        <DimensionRow label="Pronunciation" score={scores.pronunciation} color="bg-purple-400" />
        <DimensionRow label="Fluency"       score={scores.fluency}       color="bg-orange-400" />
      </div>

      {/* Feedback */}
      <div className="card space-y-3">
        <p className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Feedback</p>

        {feedback.positive_points.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-green-600 uppercase mb-1">✅ Well done</p>
            <ul className="space-y-1">
              {feedback.positive_points.map((p, i) => (
                <li key={i} className="text-sm text-gray-700 bg-green-50 rounded-xl px-3 py-1.5">{p}</li>
              ))}
            </ul>
          </div>
        )}

        {feedback.improvement_points.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-blue-600 uppercase mb-1">💡 Try next time</p>
            <ul className="space-y-1">
              {feedback.improvement_points.map((p, i) => (
                <li key={i} className="text-sm text-gray-700 bg-blue-50 rounded-xl px-3 py-1.5">{p}</li>
              ))}
            </ul>
          </div>
        )}

        {feedback.grammar_focus.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-purple-600 uppercase mb-1">📝 Grammar tip</p>
            <ul className="space-y-1">
              {feedback.grammar_focus.map((p, i) => (
                <li key={i} className="text-sm text-gray-700 bg-purple-50 rounded-xl px-3 py-1.5">{p}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Transcript */}
      <div className="card">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">What you said</p>
        <p className="text-sm text-gray-700 italic">
          "{transcript.clean || transcript.raw || '(nothing detected)'}"
        </p>
        {score_details && (
          <div className="flex gap-4 mt-2 text-xs text-gray-400">
            <span>🗣 {score_details.speech_rate_wpm} wpm</span>
            <span>⏸ {score_details.pause_count} pauses</span>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button onClick={onRetry} className="btn-outline flex-1">
          🔄 Try Again
        </button>
        <button onClick={onNewImage} className="btn-primary flex-1">
          🖼️ New Image
        </button>
      </div>
    </div>
  )
}
