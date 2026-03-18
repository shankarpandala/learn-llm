import { useState } from 'react'
import { InlineMath } from 'react-katex'
import { motion, AnimatePresence } from 'framer-motion'

function renderMathText(text) {
  if (!text) return null
  const parts = text.split(/(\$[^$]+\$)/g)
  return parts.map((part, i) => {
    if (part.startsWith('$') && part.endsWith('$')) {
      return <InlineMath key={i} math={part.slice(1, -1)} />
    }
    return <span key={i}>{part}</span>
  })
}

const DIFFICULTY_STYLES = {
  easy: {
    bg: 'bg-green-100 dark:bg-green-500/20',
    text: 'text-green-700 dark:text-green-400',
    border: 'border-green-300 dark:border-green-500/40',
    label: 'Easy',
  },
  medium: {
    bg: 'bg-yellow-100 dark:bg-yellow-500/20',
    text: 'text-yellow-700 dark:text-yellow-400',
    border: 'border-yellow-300 dark:border-yellow-500/40',
    label: 'Medium',
  },
  hard: {
    bg: 'bg-red-100 dark:bg-red-500/20',
    text: 'text-red-700 dark:text-red-400',
    border: 'border-red-300 dark:border-red-500/40',
    label: 'Hard',
  },
  challenge: {
    bg: 'bg-purple-100 dark:bg-purple-500/20',
    text: 'text-purple-700 dark:text-purple-400',
    border: 'border-purple-300 dark:border-purple-500/40',
    label: 'Challenge',
  },
}

function DifficultyBadge({ level }) {
  const style = DIFFICULTY_STYLES[level] || DIFFICULTY_STYLES.medium
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${style.bg} ${style.text} ${style.border}`}
    >
      {style.label}
    </span>
  )
}

function CollapsibleSection({ label, children }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline"
        aria-expanded={open}
      >
        <svg
          className={`h-3 w-3 shrink-0 transition-transform ${open ? 'rotate-90' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        {open ? `Hide ${label}` : `Show ${label}`}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 rounded-lg bg-gray-50 p-3 text-sm text-gray-700 dark:bg-gray-800/60 dark:text-gray-300">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/**
 * Exercise set with numbered items, difficulty badges, collapsible hints and solutions.
 *
 * Props:
 *   title     {string}  Set title (e.g. "Practice Exercises")
 *   exercises {array}   Array of:
 *     {
 *       problem:    string   — problem statement with $math$
 *       difficulty: string   — 'easy' | 'medium' | 'hard' | 'challenge'
 *       hint:       string   — optional hint with $math$
 *       solution:   string   — optional solution with $math$
 *     }
 *   id        {string}  Optional anchor id
 */
export default function ExerciseBlock({ title = 'Exercises', exercises = [], id }) {
  return (
    <div
      id={id}
      className="my-6 rounded-xl border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-900/60"
    >
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-gray-200 px-5 py-3 dark:border-gray-700">
        <svg
          className="h-5 w-5 text-gray-500 dark:text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
          />
        </svg>
        <h4 className="text-base font-semibold text-gray-900 dark:text-gray-100">
          {title}
        </h4>
        <span className="ml-auto rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-600 dark:bg-gray-800 dark:text-gray-400">
          {exercises.length} {exercises.length === 1 ? 'problem' : 'problems'}
        </span>
      </div>

      {/* Exercise list */}
      <div className="divide-y divide-gray-100 dark:divide-gray-800">
        {exercises.map((ex, i) => (
          <div key={i} className="px-5 py-4">
            <div className="flex items-start gap-3">
              {/* Number */}
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gray-200 text-[11px] font-bold text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                {i + 1}
              </span>

              <div className="flex-1 min-w-0">
                {/* Problem + difficulty */}
                <div className="flex flex-wrap items-start gap-2">
                  <p className="flex-1 text-sm leading-relaxed text-gray-700 dark:text-gray-300">
                    {renderMathText(ex.problem)}
                  </p>
                  {ex.difficulty && <DifficultyBadge level={ex.difficulty} />}
                </div>

                {/* Hint */}
                {ex.hint && (
                  <CollapsibleSection label="Hint">
                    {renderMathText(ex.hint)}
                  </CollapsibleSection>
                )}

                {/* Solution */}
                {ex.solution && (
                  <CollapsibleSection label="Solution">
                    {renderMathText(ex.solution)}
                  </CollapsibleSection>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
