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

/**
 * Emerald-themed example block with E badge.
 *
 * Props:
 *   title    {string}  Example title / label
 *   problem  {string}  Problem statement with $math$ support
 *   steps    {array}   Array of { formula, explanation } for the solution
 *   children {node}    Optional body JSX (used when no steps provided)
 *   id       {string}  Optional anchor id
 */
export default function ExampleBlock({ title, problem, steps, children, id }) {
  const [solutionOpen, setSolutionOpen] = useState(false)

  return (
    <div
      id={id}
      className="my-6 rounded-xl border border-emerald-200 bg-emerald-50/60 dark:border-emerald-500/30 dark:bg-emerald-950/30"
    >
      {/* Header */}
      <div className="flex items-start gap-3 px-5 pt-4 pb-2">
        <span
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-600 text-xs font-bold text-white"
          aria-hidden="true"
        >
          E
        </span>
        <h4 className="text-base font-semibold text-emerald-900 dark:text-emerald-200 leading-snug pt-0.5">
          {renderMathText(title)}
        </h4>
      </div>

      {/* Problem statement */}
      <div className="px-5 pb-3 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        {problem ? <p>{renderMathText(problem)}</p> : children}
      </div>

      {/* Collapsible solution */}
      {steps && steps.length > 0 && (
        <div className="border-t border-emerald-200 dark:border-emerald-500/20">
          <button
            onClick={() => setSolutionOpen(!solutionOpen)}
            className="flex w-full items-center gap-2 px-5 py-3 text-sm font-medium text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100/60 dark:hover:bg-emerald-900/30 transition-colors"
            aria-expanded={solutionOpen}
          >
            <svg
              className={`h-4 w-4 shrink-0 transition-transform ${solutionOpen ? 'rotate-90' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
            {solutionOpen ? 'Hide Solution' : 'Show Solution'}
          </button>

          <AnimatePresence>
            {solutionOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div className="px-5 pb-4 pl-[3.25rem]">
                  <ol className="space-y-4">
                    {steps.map((step, i) => (
                      <li key={i} className="relative pl-8">
                        {/* Step number circle */}
                        <span className="absolute left-0 top-0 flex h-6 w-6 items-center justify-center rounded-full bg-emerald-600 text-[11px] font-bold text-white">
                          {i + 1}
                        </span>
                        <div className="flex flex-col gap-1">
                          {step.formula && (
                            <span className="font-mono text-sm text-emerald-800 dark:text-emerald-200">
                              {renderMathText(step.formula)}
                            </span>
                          )}
                          {step.explanation && (
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                              {renderMathText(step.explanation)}
                            </span>
                          )}
                        </div>
                      </li>
                    ))}
                  </ol>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
