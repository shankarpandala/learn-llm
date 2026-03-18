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
 * Indigo-themed theorem callout with T badge.
 *
 * Props:
 *   title       {string}   Theorem name / label
 *   statement   {string}   Main statement with $math$ support
 *   proof       {string}   Collapsible proof text with $math$
 *   proofSteps  {array}    Alternative: array of { formula, explanation }
 *   corollaries {string[]} Optional list of corollary strings with $math$
 *   children    {node}     Optional body JSX
 *   id          {string}   Optional anchor id
 */
export default function TheoremBlock({
  title,
  statement,
  proof,
  proofSteps,
  corollaries,
  children,
  id,
}) {
  const [proofOpen, setProofOpen] = useState(false)

  return (
    <div
      id={id}
      className="my-6 rounded-xl border border-indigo-200 bg-indigo-50/60 dark:border-indigo-500/30 dark:bg-indigo-950/30"
    >
      {/* Header */}
      <div className="flex items-start gap-3 px-5 pt-4 pb-2">
        <span
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-xs font-bold text-white"
          aria-hidden="true"
        >
          T
        </span>
        <h4 className="text-base font-semibold text-indigo-900 dark:text-indigo-200 leading-snug pt-0.5">
          {renderMathText(title)}
        </h4>
      </div>

      {/* Statement */}
      <div className="px-5 pb-3 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        {statement ? <p>{renderMathText(statement)}</p> : children}
      </div>

      {/* Corollaries */}
      {corollaries && corollaries.length > 0 && (
        <div className="px-5 pb-3 pl-[3.25rem]">
          <p className="text-xs font-semibold uppercase tracking-wide text-indigo-600 dark:text-indigo-400 mb-1">
            Corollaries
          </p>
          <ul className="list-disc list-inside space-y-1 text-sm text-gray-700 dark:text-gray-300">
            {corollaries.map((c, i) => (
              <li key={i}>{renderMathText(c)}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Collapsible Proof */}
      {(proof || proofSteps) && (
        <div className="border-t border-indigo-200 dark:border-indigo-500/20">
          <button
            onClick={() => setProofOpen(!proofOpen)}
            className="flex w-full items-center gap-2 px-5 py-3 text-sm font-medium text-indigo-700 dark:text-indigo-300 hover:bg-indigo-100/60 dark:hover:bg-indigo-900/30 transition-colors"
            aria-expanded={proofOpen}
          >
            <svg
              className={`h-4 w-4 shrink-0 transition-transform ${proofOpen ? 'rotate-90' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
            {proofOpen ? 'Hide Proof' : 'Show Proof'}
          </button>

          <AnimatePresence>
            {proofOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div className="px-5 pb-4 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300">
                  {proof && <p>{renderMathText(proof)}</p>}

                  {proofSteps && (
                    <ol className="space-y-3 mt-2">
                      {proofSteps.map((step, i) => (
                        <li key={i} className="flex flex-col gap-1">
                          <span className="text-xs font-semibold text-indigo-500 dark:text-indigo-400">
                            Step {i + 1}
                          </span>
                          {step.formula && (
                            <span className="font-mono text-indigo-800 dark:text-indigo-200">
                              {renderMathText(step.formula)}
                            </span>
                          )}
                          {step.explanation && (
                            <span className="text-gray-600 dark:text-gray-400">
                              {renderMathText(step.explanation)}
                            </span>
                          )}
                        </li>
                      ))}
                    </ol>
                  )}

                  {/* QED symbol */}
                  <div className="mt-4 text-right text-lg text-indigo-400 dark:text-indigo-500 select-none">
                    &#9632;
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
