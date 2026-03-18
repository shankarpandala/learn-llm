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
 * Slate-themed proof block with numbered steps.
 *
 * Props:
 *   title        {string}  Proof title (e.g. "Proof of Theorem 2.1")
 *   steps        {array}   Array of { formula, explanation }
 *   collapsible  {bool}    Whether the proof is collapsible (default true)
 *   defaultOpen  {bool}    Whether to start open (default false)
 *   children     {node}    Optional body JSX
 *   id           {string}  Optional anchor id
 */
export default function ProofBlock({
  title = 'Proof',
  steps,
  collapsible = true,
  defaultOpen = false,
  children,
  id,
}) {
  const [open, setOpen] = useState(defaultOpen || !collapsible)

  const proofContent = (
    <div className="px-5 pb-4 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300">
      {steps && steps.length > 0 ? (
        <ol className="space-y-3">
          {steps.map((step, i) => (
            <li key={i} className="flex flex-col gap-1">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                Step {i + 1}
              </span>
              {step.formula && (
                <span className="font-mono text-slate-800 dark:text-slate-200">
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
      ) : (
        children
      )}

      {/* QED symbol */}
      <div className="mt-4 text-right text-lg text-slate-400 dark:text-slate-500 select-none">
        &#9632;
      </div>
    </div>
  )

  return (
    <div
      id={id}
      className="my-6 rounded-xl border border-slate-200 bg-slate-50/60 dark:border-slate-600/30 dark:bg-slate-900/40"
    >
      {collapsible ? (
        <>
          <button
            onClick={() => setOpen(!open)}
            className="flex w-full items-center gap-3 px-5 py-3 text-left"
            aria-expanded={open}
          >
            <svg
              className={`h-4 w-4 shrink-0 text-slate-500 transition-transform ${open ? 'rotate-90' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              {renderMathText(title)}
            </span>
          </button>

          <AnimatePresence>
            {open && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                {proofContent}
              </motion.div>
            )}
          </AnimatePresence>
        </>
      ) : (
        <>
          <div className="px-5 pt-4 pb-2">
            <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              {renderMathText(title)}
            </h4>
          </div>
          {proofContent}
        </>
      )}
    </div>
  )
}
