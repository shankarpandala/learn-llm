import { useState } from 'react'
import { InlineMath } from 'react-katex'

/**
 * Splits text on $...$ delimiters and renders KaTeX inline math.
 */
function renderMathText(text) {
  if (!text) return null
  const parts = text.split(/(\$[^$]+\$)/g)
  return parts.map((part, i) => {
    if (part.startsWith('$') && part.endsWith('$')) {
      const math = part.slice(1, -1)
      return <InlineMath key={i} math={math} />
    }
    return <span key={i}>{part}</span>
  })
}

/**
 * Purple-themed definition callout with D badge.
 *
 * Props:
 *   title      {string}  Definition title
 *   children   {node}    Definition body (JSX) — used if `definition` not provided
 *   definition {string}  Plain-text definition with $math$ support
 *   notation   {string}  Optional notation line with $math$ support
 *   id         {string}  Optional HTML id for anchor linking
 */
export default function DefinitionBlock({ title, children, definition, notation, id }) {
  return (
    <div
      id={id}
      className="my-6 rounded-xl border border-purple-200 bg-purple-50/60 dark:border-purple-500/30 dark:bg-purple-950/30"
    >
      {/* Header */}
      <div className="flex items-start gap-3 px-5 pt-4 pb-2">
        <span
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-purple-600 text-xs font-bold text-white"
          aria-hidden="true"
        >
          D
        </span>
        <h4 className="text-base font-semibold text-purple-900 dark:text-purple-200 leading-snug pt-0.5">
          {renderMathText(title)}
        </h4>
      </div>

      {/* Body */}
      <div className="px-5 pb-4 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        {definition ? (
          <p>{renderMathText(definition)}</p>
        ) : (
          children
        )}

        {/* Optional notation */}
        {notation && (
          <p className="mt-3 text-sm italic text-purple-700 dark:text-purple-400">
            <span className="font-medium">Notation: </span>
            {renderMathText(notation)}
          </p>
        )}
      </div>
    </div>
  )
}
