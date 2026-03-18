import { InlineMath } from 'react-katex'

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
 * Amber-themed warning callout with triangle icon.
 *
 * Props:
 *   title    {string}  Warning title (default "Warning")
 *   content  {string}  Warning text with $math$ support
 *   children {node}    Optional JSX body (used if content not provided)
 *   id       {string}  Optional anchor id
 */
export default function WarningBlock({ title = 'Warning', content, children, id }) {
  return (
    <div
      id={id}
      className="my-6 rounded-xl border border-amber-300 bg-amber-50/70 dark:border-amber-500/30 dark:bg-amber-950/30"
    >
      <div className="flex items-start gap-3 px-5 py-4">
        {/* Triangle warning icon */}
        <span className="shrink-0 mt-0.5 text-amber-500 dark:text-amber-400" aria-hidden="true">
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        </span>

        <div className="min-w-0 flex-1">
          {/* Title */}
          {title && (
            <h4 className="text-sm font-semibold text-amber-900 dark:text-amber-200 mb-1">
              {renderMathText(title)}
            </h4>
          )}

          {/* Content */}
          <div className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            {content ? <p>{renderMathText(content)}</p> : children}
          </div>
        </div>
      </div>
    </div>
  )
}
