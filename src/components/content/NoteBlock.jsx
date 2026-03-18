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

const NOTE_TYPES = {
  note: {
    border: 'border-blue-200 dark:border-blue-500/30',
    bg: 'bg-blue-50/60 dark:bg-blue-950/30',
    iconColor: 'text-blue-500 dark:text-blue-400',
    titleColor: 'text-blue-900 dark:text-blue-200',
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    ),
  },
  historical: {
    border: 'border-amber-200 dark:border-amber-500/30',
    bg: 'bg-amber-50/60 dark:bg-amber-950/30',
    iconColor: 'text-amber-500 dark:text-amber-400',
    titleColor: 'text-amber-900 dark:text-amber-200',
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    ),
  },
  intuition: {
    border: 'border-violet-200 dark:border-violet-500/30',
    bg: 'bg-violet-50/60 dark:bg-violet-950/30',
    iconColor: 'text-violet-500 dark:text-violet-400',
    titleColor: 'text-violet-900 dark:text-violet-200',
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
        />
      </svg>
    ),
  },
  tip: {
    border: 'border-teal-200 dark:border-teal-500/30',
    bg: 'bg-teal-50/60 dark:bg-teal-950/30',
    iconColor: 'text-teal-500 dark:text-teal-400',
    titleColor: 'text-teal-900 dark:text-teal-200',
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M5 13l4 4L19 7"
        />
      </svg>
    ),
  },
}

/**
 * Multi-type note callout block.
 *
 * Props:
 *   type     {string}  'note' | 'historical' | 'intuition' | 'tip'
 *   title    {string}  Note title
 *   content  {string}  Plain text content with $math$ support
 *   children {node}    Optional JSX body (used if content not provided)
 *   id       {string}  Optional anchor id
 */
export default function NoteBlock({ type = 'note', title, content, children, id }) {
  const config = NOTE_TYPES[type] || NOTE_TYPES.note

  return (
    <div
      id={id}
      className={`my-6 rounded-xl border ${config.border} ${config.bg}`}
    >
      <div className="flex items-start gap-3 px-5 py-4">
        {/* Icon */}
        <span className={`shrink-0 mt-0.5 ${config.iconColor}`} aria-hidden="true">
          {config.icon}
        </span>

        <div className="min-w-0 flex-1">
          {/* Title */}
          {title && (
            <h4 className={`text-sm font-semibold ${config.titleColor} mb-1`}>
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
