import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const TYPE_CONFIG = {
  foundational: {
    label: 'Foundational Papers',
    color: 'text-red-600 dark:text-red-400',
    badgeBg: 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-400',
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
  textbook: {
    label: 'Textbooks',
    color: 'text-blue-600 dark:text-blue-400',
    badgeBg: 'bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-400',
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
      </svg>
    ),
  },
  survey: {
    label: 'Surveys & Reviews',
    color: 'text-purple-600 dark:text-purple-400',
    badgeBg: 'bg-purple-100 text-purple-700 dark:bg-purple-500/20 dark:text-purple-400',
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
  },
  tutorial: {
    label: 'Tutorials & Guides',
    color: 'text-emerald-600 dark:text-emerald-400',
    badgeBg: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400',
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
      </svg>
    ),
  },
}

const TYPE_ORDER = ['foundational', 'textbook', 'survey', 'tutorial']

function ReferenceItem({ reference }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <li className="py-3 first:pt-0 last:pb-0">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          {/* Title + URL */}
          <div className="flex flex-wrap items-baseline gap-x-2">
            {reference.url ? (
              <a
                href={reference.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-medium text-gray-900 dark:text-gray-100 hover:text-indigo-600 dark:hover:text-indigo-400 underline decoration-gray-300 dark:decoration-gray-600 underline-offset-2 transition-colors"
              >
                {reference.title}
              </a>
            ) : (
              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {reference.title}
              </span>
            )}
            {reference.year && (
              <span className="text-xs text-gray-400 dark:text-gray-500">
                ({reference.year})
              </span>
            )}
          </div>

          {/* Authors */}
          {reference.authors && (
            <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
              {reference.authors}
            </p>
          )}

          {/* Expandable "why important" */}
          {reference.whyImportant && (
            <div className="mt-1.5">
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline"
                aria-expanded={expanded}
              >
                <svg
                  className={`h-3 w-3 shrink-0 transition-transform ${expanded ? 'rotate-90' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
                {expanded ? 'Hide importance' : 'Why important?'}
              </button>

              <AnimatePresence>
                {expanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <p className="mt-1.5 rounded-lg bg-gray-50 p-2.5 text-xs leading-relaxed text-gray-600 dark:bg-gray-800/60 dark:text-gray-400">
                      {reference.whyImportant}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>
      </div>
    </li>
  )
}

/**
 * Grouped reference list by type.
 *
 * Props:
 *   references {array}  Array of:
 *     {
 *       type:          string — 'foundational' | 'textbook' | 'survey' | 'tutorial'
 *       title:         string — reference title
 *       authors:       string — author names
 *       year:          string | number — publication year
 *       url:           string — link to resource
 *       whyImportant:  string — expandable description
 *     }
 *   title      {string}  Section title (default "References & Further Reading")
 *   id         {string}  Optional anchor id
 */
export default function ReferenceList({
  references = [],
  title = 'References & Further Reading',
  id,
}) {
  // Group by type
  const grouped = {}
  for (const ref of references) {
    const type = ref.type || 'tutorial'
    if (!grouped[type]) grouped[type] = []
    grouped[type].push(ref)
  }

  // Render in defined order
  const orderedGroups = TYPE_ORDER.filter((t) => grouped[t] && grouped[t].length > 0)

  return (
    <div
      id={id}
      className="my-8 rounded-xl border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-900/60"
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
            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
          />
        </svg>
        <h4 className="text-base font-semibold text-gray-900 dark:text-gray-100">
          {title}
        </h4>
        <span className="ml-auto rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-600 dark:bg-gray-800 dark:text-gray-400">
          {references.length} {references.length === 1 ? 'reference' : 'references'}
        </span>
      </div>

      {/* Grouped references */}
      <div className="divide-y divide-gray-100 dark:divide-gray-800">
        {orderedGroups.map((type) => {
          const config = TYPE_CONFIG[type]
          const refs = grouped[type]
          return (
            <div key={type} className="px-5 py-4">
              {/* Group label */}
              <div className="mb-3 flex items-center gap-2">
                <span className={config.color}>{config.icon}</span>
                <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${config.badgeBg}`}>
                  {config.label}
                </span>
                <span className="text-xs text-gray-400 dark:text-gray-500">
                  ({refs.length})
                </span>
              </div>

              {/* Reference items */}
              <ul className="divide-y divide-gray-100 dark:divide-gray-800/60 pl-6">
                {refs.map((ref, i) => (
                  <ReferenceItem key={i} reference={ref} />
                ))}
              </ul>
            </div>
          )
        })}
      </div>
    </div>
  )
}
