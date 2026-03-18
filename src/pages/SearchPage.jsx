import { useState, useMemo, useEffect, useCallback } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import CURRICULUM from '../subjects/index.js'

function SearchIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-400" aria-hidden="true">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  )
}

function ClearIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </svg>
  )
}

// Build a flat searchable index from CURRICULUM
function buildSearchIndex() {
  const items = []
  for (const subject of CURRICULUM) {
    // Subject itself
    items.push({
      type: 'subject',
      subjectId: subject.id,
      chapterId: null,
      sectionId: null,
      title: subject.title,
      description: subject.description || '',
      href: `/subjects/${subject.id}`,
      breadcrumbs: [subject.title],
      subject,
    })
    for (const chapter of subject.chapters || []) {
      // Chapter
      items.push({
        type: 'chapter',
        subjectId: subject.id,
        chapterId: chapter.id,
        sectionId: null,
        title: chapter.title,
        description: chapter.description || '',
        href: `/subjects/${subject.id}/${chapter.id}`,
        breadcrumbs: [subject.title, chapter.title],
        subject,
        chapter,
      })
      for (const section of chapter.sections || []) {
        // Section
        items.push({
          type: 'section',
          subjectId: subject.id,
          chapterId: chapter.id,
          sectionId: section.id,
          title: section.title,
          description: section.description || '',
          href: `/subjects/${subject.id}/${chapter.id}/${section.id}`,
          breadcrumbs: [subject.title, chapter.title, section.title],
          subject,
          chapter,
          section,
        })
      }
    }
  }
  return items
}

const SEARCH_INDEX = buildSearchIndex()

function normalize(str) {
  return str.toLowerCase().trim()
}

function search(query) {
  if (!query || query.trim().length < 2) return []
  const q = normalize(query)
  const scored = []
  for (const item of SEARCH_INDEX) {
    const titleN = normalize(item.title)
    const descN = normalize(item.description)

    let score = 0
    if (titleN === q) score += 100
    else if (titleN.startsWith(q)) score += 80
    else if (titleN.includes(q)) score += 50
    if (descN.includes(q)) score += 20
    if (item.type === 'section') score += 5

    if (score > 0) {
      scored.push({ ...item, score })
    }
  }
  return scored.sort((a, b) => b.score - a.score)
}

function groupBySubject(results) {
  const map = new Map()
  for (const result of results) {
    const key = result.subject.id
    if (!map.has(key)) {
      map.set(key, { subject: result.subject, items: [] })
    }
    map.get(key).items.push(result)
  }
  return Array.from(map.values())
}

function Highlight({ text, query }) {
  if (!query || query.trim().length < 2) {
    return <span>{text}</span>
  }
  const q = query.trim()
  const regex = new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  const parts = text.split(regex)
  return (
    <span>
      {parts.map((part, idx) =>
        regex.test(part) ? (
          <mark key={idx} className="bg-yellow-200 text-yellow-900 dark:bg-yellow-700/50 dark:text-yellow-200 rounded-sm px-0.5">
            {part}
          </mark>
        ) : (
          <span key={idx}>{part}</span>
        )
      )}
    </span>
  )
}

const TYPE_BADGE = {
  subject: { label: 'Subject', className: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' },
  chapter: { label: 'Chapter', className: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' },
  section: { label: 'Section', className: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' },
}

export default function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const urlQuery = searchParams.get('q') || ''
  const [inputValue, setInputValue] = useState(urlQuery)

  // Sync URL → input on mount / URL change
  useEffect(() => {
    setInputValue(urlQuery)
  }, [urlQuery])

  // Debounce URL update
  useEffect(() => {
    const timer = setTimeout(() => {
      const trimmed = inputValue.trim()
      if (trimmed !== urlQuery) {
        if (trimmed) {
          setSearchParams({ q: trimmed }, { replace: true })
        } else {
          setSearchParams({}, { replace: true })
        }
      }
    }, 300)
    return () => clearTimeout(timer)
  }, [inputValue, urlQuery, setSearchParams])

  const results = useMemo(() => search(urlQuery), [urlQuery])
  const grouped = useMemo(() => groupBySubject(results), [results])

  const handleClear = useCallback(() => {
    setInputValue('')
    setSearchParams({}, { replace: true })
  }, [setSearchParams])

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950/50">
      <div className="mx-auto max-w-3xl px-6 py-10">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <h1 className="text-2xl font-extrabold text-gray-900 dark:text-white sm:text-3xl">
            Search
          </h1>
          <p className="mt-1 text-gray-500 dark:text-gray-400">
            Search across subjects, chapters, and sections.
          </p>
        </motion.div>

        {/* Search input */}
        <div className="relative mt-6">
          <label htmlFor="search-input" className="sr-only">Search LearnLLM</label>
          <div className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2">
            <SearchIcon />
          </div>
          <input
            id="search-input"
            type="search"
            autoFocus
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Search topics, models, techniques…"
            className="w-full rounded-xl border border-gray-300 bg-white py-3.5 pl-11 pr-11 text-base text-gray-900 shadow-sm placeholder:text-gray-400 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/40 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100 dark:placeholder:text-gray-500 dark:focus:border-emerald-500"
          />
          {inputValue && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded"
              aria-label="Clear search"
            >
              <ClearIcon />
            </button>
          )}
        </div>

        {/* Results count */}
        {urlQuery.trim().length >= 2 && (
          <p className="mt-3 text-sm text-gray-500 dark:text-gray-400">
            {results.length === 0
              ? `No results for "${urlQuery}"`
              : `${results.length} result${results.length !== 1 ? 's' : ''} for "${urlQuery}"`}
          </p>
        )}

        {/* Empty state */}
        {urlQuery.trim().length < 2 && (
          <div className="mt-16 text-center text-gray-400 dark:text-gray-600">
            <div className="text-6xl mb-4 select-none" aria-hidden="true">🧠</div>
            <p className="text-sm">Start typing to search across 15 subjects, 70+ chapters, and 280+ sections.</p>
          </div>
        )}

        {/* No results */}
        {urlQuery.trim().length >= 2 && results.length === 0 && (
          <motion.div
            className="mt-12 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="text-4xl mb-4 select-none" aria-hidden="true">∅</div>
            <p className="font-medium text-gray-700 dark:text-gray-300">No results found</p>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Try different keywords or browse by{' '}
              <Link to="/#subjects" className="text-emerald-600 hover:underline dark:text-emerald-400">
                subject
              </Link>.
            </p>
          </motion.div>
        )}

        {/* Grouped results */}
        <AnimatePresence mode="wait">
          {grouped.length > 0 && (
            <motion.div
              key={urlQuery}
              className="mt-6 space-y-8"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
            >
              {grouped.map(({ subject, items }) => (
                <div key={subject.id}>
                  {/* Subject group header */}
                  <div className="flex items-center gap-2 mb-3">
                    <span
                      className="h-3 w-3 rounded-full shrink-0"
                      style={{ backgroundColor: subject.colorHex }}
                      aria-hidden="true"
                    />
                    <h2 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wide">
                      {subject.title}
                    </h2>
                    <span className="text-xs text-gray-400 dark:text-gray-500">
                      {items.length} result{items.length !== 1 ? 's' : ''}
                    </span>
                  </div>

                  {/* Result items */}
                  <ul className="space-y-2">
                    {items.map((item, idx) => {
                      const badge = TYPE_BADGE[item.type] || TYPE_BADGE.section
                      return (
                        <motion.li
                          key={`${item.subjectId}::${item.chapterId}::${item.sectionId}`}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.04, duration: 0.25 }}
                        >
                          <Link
                            to={item.href}
                            className="group flex flex-col gap-1 rounded-xl border border-gray-200 bg-white p-4 hover:border-emerald-300 hover:shadow-sm transition-all dark:border-gray-800 dark:bg-gray-900 dark:hover:border-emerald-700/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                          >
                            {/* Breadcrumb path */}
                            <div className="flex flex-wrap items-center gap-1 text-xs text-gray-400 dark:text-gray-500">
                              {item.breadcrumbs.slice(0, -1).map((crumb, i) => (
                                <span key={i} className="flex items-center gap-1">
                                  {i > 0 && <span aria-hidden="true">›</span>}
                                  <span>{crumb}</span>
                                </span>
                              ))}
                            </div>

                            {/* Title row */}
                            <div className="flex items-start justify-between gap-3">
                              <h3 className="font-semibold text-gray-900 dark:text-gray-100 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors leading-snug">
                                <Highlight text={item.title} query={urlQuery} />
                              </h3>
                              <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${badge.className}`}>
                                {badge.label}
                              </span>
                            </div>

                            {/* Description */}
                            {item.description && (
                              <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-2">
                                <Highlight text={item.description} query={urlQuery} />
                              </p>
                            )}
                          </Link>
                        </motion.li>
                      )
                    })}
                  </ul>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
