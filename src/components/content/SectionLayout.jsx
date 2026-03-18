import { useState } from 'react'
import { motion } from 'framer-motion'
import DifficultyBadge from '../navigation/DifficultyBadge'
import PrevNextNav from '../navigation/PrevNextNav'
import useProgress from '../../hooks/useProgress'

/**
 * Section page wrapper with metadata header and navigation.
 *
 * Props:
 *   title          {string}   Section title
 *   subjectId      {string}   Subject identifier
 *   chapterId      {string}   Chapter identifier
 *   sectionId      {string}   Section identifier
 *   difficulty     {string}   'beginner' | 'intermediate' | 'advanced' | 'research'
 *   readingTime    {number}   Estimated reading time in minutes
 *   prerequisites  {array}    Array of { title, subjectId?, chapterId?, sectionId? }
 *   prev           {object}   PrevNextNav prev prop
 *   next           {object}   PrevNextNav next prop
 *   children       {node}     Section content
 *   id             {string}   Optional anchor id
 */
export default function SectionLayout({
  title,
  subjectId,
  chapterId,
  sectionId,
  difficulty,
  readingTime,
  prerequisites,
  prev,
  next,
  children,
  id,
}) {
  const { markComplete, isComplete } = useProgress()
  const sectionKey = `${subjectId}/${chapterId}/${sectionId}`
  const completed = isComplete(sectionKey)
  const [justCompleted, setJustCompleted] = useState(false)

  const handleMarkComplete = () => {
    markComplete(sectionKey)
    setJustCompleted(true)
    setTimeout(() => setJustCompleted(false), 2000)
  }

  return (
    <article id={id} className="mx-auto max-w-3xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 sm:text-3xl leading-tight">
          {title}
        </h1>

        {/* Metadata row */}
        <div className="mt-4 flex flex-wrap items-center gap-3">
          {difficulty && <DifficultyBadge level={difficulty} size="sm" />}

          {readingTime && (
            <span className="inline-flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              {readingTime} min read
            </span>
          )}
        </div>

        {/* Prerequisites */}
        {prerequisites && prerequisites.length > 0 && (
          <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50/50 px-4 py-3 dark:border-amber-500/20 dark:bg-amber-950/20">
            <p className="text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-400 mb-1.5">
              Prerequisites
            </p>
            <ul className="flex flex-wrap gap-2">
              {prerequisites.map((prereq, i) => (
                <li key={i}>
                  {prereq.subjectId ? (
                    <a
                      href={`/subjects/${prereq.subjectId}/${prereq.chapterId || ''}/${prereq.sectionId || ''}`}
                      className="inline-flex items-center rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800 hover:bg-amber-200 dark:bg-amber-500/20 dark:text-amber-300 dark:hover:bg-amber-500/30 transition-colors"
                    >
                      {prereq.title}
                    </a>
                  ) : (
                    <span className="inline-flex items-center rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-500/20 dark:text-amber-300">
                      {prereq.title}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </header>

      {/* Section content */}
      <div className="prose-sm sm:prose dark:prose-invert max-w-none">
        {children}
      </div>

      {/* Mark complete button */}
      <div className="mt-12 flex justify-center">
        <button
          onClick={handleMarkComplete}
          disabled={completed && !justCompleted}
          className={`inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
            completed
              ? 'border border-green-300 bg-green-50 text-green-700 dark:border-green-700/50 dark:bg-green-900/20 dark:text-green-400 cursor-default'
              : 'bg-indigo-600 text-white shadow-md hover:bg-indigo-700 hover:shadow-lg active:scale-[0.98] dark:bg-indigo-500 dark:hover:bg-indigo-600'
          }`}
        >
          {completed ? (
            <>
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              {justCompleted ? 'Marked Complete!' : 'Section Complete'}
            </>
          ) : (
            <>
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              Mark Section Complete
            </>
          )}
        </button>
      </div>

      {/* Prev / Next navigation */}
      <PrevNextNav prev={prev} next={next} />
    </article>
  )
}
