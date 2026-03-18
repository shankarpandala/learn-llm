import { useParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { getCurriculumById, getChapterById } from '../subjects/index.js'
import DifficultyBadge from '../components/navigation/DifficultyBadge.jsx'
import ProgressBar from '../components/navigation/ProgressBar.jsx'
import Breadcrumbs from '../components/layout/Breadcrumbs.jsx'
import useProgress from '../hooks/useProgress.js'

function CheckCircleIcon({ filled }) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={filled ? 'text-emerald-500 dark:text-emerald-400' : 'text-gray-300 dark:text-gray-600'}
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" />
      {filled && <polyline points="9 12 11 14 15 10" strokeWidth="2.5" stroke="white" fill="none" />}
    </svg>
  )
}

function ClockIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  )
}

export default function ChapterPage() {
  const { subjectId, chapterId } = useParams()
  const subject = getCurriculumById(subjectId)
  const chapter = getChapterById(subjectId, chapterId)
  const { isComplete } = useProgress()

  if (!subject || !chapter) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 px-6 text-center">
        <div className="text-5xl" aria-hidden="true">∅</div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Chapter Not Found</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Could not find chapter "{chapterId}" in subject "{subjectId}".
        </p>
        <Link
          to={subject ? `/subjects/${subjectId}` : '/'}
          className="rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white hover:bg-emerald-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        >
          {subject ? 'Back to Subject' : 'Back to Home'}
        </Link>
      </div>
    )
  }

  const chapterIndex = subject.chapters?.findIndex((c) => c.id === chapterId) + 1
  const sections = chapter.sections || []
  const completedCount = sections.filter((sec) => isComplete(subjectId, chapterId, sec.id)).length
  const progressValue = sections.length > 0 ? Math.round((completedCount / sections.length) * 100) : 0
  const estimatedHours = chapter.estimatedMinutes
    ? `~${Math.ceil(chapter.estimatedMinutes / 60)}h`
    : null

  const breadcrumbs = [
    { label: 'Home', href: '/' },
    { label: subject.title, href: `/subjects/${subjectId}` },
    { label: chapter.title },
  ]

  return (
    <div className="min-h-screen">
      {/* Chapter Header */}
      <div
        className="relative border-b border-gray-200 dark:border-gray-800"
        style={{ background: `linear-gradient(135deg, ${subject.colorHex}12 0%, transparent 60%)` }}
      >
        <div
          className="absolute left-0 top-0 h-full w-1.5"
          style={{ backgroundColor: subject.colorHex }}
          aria-hidden="true"
        />

        <div className="mx-auto max-w-4xl px-6 py-10 pl-10">
          <Breadcrumbs items={breadcrumbs} />

          <div className="mt-4 flex items-start gap-4">
            {/* Chapter number badge */}
            <div
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl text-xl font-extrabold text-white shadow-sm"
              style={{ backgroundColor: subject.colorHex }}
              aria-hidden="true"
            >
              {chapterIndex}
            </div>

            <div className="min-w-0">
              <motion.h1
                className="text-2xl font-extrabold text-gray-900 dark:text-white sm:text-3xl"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                {chapter.title}
              </motion.h1>

              {chapter.description && (
                <p className="mt-2 text-gray-600 dark:text-gray-400 leading-relaxed">
                  {chapter.description}
                </p>
              )}

              <div className="mt-3 flex flex-wrap items-center gap-3">
                <DifficultyBadge level={chapter.difficulty} />
                {estimatedHours && (
                  <span className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400">
                    <ClockIcon />
                    {estimatedHours}
                  </span>
                )}
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {sections.length} sections
                </span>
              </div>

              {sections.length > 0 && (
                <div className="mt-4 max-w-xs">
                  <ProgressBar
                    value={progressValue}
                    color={subject.colorHex}
                    label={`${completedCount}/${sections.length} complete`}
                    showPercent
                    size="sm"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Section List */}
      <div className="mx-auto max-w-4xl px-6 py-10">
        <h2 className="mb-6 text-lg font-bold text-gray-900 dark:text-white">
          Sections
        </h2>

        <ol className="space-y-3" aria-label="Chapter sections">
          {sections.map((section, idx) => {
            const done = isComplete(subjectId, chapterId, section.id)
            const to = `/subjects/${subjectId}/${chapterId}/${section.id}`

            return (
              <motion.li
                key={section.id}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05, duration: 0.3 }}
              >
                <Link
                  to={to}
                  className={`group flex items-start gap-4 rounded-xl border p-4 transition-all hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
                    done
                      ? 'border-emerald-200 bg-emerald-50/50 dark:border-emerald-900/40 dark:bg-emerald-950/20'
                      : 'border-gray-200 bg-white hover:border-emerald-200 hover:bg-emerald-50/30 dark:border-gray-800 dark:bg-gray-900 dark:hover:border-emerald-800/50 dark:hover:bg-emerald-950/10'
                  }`}
                >
                  {/* Section number + completion */}
                  <div className="flex shrink-0 flex-col items-center gap-1 pt-0.5">
                    <span
                      className="flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold"
                      style={{
                        backgroundColor: done ? '#10b981' : `${subject.colorHex}22`,
                        color: done ? 'white' : subject.colorHex,
                      }}
                    >
                      {idx + 1}
                    </span>
                  </div>

                  {/* Content */}
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <h3
                        className={`font-semibold leading-snug group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors ${
                          done ? 'text-gray-700 dark:text-gray-300' : 'text-gray-900 dark:text-gray-100'
                        }`}
                      >
                        {section.title}
                      </h3>
                      <div className="flex shrink-0 items-center gap-2">
                        <DifficultyBadge level={section.difficulty} size="sm" />
                        {section.readingMinutes && (
                          <span className="flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500">
                            <ClockIcon />
                            {section.readingMinutes}m
                          </span>
                        )}
                        <CheckCircleIcon filled={done} />
                      </div>
                    </div>

                    {section.description && (
                      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
                        {section.description}
                      </p>
                    )}
                  </div>
                </Link>
              </motion.li>
            )
          })}
        </ol>
      </div>
    </div>
  )
}
