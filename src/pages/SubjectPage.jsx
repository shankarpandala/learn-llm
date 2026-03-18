import { useParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { getCurriculumById, getSubjectSectionCount } from '../subjects/index.js'
import ChapterCard from '../components/navigation/ChapterCard.jsx'
import DifficultyBadge from '../components/navigation/DifficultyBadge.jsx'
import ProgressBar from '../components/navigation/ProgressBar.jsx'
import Breadcrumbs from '../components/layout/Breadcrumbs.jsx'
import useProgress from '../hooks/useProgress.js'

export default function SubjectPage() {
  const { subjectId } = useParams()
  const subject = getCurriculumById(subjectId)
  const { isComplete } = useProgress()

  if (!subject) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 px-6 text-center">
        <div className="text-5xl" aria-hidden="true">∅</div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Subject Not Found</h1>
        <p className="text-gray-500 dark:text-gray-400">No subject with ID "{subjectId}" exists.</p>
        <Link
          to="/"
          className="rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white hover:bg-emerald-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        >
          Back to Home
        </Link>
      </div>
    )
  }

  const totalSections = getSubjectSectionCount(subjectId)
  const completedSections = subject.chapters?.reduce((acc, ch) =>
    acc + (ch.sections?.filter((sec) => isComplete(subjectId, ch.id, sec.id)).length || 0), 0
  ) || 0
  const progressValue = totalSections > 0 ? Math.round((completedSections / totalSections) * 100) : 0

  const breadcrumbs = [
    { label: 'Home', href: '/' },
    { label: subject.title },
  ]

  return (
    <div className="min-h-screen">
      {/* Subject Header */}
      <div
        className="relative overflow-hidden border-b border-gray-200 dark:border-gray-800"
        style={{ background: `linear-gradient(135deg, ${subject.colorHex}18 0%, transparent 60%)` }}
      >
        {/* Left border accent */}
        <div
          className="absolute left-0 top-0 h-full w-1.5"
          style={{ backgroundColor: subject.colorHex }}
          aria-hidden="true"
        />

        <div className="mx-auto max-w-5xl px-6 py-10 pl-10">
          <Breadcrumbs items={breadcrumbs} />

          <div className="mt-4 flex items-start gap-5">
            {/* Subject icon */}
            <div
              className="hidden sm:flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl text-3xl font-bold shadow-sm"
              style={{ backgroundColor: `${subject.colorHex}22`, color: subject.colorHex }}
              aria-hidden="true"
            >
              {subject.icon}
            </div>

            <div className="min-w-0 flex-1">
              <motion.h1
                className="text-2xl font-extrabold text-gray-900 dark:text-white sm:text-3xl"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                {subject.title}
              </motion.h1>

              <p className="mt-2 text-gray-600 dark:text-gray-400 leading-relaxed max-w-2xl">
                {subject.description}
              </p>

              {/* Meta row */}
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <DifficultyBadge level={subject.difficulty} />
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  ~{subject.estimatedHours}h estimated
                </span>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {subject.chapters?.length || 0} chapters · {totalSections} sections
                </span>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  LLM Relevance:{' '}
                  <span className="font-semibold text-gray-700 dark:text-gray-300">
                    {subject.mlRelevance}%
                  </span>
                </span>
              </div>

              {/* Prerequisites */}
              {subject.prerequisites?.length > 0 && (
                <div className="mt-4 flex flex-wrap items-center gap-2">
                  <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Prerequisites:
                  </span>
                  {subject.prerequisites.map((prereqId) => {
                    const prereq = getCurriculumById(prereqId)
                    return prereq ? (
                      <Link
                        key={prereqId}
                        to={`/subjects/${prereqId}`}
                        className="rounded-full border border-gray-200 bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-700 hover:border-emerald-300 hover:text-emerald-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300 dark:hover:border-emerald-600 dark:hover:text-emerald-400 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                      >
                        {prereq.title}
                      </Link>
                    ) : (
                      <span
                        key={prereqId}
                        className="rounded-full border border-gray-200 bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400"
                      >
                        {prereqId}
                      </span>
                    )
                  })}
                </div>
              )}

              {/* LLM Relevance bar */}
              <div className="mt-5 max-w-xs">
                <ProgressBar
                  value={subject.mlRelevance}
                  color={subject.colorHex}
                  label="LLM Relevance"
                  showPercent
                  size="sm"
                />
              </div>
            </div>
          </div>

          {/* Overall progress */}
          {totalSections > 0 && (
            <div className="mt-6 max-w-sm">
              <ProgressBar
                value={progressValue}
                color={subject.colorHex}
                label={`Progress (${completedSections}/${totalSections} sections)`}
                showPercent
                size="md"
              />
            </div>
          )}
        </div>
      </div>

      {/* Chapter Grid */}
      <div className="mx-auto max-w-5xl px-6 py-10">
        <h2 className="mb-6 text-xl font-bold text-gray-900 dark:text-white">
          Chapters
        </h2>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {subject.chapters?.map((chapter, idx) => {
            const chTotal = chapter.sections?.length || 0
            const chCompleted = chapter.sections?.filter((sec) =>
              isComplete(subjectId, chapter.id, sec.id)
            ).length || 0

            return (
              <motion.div
                key={chapter.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.06, duration: 0.35 }}
              >
                <ChapterCard
                  chapter={chapter}
                  subjectId={subjectId}
                  subjectColorHex={subject.colorHex}
                  completedCount={chCompleted}
                  totalCount={chTotal}
                  chapterIndex={idx + 1}
                />
              </motion.div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
