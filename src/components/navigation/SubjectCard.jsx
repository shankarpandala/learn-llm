import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import DifficultyBadge from './DifficultyBadge.jsx'
import ProgressBar from './ProgressBar.jsx'

/**
 * Subject card for the home page grid.
 *
 * Props:
 *   subject         {object}  Subject object from CURRICULUM
 *   completedCount  {number}
 *   totalCount      {number}
 */
export default function SubjectCard({ subject, completedCount = 0, totalCount = 0 }) {
  const progressValue = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0
  const chapterCount = subject.chapters?.length || 0

  return (
    <motion.div
      whileHover={{ scale: 1.02, y: -2 }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      className="h-full"
    >
      <Link
        to={`/subjects/${subject.id}`}
        className="group flex h-full flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm transition-shadow hover:shadow-md dark:border-gray-800 dark:bg-gray-900 dark:hover:shadow-gray-900/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        aria-label={`${subject.title} — ${progressValue}% complete`}
      >
        {/* Top colored strip */}
        <div
          className="h-1.5 w-full shrink-0"
          style={{ backgroundColor: subject.colorHex }}
          aria-hidden="true"
        />

        <div className="flex flex-1 flex-col p-5 gap-4">
          {/* Icon + title row */}
          <div className="flex items-start gap-3">
            <div
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-2xl font-bold"
              style={{
                backgroundColor: `${subject.colorHex}22`,
                color: subject.colorHex,
              }}
              aria-hidden="true"
            >
              {subject.icon}
            </div>
            <div className="min-w-0">
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 leading-snug line-clamp-2 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                {subject.title}
              </h3>
            </div>
          </div>

          {/* Description */}
          <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-3 flex-1">
            {subject.description}
          </p>

          {/* Stats row */}
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <span className="flex items-center gap-1">
              <span className="font-medium text-gray-700 dark:text-gray-300">{chapterCount}</span>
              {chapterCount === 1 ? 'chapter' : 'chapters'}
            </span>
            <span aria-hidden="true">·</span>
            <span className="flex items-center gap-1">
              <span className="font-medium text-gray-700 dark:text-gray-300">
                ~{subject.estimatedHours}h
              </span>
            </span>
            <span aria-hidden="true">·</span>
            <DifficultyBadge level={subject.difficulty} size="sm" />
          </div>

          {/* Progress */}
          {totalCount > 0 && (
            <div className="space-y-1">
              <ProgressBar
                value={progressValue}
                color={subject.colorHex}
                showPercent={false}
                size="sm"
              />
              <p className="text-xs text-gray-400 dark:text-gray-500 tabular-nums">
                {completedCount} / {totalCount} sections complete
              </p>
            </div>
          )}
        </div>
      </Link>
    </motion.div>
  )
}
