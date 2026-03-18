import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import DifficultyBadge from './DifficultyBadge.jsx'
import ProgressBar from './ProgressBar.jsx'

/**
 * Chapter card for subject pages.
 *
 * Props:
 *   chapter           {object}  Chapter object
 *   subjectId         {string}
 *   subjectColorHex   {string}
 *   completedCount    {number}
 *   totalCount        {number}
 *   chapterIndex      {number}  1-based index
 */
export default function ChapterCard({
  chapter,
  subjectId,
  subjectColorHex = '#6366f1',
  completedCount = 0,
  totalCount = 0,
  chapterIndex = 1,
}) {
  const progressValue = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0
  const sectionCount = chapter.sections?.length || 0
  const estimatedHours = chapter.estimatedMinutes
    ? `~${Math.ceil(chapter.estimatedMinutes / 60)}h`
    : null

  return (
    <motion.div
      whileHover={{ scale: 1.015, y: -2 }}
      transition={{ type: 'spring', stiffness: 300, damping: 28 }}
    >
      <Link
        to={`/subjects/${subjectId}/${chapter.id}`}
        className="group flex flex-col gap-3 rounded-xl border border-gray-200 bg-white p-5 shadow-sm transition-shadow hover:shadow-md dark:border-gray-800 dark:bg-gray-900 dark:hover:shadow-gray-900/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        aria-label={`Chapter ${chapterIndex}: ${chapter.title}`}
      >
        {/* Header row */}
        <div className="flex items-start gap-3">
          {/* Chapter number badge */}
          <div
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-sm font-bold text-white"
            style={{ backgroundColor: subjectColorHex }}
            aria-hidden="true"
          >
            {chapterIndex}
          </div>
          <div className="min-w-0 flex-1">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 leading-snug group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors line-clamp-2">
              {chapter.title}
            </h3>
          </div>
        </div>

        {/* Description */}
        {chapter.description && (
          <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-2">
            {chapter.description}
          </p>
        )}

        {/* Meta row */}
        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
          <span>
            <span className="font-medium text-gray-700 dark:text-gray-300">{sectionCount}</span>
            {' '}{sectionCount === 1 ? 'section' : 'sections'}
          </span>
          {estimatedHours && (
            <>
              <span aria-hidden="true">·</span>
              <span>{estimatedHours}</span>
            </>
          )}
          <span aria-hidden="true">·</span>
          <DifficultyBadge level={chapter.difficulty} size="sm" />
        </div>

        {/* Progress */}
        {totalCount > 0 && (
          <div className="space-y-1">
            <ProgressBar
              value={progressValue}
              color={subjectColorHex}
              showPercent={false}
              size="sm"
            />
            <p className="text-xs text-gray-400 dark:text-gray-500 tabular-nums">
              {completedCount} / {totalCount} done
            </p>
          </div>
        )}
      </Link>
    </motion.div>
  )
}
