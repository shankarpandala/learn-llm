import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import CURRICULUM, { getSubjectSectionCount } from '../subjects/index.js'
import useProgress from '../hooks/useProgress.js'
import ProgressBar from '../components/navigation/ProgressBar.jsx'
import useAppStore from '../store/appStore.js'

const TOTAL_SECTIONS = CURRICULUM.reduce((acc, subject) =>
  acc + getSubjectSectionCount(subject.id), 0
)

function BookmarkIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-amber-500" aria-hidden="true">
      <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-500" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4h6v2" />
    </svg>
  )
}

export default function ProgressPage() {
  const { isComplete } = useProgress()
  const completedSections = useAppStore((s) => s.completedSections)
  const bookmarks = useAppStore((s) => s.bookmarks)
  const removeBookmark = useAppStore((s) => s.removeBookmark)

  const totalCompleted = completedSections.length
  const overallPercent = TOTAL_SECTIONS > 0
    ? Math.round((totalCompleted / TOTAL_SECTIONS) * 100)
    : 0

  // Build recent completed list (last 10, reversed)
  const recent = [...completedSections].reverse().slice(0, 10).map((key) => {
    const [subjectId, chapterId, sectionId] = key.split('::')
    const subject = CURRICULUM.find((s) => s.id === subjectId)
    const chapter = subject?.chapters?.find((c) => c.id === chapterId)
    const section = chapter?.sections?.find((s) => s.id === sectionId)
    return { subjectId, chapterId, sectionId, subject, chapter, section }
  }).filter((item) => item.section)

  // Per-subject progress
  const subjectProgress = CURRICULUM.map((subject) => {
    const total = getSubjectSectionCount(subject.id)
    const completed = subject.chapters?.reduce((acc, ch) =>
      acc + (ch.sections?.filter((sec) => isComplete(subject.id, ch.id, sec.id)).length || 0), 0
    ) || 0
    return { subject, completed, total, percent: total > 0 ? Math.round((completed / total) * 100) : 0 }
  })

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950/50">
      <div className="mx-auto max-w-4xl px-6 py-10">
        <motion.h1
          className="text-2xl font-extrabold text-gray-900 dark:text-white sm:text-3xl"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          My Progress
        </motion.h1>
        <p className="mt-1 text-gray-500 dark:text-gray-400">
          Track your learning journey across all subjects.
        </p>

        {/* ── Overall stats ── */}
        <motion.div
          className="mt-8 rounded-2xl border border-gray-200 bg-white p-6 dark:border-gray-800 dark:bg-gray-900"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.4 }}
        >
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
            {/* Big number */}
            <div className="text-center sm:text-left">
              <div className="text-5xl font-extrabold text-emerald-600 dark:text-emerald-400 tabular-nums">
                {totalCompleted}
              </div>
              <div className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                of {TOTAL_SECTIONS} sections complete
              </div>
            </div>

            {/* Divider */}
            <div className="hidden sm:block h-16 w-px bg-gray-200 dark:bg-gray-700" aria-hidden="true" />

            {/* Overall progress bar */}
            <div className="flex-1 w-full">
              <ProgressBar
                value={overallPercent}
                color="#059669"
                label="Overall completion"
                showPercent
                size="lg"
              />
            </div>
          </div>

          {/* Stat row */}
          <div className="mt-6 grid grid-cols-3 gap-4 border-t border-gray-100 pt-5 dark:border-gray-800">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 dark:text-white tabular-nums">
                {CURRICULUM.length}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">Subjects</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 dark:text-white tabular-nums">
                {CURRICULUM.reduce((acc, s) => acc + (s.chapters?.length || 0), 0)}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">Chapters</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 dark:text-white tabular-nums">
                {bookmarks.length}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">Bookmarks</div>
            </div>
          </div>
        </motion.div>

        <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-5">
          {/* ── Per-subject progress (wider) ── */}
          <div className="lg:col-span-3">
            <h2 className="mb-4 text-lg font-bold text-gray-900 dark:text-white">
              By Subject
            </h2>
            <div className="space-y-3">
              {subjectProgress.map(({ subject, completed, total, percent }, idx) => (
                <motion.div
                  key={subject.id}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.04, duration: 0.3 }}
                >
                  <Link
                    to={`/subjects/${subject.id}`}
                    className="flex items-center gap-3 rounded-xl border border-gray-200 bg-white p-4 hover:shadow-sm transition-shadow dark:border-gray-800 dark:bg-gray-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                  >
                    <span
                      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-sm font-bold"
                      style={{ backgroundColor: `${subject.colorHex}22`, color: subject.colorHex }}
                      aria-hidden="true"
                    >
                      {subject.icon}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2 mb-1.5">
                        <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                          {subject.title}
                        </span>
                        <span className="shrink-0 text-xs text-gray-400 dark:text-gray-500 tabular-nums">
                          {completed}/{total}
                        </span>
                      </div>
                      <ProgressBar
                        value={percent}
                        color={subject.colorHex}
                        showPercent={false}
                        size="sm"
                      />
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>
          </div>

          {/* ── Right column: recent + bookmarks ── */}
          <div className="lg:col-span-2 space-y-8">
            {/* Recently completed */}
            <div>
              <h2 className="mb-4 text-lg font-bold text-gray-900 dark:text-white">
                Recently Completed
              </h2>
              {recent.length === 0 ? (
                <div className="rounded-xl border border-dashed border-gray-200 p-6 text-center text-sm text-gray-400 dark:border-gray-700 dark:text-gray-500">
                  No sections completed yet. Start learning!
                </div>
              ) : (
                <ul className="space-y-2">
                  {recent.map((item, idx) => (
                    <motion.li
                      key={`${item.subjectId}::${item.chapterId}::${item.sectionId}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.05, duration: 0.3 }}
                    >
                      <Link
                        to={`/subjects/${item.subjectId}/${item.chapterId}/${item.sectionId}`}
                        className="flex items-start gap-2.5 rounded-lg border border-gray-200 bg-white p-3 text-sm hover:border-emerald-300 hover:bg-emerald-50/30 transition-colors dark:border-gray-800 dark:bg-gray-900 dark:hover:border-emerald-800/50 dark:hover:bg-emerald-950/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                      >
                        <CheckIcon />
                        <div className="min-w-0">
                          <div className="font-medium text-gray-800 dark:text-gray-200 truncate">
                            {item.section?.title}
                          </div>
                          <div className="text-xs text-gray-400 dark:text-gray-500 truncate">
                            {item.subject?.title} › {item.chapter?.title}
                          </div>
                        </div>
                      </Link>
                    </motion.li>
                  ))}
                </ul>
              )}
            </div>

            {/* Bookmarks */}
            <div>
              <h2 className="mb-4 text-lg font-bold text-gray-900 dark:text-white">
                Bookmarks
              </h2>
              {bookmarks.length === 0 ? (
                <div className="rounded-xl border border-dashed border-gray-200 p-6 text-center text-sm text-gray-400 dark:border-gray-700 dark:text-gray-500">
                  No bookmarks yet. Bookmark sections while reading.
                </div>
              ) : (
                <ul className="space-y-2">
                  {bookmarks.map((bm, idx) => (
                    <motion.li
                      key={`${bm.subjectId}::${bm.chapterId}::${bm.sectionId}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.05, duration: 0.3 }}
                      className="flex items-start gap-2 rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900"
                    >
                      <Link
                        to={`/subjects/${bm.subjectId}/${bm.chapterId}/${bm.sectionId}`}
                        className="flex flex-1 items-start gap-2 text-sm min-w-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded"
                      >
                        <BookmarkIcon />
                        <span className="font-medium text-gray-800 dark:text-gray-200 truncate">
                          {bm.title}
                        </span>
                      </Link>
                      <button
                        type="button"
                        onClick={() => removeBookmark(bm.subjectId, bm.chapterId, bm.sectionId)}
                        className="shrink-0 rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-red-500 dark:text-gray-500 dark:hover:bg-gray-800 dark:hover:text-red-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                        aria-label={`Remove bookmark: ${bm.title}`}
                      >
                        <TrashIcon />
                      </button>
                    </motion.li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
