import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { CURRICULUM } from '../../subjects/index.js'
import useProgress from '../../hooks/useProgress.js'

function ChevronIcon({ open }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`transition-transform duration-200 ${open ? 'rotate-90' : 'rotate-0'}`}
      aria-hidden="true"
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-emerald-500 dark:text-emerald-400 shrink-0"
      aria-label="Completed"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}

function SectionLink({ section, chapter, subject, isActive, isDone }) {
  const to = `/subjects/${subject.id}/${chapter.id}/${section.id}`
  return (
    <Link
      to={to}
      className={`group flex items-center gap-2 rounded-md py-1.5 pl-8 pr-3 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
        isActive
          ? 'bg-emerald-50 font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
          : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800/60 dark:hover:text-gray-200'
      }`}
      aria-current={isActive ? 'page' : undefined}
    >
      <span className="flex-1 leading-snug truncate">{section.title}</span>
      {isDone && <CheckIcon />}
    </Link>
  )
}

function ChapterRow({ chapter, subject, activePathname }) {
  const { isComplete } = useProgress()
  const defaultOpen = chapter.sections?.some(
    (sec) => activePathname.includes(`/${chapter.id}/${sec.id}`)
  )
  const [open, setOpen] = useState(defaultOpen || false)

  const completedCount = chapter.sections?.filter((sec) =>
    isComplete(subject.id, chapter.id, sec.id)
  ).length || 0
  const total = chapter.sections?.length || 0

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 rounded-md py-1.5 pl-5 pr-3 text-left text-sm font-medium text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        aria-expanded={open}
      >
        <ChevronIcon open={open} />
        <span className="flex-1 truncate leading-snug">{chapter.title}</span>
        {completedCount > 0 && (
          <span className="ml-auto text-xs tabular-nums text-gray-400 dark:text-gray-500">
            {completedCount}/{total}
          </span>
        )}
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="chapter-sections"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="pb-1">
              {chapter.sections?.map((section) => {
                const isActive = activePathname.includes(`/${chapter.id}/${section.id}`)
                const isDone = isComplete(subject.id, chapter.id, section.id)
                return (
                  <SectionLink
                    key={section.id}
                    section={section}
                    chapter={chapter}
                    subject={subject}
                    isActive={isActive}
                    isDone={isDone}
                  />
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function SubjectRow({ subject, activePathname }) {
  const isSubjectActive = activePathname.includes(`/subjects/${subject.id}`)
  const defaultOpen = isSubjectActive
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="border-b border-gray-100 dark:border-gray-800/60 last:border-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`flex w-full items-center gap-2.5 px-4 py-2.5 text-left text-sm font-semibold transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
          isSubjectActive
            ? 'text-gray-900 dark:text-gray-50'
            : 'text-gray-700 dark:text-gray-300'
        }`}
        aria-expanded={open}
      >
        <span
          className="h-3 w-3 rounded-full shrink-0"
          style={{ backgroundColor: subject.colorHex }}
          aria-hidden="true"
        />
        <span className="flex-1 truncate leading-snug">{subject.title}</span>
        <ChevronIcon open={open} />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="subject-chapters"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="pb-2">
              {subject.chapters?.map((chapter) => (
                <ChapterRow
                  key={chapter.id}
                  chapter={chapter}
                  subject={subject}
                  activePathname={activePathname}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function Sidebar({ isOpen, onClose }) {
  const location = useLocation()

  const sidebarContent = (
    <aside
      className="flex h-full flex-col bg-white dark:bg-gray-950"
      aria-label="Site navigation"
    >
      {/* Header */}
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-gray-200 px-4 dark:border-gray-800">
        <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
          Curriculum
        </span>
        <button
          type="button"
          onClick={onClose}
          className="lg:hidden rounded-md p-1.5 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
          aria-label="Close sidebar"
        >
          <CloseIcon />
        </button>
      </div>

      {/* Subject list */}
      <nav className="flex-1 overflow-y-auto" aria-label="Curriculum subjects">
        {CURRICULUM.map((subject) => (
          <SubjectRow
            key={subject.id}
            subject={subject}
            activePathname={location.pathname}
          />
        ))}
      </nav>

      {/* Footer links */}
      <div className="shrink-0 border-t border-gray-100 p-4 dark:border-gray-800">
        <Link
          to="/progress"
          className="block rounded-md bg-emerald-50 px-3 py-2 text-center text-sm font-medium text-emerald-700 hover:bg-emerald-100 dark:bg-emerald-900/30 dark:text-emerald-300 dark:hover:bg-emerald-900/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        >
          View My Progress →
        </Link>
      </div>
    </aside>
  )

  return (
    <>
      {/* Desktop: fixed sidebar */}
      <div className="hidden lg:block fixed top-14 left-0 h-[calc(100vh-3.5rem)] w-[280px] overflow-hidden border-r border-gray-200 dark:border-gray-800 z-40">
        {sidebarContent}
      </div>

      {/* Mobile: drawer overlay */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-40 bg-black/50 lg:hidden"
              onClick={onClose}
              aria-hidden="true"
            />
            {/* Drawer */}
            <motion.div
              key="drawer"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              className="fixed top-0 left-0 z-50 h-full w-[280px] shadow-2xl lg:hidden"
            >
              {sidebarContent}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  )
}
