import { Link } from 'react-router-dom'

/**
 * Breadcrumb navigation.
 *
 * Props:
 *   items  {Array<{label: string, href?: string}>}
 *          Last item should have no href (current page).
 */
export default function Breadcrumbs({ items = [] }) {
  if (!items.length) return null

  return (
    <nav
      aria-label="Breadcrumb"
      className="hidden md:flex items-center flex-wrap gap-1 text-sm text-gray-500 dark:text-gray-400"
    >
      {items.map((item, idx) => {
        const isLast = idx === items.length - 1
        return (
          <span key={idx} className="flex items-center gap-1">
            {idx > 0 && (
              <span className="select-none text-gray-400 dark:text-gray-600" aria-hidden="true">
                ›
              </span>
            )}
            {isLast || !item.href ? (
              <span
                className="font-medium text-gray-800 dark:text-gray-200 truncate max-w-[200px]"
                aria-current="page"
              >
                {item.label}
              </span>
            ) : (
              <Link
                to={item.href}
                className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors truncate max-w-[160px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded"
              >
                {item.label}
              </Link>
            )}
          </span>
        )
      })}
    </nav>
  )
}
