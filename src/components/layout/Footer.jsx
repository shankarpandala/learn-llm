import { Link } from 'react-router-dom'

export default function Footer() {
  const year = new Date().getFullYear()

  return (
    <footer className="mt-auto border-t border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-950">
      <div className="mx-auto max-w-7xl px-6 py-10 md:py-12">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          {/* Brand */}
          <div className="space-y-3">
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 text-lg font-bold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded"
            >
              <span className="bg-gradient-to-r from-emerald-500 to-teal-600 bg-clip-text text-transparent">
                🧠 LearnLLM
              </span>
            </Link>
            <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
              An interactive learning platform covering everything you need to
              understand, build, and work with large language models.
            </p>
          </div>

          {/* Links */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wide">
              Links
            </h3>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://github.com/shankarpandala/learn-llm"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 transition-colors"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/shankarpandala/learn-llm/blob/main/LICENSE"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 transition-colors"
                >
                  MIT License
                </a>
              </li>
              <li>
                <Link
                  to="/progress"
                  className="text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 transition-colors"
                >
                  My Progress
                </Link>
              </li>
            </ul>
          </div>

          {/* Built with */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wide">
              Built With
            </h3>
            <ul className="space-y-2">
              {[
                { label: 'React 19', href: 'https://react.dev' },
                { label: 'KaTeX', href: 'https://katex.org' },
                { label: 'Tailwind CSS', href: 'https://tailwindcss.com' },
                { label: 'Framer Motion', href: 'https://www.framer.com/motion/' },
                { label: 'D3.js', href: 'https://d3js.org' },
              ].map(({ label, href }) => (
                <li key={label}>
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 transition-colors"
                  >
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Copyright */}
        <div className="mt-8 border-t border-gray-100 pt-6 dark:border-gray-800">
          <p className="text-center text-xs text-gray-400 dark:text-gray-500">
            &copy; {year} LearnLLM. Released under the MIT License. Content is
            for educational purposes.
          </p>
        </div>
      </div>
    </footer>
  )
}
