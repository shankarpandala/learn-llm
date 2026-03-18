import { HashRouter, Routes, Route } from 'react-router-dom'
import { lazy, Suspense, useState } from 'react'
import Navbar from './components/layout/Navbar.jsx'
import Sidebar from './components/layout/Sidebar.jsx'
import Footer from './components/layout/Footer.jsx'
import useTheme from './hooks/useTheme.js'

const HomePage = lazy(() => import('./pages/HomePage.jsx'))
const SubjectPage = lazy(() => import('./pages/SubjectPage.jsx'))
const ChapterPage = lazy(() => import('./pages/ChapterPage.jsx'))
const SectionPage = lazy(() => import('./pages/SectionPage.jsx'))
const ProgressPage = lazy(() => import('./pages/ProgressPage.jsx'))
const SearchPage = lazy(() => import('./pages/SearchPage.jsx'))

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center min-h-[40vh] text-zinc-400 text-lg">
      Loading…
    </div>
  )
}

function AppShell({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex flex-col min-h-screen">
      <Navbar
        onToggleSidebar={() => setSidebarOpen((o) => !o)}
        sidebarOpen={sidebarOpen}
      />
      <div className="flex flex-1">
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        {/* On desktop the sidebar is fixed 280px wide, so offset the main content */}
        <main className="flex-1 min-w-0 lg:ml-[280px]">
          {children}
        </main>
      </div>
      <Footer />
    </div>
  )
}

export default function App() {
  useTheme()

  return (
    <HashRouter>
      <AppShell>
        <Suspense fallback={<LoadingFallback />}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/subjects/:subjectId" element={<SubjectPage />} />
            <Route path="/subjects/:subjectId/:chapterId" element={<ChapterPage />} />
            <Route path="/subjects/:subjectId/:chapterId/:sectionId" element={<SectionPage />} />
            <Route path="/progress" element={<ProgressPage />} />
            <Route path="/search" element={<SearchPage />} />
          </Routes>
        </Suspense>
      </AppShell>
    </HashRouter>
  )
}
