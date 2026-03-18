import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const useAppStore = create(
  persist(
    (set, get) => ({
      // State
      theme: 'dark',
      completedSections: [],
      bookmarks: [],

      // Theme actions
      setTheme: (theme) => set({ theme }),
      toggleTheme: () =>
        set((state) => ({ theme: state.theme === 'dark' ? 'light' : 'dark' })),

      // Section completion actions
      markSectionComplete: (subjectId, chapterId, sectionId) => {
        const key = `${subjectId}::${chapterId}::${sectionId}`;
        set((state) => {
          if (state.completedSections.includes(key)) return state;
          return { completedSections: [...state.completedSections, key] };
        });
      },

      isComplete: (subjectId, chapterId, sectionId) => {
        const key = `${subjectId}::${chapterId}::${sectionId}`;
        return get().completedSections.includes(key);
      },

      // Bookmark actions
      bookmarkSection: (subjectId, chapterId, sectionId, title) => {
        set((state) => {
          const exists = state.bookmarks.some(
            (b) =>
              b.subjectId === subjectId &&
              b.chapterId === chapterId &&
              b.sectionId === sectionId
          );
          if (exists) return state;
          return {
            bookmarks: [
              ...state.bookmarks,
              { subjectId, chapterId, sectionId, title },
            ],
          };
        });
      },

      removeBookmark: (subjectId, chapterId, sectionId) => {
        set((state) => ({
          bookmarks: state.bookmarks.filter(
            (b) =>
              !(
                b.subjectId === subjectId &&
                b.chapterId === chapterId &&
                b.sectionId === sectionId
              )
          ),
        }));
      },

      getSubjectProgress: (subjectId, totalSections) => {
        if (!totalSections || totalSections === 0) return 0;
        const completed = get().completedSections.filter((key) =>
          key.startsWith(`${subjectId}::`)
        ).length;
        return Math.round((completed / totalSections) * 100);
      },
    }),
    {
      name: 'learn-llm-app-store',
      partialize: (state) => ({
        theme: state.theme,
        completedSections: state.completedSections,
        bookmarks: state.bookmarks,
      }),
    }
  )
);

export default useAppStore;
