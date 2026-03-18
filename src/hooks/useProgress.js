import useAppStore from '../store/appStore';

function useProgress() {
  const markComplete = useAppStore((state) => state.markSectionComplete);
  const isComplete = useAppStore((state) => state.isComplete);
  const getSubjectProgress = useAppStore((state) => state.getSubjectProgress);

  return { markComplete, isComplete, getSubjectProgress };
}

export default useProgress;
