/**
 * Subject colors and utility functions for Learn-LLM.
 */

// 15 subject colors with CSS variable references and hex fallbacks
export const SUBJECT_COLORS = {
  '01-text-fundamentals': {
    cssVar: '--color-subject-text-fundamentals',
    hex: '#6366f1',
    light: '#e0e7ff',
    name: 'Indigo',
  },
  '02-embeddings': {
    cssVar: '--color-subject-embeddings',
    hex: '#8b5cf6',
    light: '#ede9fe',
    name: 'Violet',
  },
  '03-neural-nlp': {
    cssVar: '--color-subject-neural-nlp',
    hex: '#06b6d4',
    light: '#cffafe',
    name: 'Cyan',
  },
  '04-transformer-architecture': {
    cssVar: '--color-subject-transformer-architecture',
    hex: '#10b981',
    light: '#d1fae5',
    name: 'Emerald',
  },
  '05-pretraining': {
    cssVar: '--color-subject-pretraining',
    hex: '#3b82f6',
    light: '#dbeafe',
    name: 'Blue',
  },
  '06-finetuning': {
    cssVar: '--color-subject-finetuning',
    hex: '#f59e0b',
    light: '#fef3c7',
    name: 'Amber',
  },
  '07-landmark-models': {
    cssVar: '--color-subject-landmark-models',
    hex: '#ef4444',
    light: '#fee2e2',
    name: 'Red',
  },
  '08-vision-language': {
    cssVar: '--color-subject-vision-language',
    hex: '#ec4899',
    light: '#fce7f3',
    name: 'Pink',
  },
  '09-tabular-models': {
    cssVar: '--color-subject-tabular-models',
    hex: '#14b8a6',
    light: '#ccfbf1',
    name: 'Teal',
  },
  '10-efficient-models': {
    cssVar: '--color-subject-efficient-models',
    hex: '#f97316',
    light: '#ffedd5',
    name: 'Orange',
  },
  '11-practical-finetuning': {
    cssVar: '--color-subject-practical-finetuning',
    hex: '#a855f7',
    light: '#f3e8ff',
    name: 'Purple',
  },
  '12-inference-serving': {
    cssVar: '--color-subject-inference-serving',
    hex: '#0ea5e9',
    light: '#e0f2fe',
    name: 'Sky',
  },
  '13-rag': {
    cssVar: '--color-subject-rag',
    hex: '#84cc16',
    light: '#ecfccb',
    name: 'Lime',
  },
  '14-agents': {
    cssVar: '--color-subject-agents',
    hex: '#f43f5e',
    light: '#ffe4e6',
    name: 'Rose',
  },
  '15-evaluation-safety': {
    cssVar: '--color-subject-evaluation-safety',
    hex: '#64748b',
    light: '#f1f5f9',
    name: 'Slate',
  },
};

// Difficulty level colors
export const DIFFICULTY_COLORS = {
  beginner: {
    hex: '#22c55e',
    light: '#dcfce7',
    textClass: 'text-green-400',
    bgClass: 'bg-green-500/20',
    borderClass: 'border-green-500/40',
  },
  intermediate: {
    hex: '#eab308',
    light: '#fef9c3',
    textClass: 'text-yellow-400',
    bgClass: 'bg-yellow-500/20',
    borderClass: 'border-yellow-500/40',
  },
  advanced: {
    hex: '#f97316',
    light: '#ffedd5',
    textClass: 'text-orange-400',
    bgClass: 'bg-orange-500/20',
    borderClass: 'border-orange-500/40',
  },
  research: {
    hex: '#ef4444',
    light: '#fee2e2',
    textClass: 'text-red-400',
    bgClass: 'bg-red-500/20',
    borderClass: 'border-red-500/40',
  },
};

/**
 * Get color config for a subject by its ID.
 * Falls back to a default indigo color if subjectId not found.
 */
export function getSubjectColor(subjectId) {
  // Try exact match first
  if (SUBJECT_COLORS[subjectId]) return SUBJECT_COLORS[subjectId];

  // Try partial match (e.g., '03' matches '03-neural-nlp')
  const found = Object.entries(SUBJECT_COLORS).find(
    ([key]) => key.startsWith(subjectId) || subjectId.startsWith(key.split('-')[0])
  );
  if (found) return found[1];

  // Default fallback
  return { cssVar: '--color-subject-default', hex: '#6366f1', light: '#e0e7ff', name: 'Indigo' };
}

/**
 * Get color config for a difficulty level.
 */
export function getDifficultyColor(level) {
  return (
    DIFFICULTY_COLORS[level] || {
      hex: '#6b7280',
      light: '#f3f4f6',
      textClass: 'text-gray-400',
      bgClass: 'bg-gray-500/20',
      borderClass: 'border-gray-500/40',
    }
  );
}
