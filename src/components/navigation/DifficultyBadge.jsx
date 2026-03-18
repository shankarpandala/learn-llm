import React from 'react';

/**
 * Configuration for each difficulty level.
 */
const LEVEL_CONFIG = {
  beginner: {
    label: 'Beginner',
    dotColor: 'bg-green-500',
    className:
      'border-green-300 bg-green-100 text-green-700 dark:border-green-700/60 dark:bg-green-900/30 dark:text-green-400',
  },
  intermediate: {
    label: 'Intermediate',
    dotColor: 'bg-yellow-500',
    className:
      'border-yellow-300 bg-yellow-100 text-yellow-700 dark:border-yellow-700/60 dark:bg-yellow-900/30 dark:text-yellow-400',
  },
  advanced: {
    label: 'Advanced',
    dotColor: 'bg-orange-500',
    className:
      'border-orange-300 bg-orange-100 text-orange-700 dark:border-orange-700/60 dark:bg-orange-900/30 dark:text-orange-400',
  },
  research: {
    label: 'Research',
    dotColor: 'bg-red-500',
    className:
      'border-red-300 bg-red-100 text-red-700 dark:border-red-700/60 dark:bg-red-900/30 dark:text-red-400',
  },
};

/**
 * Difficulty level badge.
 *
 * Props:
 *   level  {string}  'beginner' | 'intermediate' | 'advanced' | 'research'
 *   size   {string}  'sm' | 'md' (default 'md')
 */
function DifficultyBadge({ level, size = 'md' }) {
  const config = LEVEL_CONFIG[level] || {
    label: level || 'Unknown',
    dotColor: 'bg-gray-500',
    className: 'border-gray-300 bg-gray-100 text-gray-600 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-400',
  };

  const sizeClasses =
    size === 'sm'
      ? 'px-2 py-0.5 text-xs gap-1'
      : 'px-2.5 py-1 text-xs gap-1.5 sm:text-sm';

  return (
    <span
      className={`inline-flex items-center rounded-full border font-semibold capitalize ${sizeClasses} ${config.className}`}
    >
      <span className={`h-2 w-2 rounded-full ${config.dotColor}`} aria-hidden="true" />
      {config.label}
    </span>
  );
}

export default DifficultyBadge;
