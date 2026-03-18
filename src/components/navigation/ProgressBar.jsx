import React from 'react';

/**
 * Horizontal progress bar.
 *
 * Props:
 *   value        {number}   Progress value 0–100
 *   color        {string}   CSS color string for the fill (optional)
 *   label        {string}   Label text displayed above the bar (optional)
 *   showPercent  {bool}     Show percentage value (default true)
 *   size         {string}   'sm' | 'md' | 'lg' (controls bar height)
 *   animate      {bool}     Animate the fill on mount (default true)
 */
function ProgressBar({
  value = 0,
  color,
  label,
  showPercent = true,
  size = 'md',
  animate = true,
}) {
  const clamped = Math.max(0, Math.min(100, value));

  const heightClasses = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  };

  const barHeight = heightClasses[size] || heightClasses.md;

  // Use inline style for custom color; otherwise fall back to Tailwind indigo.
  const fillStyle = color
    ? { width: `${clamped}%`, backgroundColor: color }
    : { width: `${clamped}%` };

  const fillClass = color
    ? `${barHeight} rounded-full transition-all duration-700`
    : `${barHeight} rounded-full bg-indigo-500 dark:bg-indigo-400 transition-all duration-700`;

  return (
    <div className="w-full space-y-1.5">
      {/* Label row */}
      {(label || showPercent) && (
        <div className="flex items-center justify-between">
          {label && (
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
              {label}
            </span>
          )}
          {showPercent && (
            <span className="ml-auto text-xs font-semibold tabular-nums text-gray-700 dark:text-gray-300">
              {clamped}%
            </span>
          )}
        </div>
      )}

      {/* Bar track */}
      <div
        className={`w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700 ${barHeight}`}
        role="progressbar"
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label || 'Progress'}
      >
        <div
          className={fillClass}
          style={fillStyle}
        />
      </div>
    </div>
  );
}

export default ProgressBar;
