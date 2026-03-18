import { useState, useCallback } from 'react'

/**
 * Minimal Python tokenizer for syntax highlighting.
 * Returns an array of { type, value } tokens.
 */
function tokenizePython(code) {
  const tokens = []
  let i = 0

  const KEYWORDS = new Set([
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
    'while', 'with', 'yield',
  ])

  const BUILTINS = new Set([
    'print', 'range', 'len', 'int', 'float', 'str', 'list', 'dict',
    'set', 'tuple', 'bool', 'type', 'isinstance', 'enumerate', 'zip',
    'map', 'filter', 'sorted', 'reversed', 'min', 'max', 'sum', 'abs',
    'any', 'all', 'open', 'input', 'super', 'property', 'staticmethod',
    'classmethod', 'hasattr', 'getattr', 'setattr', 'delattr', 'callable',
    'iter', 'next', 'repr', 'format', 'id', 'hash', 'hex', 'oct', 'bin',
    'ord', 'chr', 'round', 'pow', 'divmod', 'vars', 'dir', 'help',
    'ValueError', 'TypeError', 'KeyError', 'IndexError', 'RuntimeError',
    'StopIteration', 'Exception', 'NotImplementedError', 'AttributeError',
    'ImportError', 'FileNotFoundError', 'ZeroDivisionError', 'OSError',
  ])

  while (i < code.length) {
    const ch = code[i]

    // Whitespace
    if (ch === ' ' || ch === '\t') {
      let start = i
      while (i < code.length && (code[i] === ' ' || code[i] === '\t')) i++
      tokens.push({ type: 'plain', value: code.slice(start, i) })
      continue
    }

    // Newline
    if (ch === '\n') {
      tokens.push({ type: 'plain', value: '\n' })
      i++
      continue
    }

    // Comments
    if (ch === '#') {
      let start = i
      while (i < code.length && code[i] !== '\n') i++
      tokens.push({ type: 'comment', value: code.slice(start, i) })
      continue
    }

    // Triple-quoted strings
    if (
      (ch === '"' || ch === "'") &&
      code.slice(i, i + 3) === ch.repeat(3)
    ) {
      const quote = ch.repeat(3)
      let start = i
      i += 3
      while (i < code.length && code.slice(i, i + 3) !== quote) {
        if (code[i] === '\\') i++
        i++
      }
      i += 3
      tokens.push({ type: 'string', value: code.slice(start, i) })
      continue
    }

    // Single/double quoted strings
    if (ch === '"' || ch === "'") {
      let start = i
      const quote = ch
      i++
      while (i < code.length && code[i] !== quote && code[i] !== '\n') {
        if (code[i] === '\\') i++
        i++
      }
      if (i < code.length && code[i] === quote) i++
      tokens.push({ type: 'string', value: code.slice(start, i) })
      continue
    }

    // f-strings (simplified — treat as string)
    if ((ch === 'f' || ch === 'F') && i + 1 < code.length && (code[i + 1] === '"' || code[i + 1] === "'")) {
      let start = i
      i++ // skip f
      const quote = code[i]
      i++
      while (i < code.length && code[i] !== quote && code[i] !== '\n') {
        if (code[i] === '\\') i++
        i++
      }
      if (i < code.length && code[i] === quote) i++
      tokens.push({ type: 'string', value: code.slice(start, i) })
      continue
    }

    // Numbers
    if (/\d/.test(ch) || (ch === '.' && i + 1 < code.length && /\d/.test(code[i + 1]))) {
      let start = i
      // hex
      if (ch === '0' && i + 1 < code.length && (code[i + 1] === 'x' || code[i + 1] === 'X')) {
        i += 2
        while (i < code.length && /[0-9a-fA-F_]/.test(code[i])) i++
      } else {
        while (i < code.length && /[\d_]/.test(code[i])) i++
        if (i < code.length && code[i] === '.') {
          i++
          while (i < code.length && /[\d_]/.test(code[i])) i++
        }
        if (i < code.length && (code[i] === 'e' || code[i] === 'E')) {
          i++
          if (i < code.length && (code[i] === '+' || code[i] === '-')) i++
          while (i < code.length && /[\d_]/.test(code[i])) i++
        }
      }
      // j suffix for complex
      if (i < code.length && (code[i] === 'j' || code[i] === 'J')) i++
      tokens.push({ type: 'number', value: code.slice(start, i) })
      continue
    }

    // Identifiers / keywords
    if (/[a-zA-Z_]/.test(ch)) {
      let start = i
      while (i < code.length && /[a-zA-Z0-9_]/.test(code[i])) i++
      const word = code.slice(start, i)
      if (KEYWORDS.has(word)) {
        tokens.push({ type: 'keyword', value: word })
      } else if (BUILTINS.has(word)) {
        tokens.push({ type: 'builtin', value: word })
      } else if (word === 'self' || word === 'cls') {
        tokens.push({ type: 'self', value: word })
      } else {
        tokens.push({ type: 'identifier', value: word })
      }
      continue
    }

    // Decorators
    if (ch === '@') {
      let start = i
      i++
      while (i < code.length && /[a-zA-Z0-9_.]/.test(code[i])) i++
      tokens.push({ type: 'decorator', value: code.slice(start, i) })
      continue
    }

    // Operators and punctuation
    tokens.push({ type: 'operator', value: ch })
    i++
  }

  return tokens
}

const TOKEN_COLORS = {
  keyword:    'text-purple-400',
  builtin:    'text-cyan-400',
  string:     'text-green-400',
  number:     'text-orange-400',
  comment:    'text-gray-500 italic',
  decorator:  'text-yellow-400',
  self:       'text-red-400',
  operator:   'text-gray-400',
  identifier: 'text-gray-200',
  plain:      '',
}

/**
 * Syntax-highlighted Python code block.
 *
 * Props:
 *   code      {string}  Python source code
 *   title     {string}  Optional filename / title for the header
 *   colabUrl  {string}  Optional Google Colab link
 *   showLines {bool}    Show line numbers (default true)
 *   id        {string}  Optional anchor id
 */
export default function PythonCode({
  code = '',
  title,
  colabUrl,
  showLines = true,
  id,
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [code])

  const tokens = tokenizePython(code)
  const lines = code.split('\n')
  const lineCount = lines.length

  // Build highlighted code by rendering tokens
  const highlighted = tokens.map((token, i) => {
    const cls = TOKEN_COLORS[token.type]
    return cls ? (
      <span key={i} className={cls}>{token.value}</span>
    ) : (
      <span key={i}>{token.value}</span>
    )
  })

  return (
    <div
      id={id}
      className="my-6 overflow-hidden rounded-xl border border-gray-700 bg-gray-900 shadow-sm"
    >
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-gray-700 bg-gray-800/80 px-4 py-2">
        <div className="flex items-center gap-3">
          {/* Traffic light dots */}
          <div className="flex items-center gap-1.5" aria-hidden="true">
            <span className="h-3 w-3 rounded-full bg-red-500/80" />
            <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
            <span className="h-3 w-3 rounded-full bg-green-500/80" />
          </div>

          {/* Title / filename */}
          {title && (
            <span className="text-xs font-medium text-gray-400">{title}</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Language label */}
          <span className="rounded bg-gray-700/60 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-gray-400">
            Python
          </span>

          {/* Colab link */}
          {colabUrl && (
            <a
              href={colabUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 rounded bg-yellow-600/20 px-2 py-0.5 text-[10px] font-semibold text-yellow-400 hover:bg-yellow-600/30 transition-colors"
            >
              <svg className="h-3 w-3" viewBox="0 0 24 24" fill="currentColor">
                <path d="M16.9414 4.9757a7.033 7.033 0 0 0-9.8728 1.7054l3.6068 2.5708a3.52 3.52 0 0 1 4.9392-.8525 3.521 3.521 0 0 1 .8525 4.9392l3.6068 2.5709a7.033 7.033 0 0 0-.3325-8.8338zm-9.1057 4.1884L4.229 6.5933a7.033 7.033 0 0 0 .3325 8.8338 7.033 7.033 0 0 0 9.8728-1.7054l-3.6068-2.5708a3.52 3.52 0 0 1-4.9392.8524 3.52 3.52 0 0 1-.8525-4.939z" />
              </svg>
              Colab
            </a>
          )}

          {/* Copy button */}
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 rounded bg-gray-700/60 px-2 py-0.5 text-[10px] font-semibold text-gray-400 hover:bg-gray-600/60 hover:text-gray-300 transition-colors"
            aria-label="Copy code"
          >
            {copied ? (
              <>
                <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                Copied
              </>
            ) : (
              <>
                <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                Copy
              </>
            )}
          </button>
        </div>
      </div>

      {/* Code area */}
      <div className="overflow-x-auto">
        <pre className="p-4 text-sm leading-relaxed font-mono">
          {showLines ? (
            <table className="w-full border-collapse">
              <tbody>
                {lines.map((line, idx) => (
                  <tr key={idx} className="hover:bg-gray-800/50">
                    <td className="select-none pr-4 text-right align-top text-xs text-gray-600 w-8">
                      {idx + 1}
                    </td>
                    <td className="whitespace-pre text-gray-200">
                      {/* Re-render tokens for this line */}
                      {tokenizePython(line).map((token, j) => {
                        const cls = TOKEN_COLORS[token.type]
                        return cls ? (
                          <span key={j} className={cls}>{token.value}</span>
                        ) : (
                          <span key={j}>{token.value}</span>
                        )
                      })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <code>{highlighted}</code>
          )}
        </pre>
      </div>
    </div>
  )
}
