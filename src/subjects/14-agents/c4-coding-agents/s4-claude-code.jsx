import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ClaudeCode() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Claude Code Capabilities</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Claude Code is Anthropic's agentic coding tool that operates directly in
        your terminal. It combines Claude's reasoning with the ability to read files,
        write code, run commands, search codebases, and interact with git. It represents
        a practical implementation of the agent patterns covered throughout this subject.
      </p>

      <DefinitionBlock
        title="Claude Code"
        definition="An agentic coding assistant that runs as a CLI tool in your terminal. It can read and edit files, execute shell commands, search codebases, run tests, create commits, and manage git workflows. It operates in a ReAct-style loop with human-in-the-loop approval for potentially dangerous operations."
        id="def-claude-code"
      />

      <h2 className="text-2xl font-semibold">Architecture and Tool Set</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Claude Code implements the agent patterns we have studied: a ReAct loop with
        specialized tools for coding tasks, self-reflection for quality, and
        plan-and-execute for complex multi-step changes.
      </p>

      <ExampleBlock
        title="Claude Code's Tool Set"
        problem="What tools does Claude Code use internally to accomplish coding tasks?"
        steps={[
          { formula: 'Read: Read files with line numbers and ranges', explanation: 'Equivalent to cat -n, supports reading specific line ranges for large files.' },
          { formula: 'Edit: Make targeted string replacements in files', explanation: 'Precise edits rather than rewriting entire files, reducing errors.' },
          { formula: 'Bash: Execute shell commands (git, npm, python, etc.)', explanation: 'Full access to the development environment with user approval.' },
          { formula: 'Grep: Search file contents with regex patterns', explanation: 'Fast codebase search powered by ripgrep.' },
          { formula: 'Glob: Find files by name pattern', explanation: 'Locate files across the project structure.' },
          { formula: 'Write: Create new files or complete rewrites', explanation: 'For new files that do not exist yet.' },
        ]}
        id="example-claude-code-tools"
      />

      <PythonCode
        title="claude_code_workflow.py"
        code={`# Claude Code workflow: what happens under the hood
# when you ask it to "fix the failing tests in auth module"

# This is a conceptual representation of Claude Code's internal loop

workflow_steps = """
1. UNDERSTAND THE REQUEST
   - Parse the user's intent: fix failing tests
   - Identify scope: auth module

2. EXPLORE THE CODEBASE (using Grep, Glob, Read tools)
   - Glob("**/auth/**/*.test.*")  -> find test files
   - Glob("**/auth/**/*.py")      -> find source files
   - Bash("python -m pytest tests/auth/ -v")  -> see which tests fail

3. ANALYZE FAILURES (using Read tool)
   - Read the failing test file to understand expected behavior
   - Read the source file to understand current implementation
   - Read the error traceback to pinpoint the bug

4. PLAN THE FIX (internal reasoning)
   - Identify root cause from error + code analysis
   - Determine which files need changes
   - Consider impact on other tests

5. IMPLEMENT (using Edit tool)
   - Edit the specific lines that contain the bug
   - Use targeted string replacement, not full file rewrites

6. VERIFY (using Bash tool)
   - Bash("python -m pytest tests/auth/ -v")  -> run tests again
   - If tests still fail: analyze new errors and iterate
   - If tests pass: check that no other tests broke

7. REPORT
   - Summarize what was changed and why
   - Show the test results
"""

# Example of how Claude Code uses the Edit tool internally
# Instead of rewriting a file, it makes surgical changes:

edit_example = {
    "tool": "Edit",
    "file_path": "/home/user/project/src/auth/session.py",
    "old_string": "    if token.expires_at < datetime.now():",
    "new_string": "    if token.expires_at < datetime.utcnow():",
}
# This fixes a timezone bug with a single-line change`}
        id="code-claude-code-workflow"
      />

      <h2 className="text-2xl font-semibold">Effective Usage Patterns</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Getting the most out of Claude Code involves understanding how to give it
        context, structure requests, and leverage its agentic capabilities.
      </p>

      <PythonCode
        title="effective_claude_code_usage.py"
        code={`# Patterns for effective Claude Code usage

# 1. SPECIFIC REQUESTS (better than vague)
bad  = "fix the bugs"
good = "Fix the TypeError in user_service.py line 45 where None is passed to .strip()"

# 2. PROVIDE CONTEXT VIA FILES
# Use CLAUDE.md at the project root to give Claude Code persistent context
claude_md = """
# CLAUDE.md - Project Context for Claude Code

## Architecture
- FastAPI backend in /src
- React frontend in /client
- PostgreSQL database with SQLAlchemy ORM
- Tests use pytest with fixtures in conftest.py

## Conventions
- Use type hints on all functions
- Follow Google-style docstrings
- Database queries go through the repository pattern in /src/repos/
- All API endpoints need input validation with Pydantic

## Common Commands
- Run tests: pytest -xvs
- Start dev server: uvicorn src.main:app --reload
- Run linter: ruff check src/
- Run type checker: mypy src/
"""

# 3. ITERATIVE REFINEMENT
# Claude Code naturally supports multi-turn conversations:
# Turn 1: "Add a rate limiter to the API endpoints"
# Turn 2: "Now add tests for the rate limiter"
# Turn 3: "The tests are passing but add a test for the Redis fallback case"

# 4. COMPLEX MULTI-FILE CHANGES
# Claude Code excels at changes that span multiple files:
complex_task = """
Refactor the payment processing module:
1. Extract the Stripe logic into a separate PaymentProvider interface
2. Create StripeProvider and MockProvider implementations
3. Update all call sites to use the interface
4. Add tests for both providers
5. Update the dependency injection configuration
"""

# 5. GIT WORKFLOWS
# Claude Code can manage the full git workflow:
# "Create a new branch, implement the feature, write tests,
#  and create a PR with a description of the changes"

# 6. HEADLESS / CI MODE
# Claude Code can run non-interactively for automation:
# claude -p "Run the test suite and fix any failures" --allowedTools Edit,Bash,Read`}
        id="code-effective-usage"
      />

      <NoteBlock
        type="tip"
        title="CLAUDE.md for Project Context"
        content="Create a CLAUDE.md file at the root of your repository with project architecture, conventions, common commands, and important context. Claude Code reads this file automatically and uses it to make better decisions about code style, project structure, and tooling. This is far more effective than repeating context in every prompt."
        id="note-claude-md"
      />

      <NoteBlock
        type="note"
        title="Safety Model: Human-in-the-Loop"
        content="Claude Code implements a layered safety model. Read-only operations (searching, reading files) run automatically. Write operations (editing files) proceed with notification. Potentially dangerous operations (running arbitrary shell commands, git push) require explicit user approval. This balances productivity with safety, following the principle of least privilege."
        id="note-safety-model"
      />

      <WarningBlock
        title="Agent Limitations"
        content="Even the best coding agents have limitations. They can struggle with: large-scale architectural refactors spanning dozens of files, deeply domain-specific logic requiring expert knowledge, performance optimization requiring profiling data, and security-sensitive code where subtle bugs have outsized impact. Use coding agents as powerful assistants, not as replacements for engineering judgment."
        id="warning-limitations"
      />

      <NoteBlock
        type="historical"
        title="Evolution of Coding Agents"
        content="Coding assistants evolved from autocomplete (GitHub Copilot, 2021) to chat-based helpers (ChatGPT, 2022) to agentic tools (Devin, Claude Code, Cursor, 2024-2025). Each generation added more autonomy: autocomplete suggests lines, chat produces functions, and agents navigate repos, run tests, and create PRs. The trend is toward greater autonomy with better safety guardrails."
        id="note-evolution"
      />
    </div>
  )
}
