import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RepoUnderstanding() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Codebase Navigation and Understanding</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Real-world coding tasks rarely involve writing code from scratch. More often,
        an agent must understand an existing codebase, find relevant files, trace
        dependencies, and make targeted changes. Codebase navigation is the foundation
        of effective coding agents.
      </p>

      <DefinitionBlock
        title="Codebase Navigation"
        definition="The ability of a coding agent to explore, search, and understand the structure, conventions, and dependencies of an existing software repository. This includes finding relevant files, understanding module relationships, identifying patterns, and locating the right place to make changes."
        id="def-repo-navigation"
      />

      <h2 className="text-2xl font-semibold">Search-Based Navigation</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Coding agents navigate repositories primarily through search tools: grep for
        content, find/glob for files, and AST analysis for structure. The quality of
        an agent's search strategy determines how quickly it finds relevant code.
      </p>

      <PythonCode
        title="repo_navigation_tools.py"
        code={`import os
import subprocess
import json

class RepoNavigator:
    """Tools for navigating and understanding a codebase."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def search_content(self, pattern: str, file_glob: str = "*") -> list[dict]:
        """Search file contents using ripgrep (rg)."""
        try:
            result = subprocess.run(
                ["rg", "--json", "-g", file_glob, pattern, self.repo_path],
                capture_output=True, text=True, timeout=30
            )
            matches = []
            for line in result.stdout.strip().split("\\n"):
                if not line:
                    continue
                data = json.loads(line)
                if data["type"] == "match":
                    matches.append({
                        "file": data["data"]["path"]["text"],
                        "line": data["data"]["line_number"],
                        "text": data["data"]["lines"]["text"].strip(),
                    })
            return matches[:20]  # Limit results
        except Exception as e:
            return [{"error": str(e)}]

    def find_files(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern."""
        import glob
        return sorted(glob.glob(
            os.path.join(self.repo_path, "**", pattern),
            recursive=True
        ))[:30]

    def read_file(self, filepath: str, start: int = 0, end: int = 100) -> str:
        """Read a file with line range."""
        full_path = os.path.join(self.repo_path, filepath)
        with open(full_path) as f:
            lines = f.readlines()
        selected = lines[start:end]
        return "".join(
            f"{i+start+1:4d} | {line}"
            for i, line in enumerate(selected)
        )

    def get_structure(self, max_depth: int = 3) -> str:
        """Get the directory tree structure."""
        result = subprocess.run(
            ["find", self.repo_path, "-maxdepth", str(max_depth),
             "-not", "-path", "*node_modules*",
             "-not", "-path", "*.git*"],
            capture_output=True, text=True
        )
        return result.stdout

    def get_definitions(self, filepath: str) -> list[str]:
        """Extract function and class definitions from a Python file."""
        full_path = os.path.join(self.repo_path, filepath)
        defs = []
        with open(full_path) as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith(("def ", "class ", "async def ")):
                    defs.append(f"L{i}: {stripped}")
        return defs

# Usage
nav = RepoNavigator("/path/to/project")
# nav.search_content("def authenticate", "*.py")
# nav.find_files("*.test.py")
# nav.get_definitions("src/auth/handlers.py")`}
        id="code-repo-navigation"
      />

      <ExampleBlock
        title="Navigation Strategy for Bug Fixing"
        problem="Agent receives: 'Fix the login timeout bug reported in issue #234'"
        steps={[
          { formula: 'Step 1: Search for "login" and "timeout" in the codebase', explanation: 'Cast a wide net to find relevant files.' },
          { formula: 'Step 2: Identify the authentication module structure', explanation: 'Read directory tree around auth-related files.' },
          { formula: 'Step 3: Read the login handler and timeout configuration', explanation: 'Focus on the most likely locations for the bug.' },
          { formula: 'Step 4: Trace the call chain from login to session creation', explanation: 'Follow the code path to find where timeout is set.' },
          { formula: 'Step 5: Read related tests to understand expected behavior', explanation: 'Tests reveal the intended behavior and edge cases.' },
        ]}
        id="example-nav-strategy"
      />

      <PythonCode
        title="agent_codebase_exploration.py"
        code={`import anthropic

client = anthropic.Anthropic()

# Tools that a coding agent uses for repo understanding
tools = [
    {
        "name": "search_code",
        "description": (
            "Search for a pattern in the codebase using regex. "
            "Returns matching lines with file paths and line numbers. "
            "Use this to find function definitions, class usages, "
            "imports, and specific code patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "file_glob": {
                    "type": "string",
                    "description": "File pattern filter, e.g. '*.py', '*.ts'",
                    "default": "*"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "read_file",
        "description": (
            "Read a file's contents. Returns numbered lines. "
            "Use start_line and end_line for large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer", "default": 200}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories in a directory path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."}
            },
            "required": []
        }
    },
    {
        "name": "find_references",
        "description": (
            "Find all references to a symbol (function, class, variable) "
            "across the codebase. Useful for understanding how code is used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name to find references for"}
            },
            "required": ["symbol"]
        }
    }
]

# The agent explores the codebase interactively
messages = [{
    "role": "user",
    "content": (
        "Understand how the authentication system works in this project. "
        "Find the main auth module, trace the login flow, and summarize "
        "the architecture."
    )
}]

# The agent would iterate: search -> read -> search -> read -> summarize`}
        id="code-agent-exploration"
      />

      <NoteBlock
        type="intuition"
        title="Context Window as Working Memory"
        content="A coding agent's context window is its working memory. It can only reason about code it has read into context. Effective agents are strategic about what they read: they start with high-level structure (directory listing, file outlines), then drill into specific files based on relevance. Poor agents read files sequentially and run out of context before finding the relevant code."
        id="note-context-window"
      />

      <WarningBlock
        title="Large Codebases Exceed Context"
        content="Production repositories can have millions of lines of code. No LLM can fit an entire codebase in its context window. Agents must use search tools strategically, read only relevant sections, and maintain a mental model of the codebase structure. Retrieval-augmented approaches (embedding the codebase and searching semantically) help but are not a complete solution."
        id="warning-large-codebases"
      />

      <NoteBlock
        type="tip"
        title="Indexing for Faster Navigation"
        content="Pre-index the codebase with tools like tree-sitter (for AST parsing), ctags (for symbol definitions), or embedding-based search (for semantic similarity). An indexed codebase lets the agent find relevant code in milliseconds rather than running grep over the entire repository for each search."
        id="note-indexing"
      />
    </div>
  )
}
