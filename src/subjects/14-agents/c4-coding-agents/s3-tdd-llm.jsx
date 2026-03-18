import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TDDWithLLMs() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Test-Driven Development with LLMs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Test-driven development (TDD) pairs naturally with LLM coding agents. Tests
        provide an objective, automated way to verify generated code. The TDD loop
        (write tests, generate code, run tests, fix failures) gives the agent concrete
        feedback at each step, dramatically improving code reliability.
      </p>

      <DefinitionBlock
        title="LLM-Assisted TDD"
        definition="A development workflow where an LLM agent participates in the test-driven development cycle: either generating tests from specifications, generating code to pass existing tests, or both. The test suite serves as an automated oracle that guides the agent toward correct implementations."
        id="def-tdd-llm"
      />

      <ExampleBlock
        title="TDD Cycle with an LLM Agent"
        problem="Build a URL shortener service using TDD with an LLM."
        steps={[
          { formula: 'Red: Agent generates tests from the specification', explanation: 'Tests define the expected behavior before any implementation exists.' },
          { formula: 'Green: Agent generates minimal code to pass tests', explanation: 'The agent focuses on making tests pass, not on elegance.' },
          { formula: 'Refactor: Agent improves code quality while keeping tests green', explanation: 'With passing tests as a safety net, the agent can optimize.' },
          { formula: 'Iterate: Add more tests for edge cases and new features', explanation: 'The cycle repeats, building up coverage incrementally.' },
        ]}
        id="example-tdd-cycle"
      />

      <PythonCode
        title="tdd_agent.py"
        code={`import anthropic
import subprocess
import tempfile
import os

client = anthropic.Anthropic()

def generate_tests(spec: str) -> str:
    """Generate pytest tests from a specification."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=(
            "You are a test engineer. Write comprehensive pytest tests "
            "based on the specification. Include:\\n"
            "- Happy path tests\\n"
            "- Edge cases (empty input, None, large values)\\n"
            "- Error cases (invalid input, type errors)\\n"
            "Return ONLY the test code with imports."
        ),
        messages=[{"role": "user", "content": f"Specification:\\n{spec}"}]
    )
    code = response.content[0].text
    if code.startswith("\`\`\`"):
        code = "\\n".join(code.split("\\n")[1:-1])
    return code

def generate_implementation(spec: str, tests: str, error: str = "") -> str:
    """Generate implementation code to pass the tests."""
    prompt = f"Specification:\\n{spec}\\n\\nTests to pass:\\n{tests}"
    if error:
        prompt += f"\\n\\nPrevious attempt failed with:\\n{error}"
    prompt += "\\n\\nWrite the implementation. Return ONLY the code."

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    code = response.content[0].text
    if code.startswith("\`\`\`"):
        code = "\\n".join(code.split("\\n")[1:-1])
    return code

def run_tests(implementation: str, tests: str) -> dict:
    """Run the tests against the implementation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write implementation
        impl_path = os.path.join(tmpdir, "solution.py")
        with open(impl_path, "w") as f:
            f.write(implementation)

        # Write tests (importing from solution)
        test_path = os.path.join(tmpdir, "test_solution.py")
        with open(test_path, "w") as f:
            f.write(tests)

        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
            cwd=tmpdir,
        )
        return {
            "passed": result.returncode == 0,
            "output": result.stdout + result.stderr,
        }

def tdd_loop(spec: str, max_attempts: int = 4) -> dict:
    """Full TDD loop: generate tests, then iterate on implementation."""

    # Step 1: Generate tests (Red phase)
    print("Generating tests from specification...")
    tests = generate_tests(spec)
    print(f"Generated {tests.count('def test_')} test functions")

    # Step 2: Iterate on implementation (Green phase)
    error = ""
    for attempt in range(1, max_attempts + 1):
        print(f"\\nImplementation attempt {attempt}...")
        implementation = generate_implementation(spec, tests, error)

        result = run_tests(implementation, tests)
        print(f"Tests: {'PASSED' if result['passed'] else 'FAILED'}")

        if result["passed"]:
            print("All tests pass!")
            return {
                "implementation": implementation,
                "tests": tests,
                "attempts": attempt,
            }

        error = result["output"]
        print(f"Errors: {error[:200]}...")

    return {"implementation": implementation, "tests": tests, "attempts": max_attempts}

result = tdd_loop("""
Module: stack.py
Class: Stack

A generic stack data structure with the following methods:
- push(item): Add an item to the top of the stack
- pop(): Remove and return the top item. Raise IndexError if empty.
- peek(): Return the top item without removing. Raise IndexError if empty.
- is_empty(): Return True if the stack has no items.
- size(): Return the number of items in the stack.
- __contains__(item): Support 'in' operator for membership testing.
""")`}
        id="code-tdd-agent"
      />

      <h2 className="text-2xl font-semibold">Test Generation Strategies</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LLMs can generate tests from multiple sources: natural language specs,
        existing code (for regression tests), docstrings, or even by analyzing
        code paths and generating tests for each branch.
      </p>

      <PythonCode
        title="test_generation_strategies.py"
        code={`import anthropic

client = anthropic.Anthropic()

def generate_tests_from_code(source_code: str) -> str:
    """Generate tests by analyzing existing code."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": (
                f"Analyze this code and generate comprehensive pytest tests.\\n"
                f"Cover: all code paths, edge cases, error handling, "
                f"boundary values, and type edge cases.\\n\\n"
                f"Code:\\n{source_code}\\n\\n"
                f"Generate only the test file with imports."
            )
        }]
    )
    return response.content[0].text

def generate_property_tests(spec: str) -> str:
    """Generate property-based tests using Hypothesis."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Write property-based tests using the Hypothesis library "
                f"for the following specification. Focus on invariants "
                f"and properties that should always hold.\\n\\n"
                f"Spec: {spec}\\n\\n"
                f"Return only the test code with imports."
            )
        }]
    )
    return response.content[0].text

def generate_mutation_tests(source_code: str, tests: str) -> str:
    """Suggest improvements to tests by considering mutations."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Consider these mutations to the code that should be "
                f"caught by the tests but might not be:\\n"
                f"- Off-by-one errors\\n"
                f"- Swapped comparison operators\\n"
                f"- Removed boundary checks\\n\\n"
                f"Code:\\n{source_code}\\n\\n"
                f"Current tests:\\n{tests}\\n\\n"
                f"Suggest additional tests that would catch these mutations."
            )
        }]
    )
    return response.content[0].text`}
        id="code-test-strategies"
      />

      <NoteBlock
        type="intuition"
        title="Tests as Specifications"
        content="In LLM-assisted TDD, tests serve a dual purpose: they verify correctness and they communicate intent to the model. A well-written test suite is effectively a formal specification that the model can target. The more precise and comprehensive the tests, the better the generated code. This inverts the traditional complaint that 'tests are extra work' -- with LLMs, tests are the most valuable artifact you can write."
        id="note-tests-as-specs"
      />

      <WarningBlock
        title="LLM-Generated Tests May Be Wrong"
        content="Tests generated by an LLM can contain the same misconceptions as LLM-generated code. A test that encodes incorrect behavior will lead the agent to produce incorrect code that 'passes' the tests. Always review generated tests for correctness, especially the expected values in assertions. Consider having a separate model or human review the test suite."
        id="warning-wrong-tests"
      />

      <NoteBlock
        type="tip"
        title="Incremental TDD"
        content="Instead of generating all tests at once, generate tests incrementally: start with the simplest case, get it passing, then add complexity. This mirrors human TDD practice and gives the model smaller, more tractable problems to solve at each step."
        id="note-incremental-tdd"
      />
    </div>
  )
}
