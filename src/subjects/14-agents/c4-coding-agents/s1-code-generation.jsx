import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CodeGeneration() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Code Generation Quality</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LLM-powered code generation has evolved from simple autocomplete to producing
        entire functions, classes, and modules. Understanding what drives code quality
        in LLM outputs, how to measure it, and how to improve it through prompting
        techniques is essential for building effective coding agents.
      </p>

      <DefinitionBlock
        title="Code Generation"
        definition="The task of producing syntactically correct, functionally accurate source code from a natural language specification or partial code context. Quality is measured across dimensions: correctness (passes tests), readability, efficiency, security, and adherence to conventions."
        id="def-code-generation"
      />

      <h2 className="text-2xl font-semibold">Benchmarking Code Quality</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The standard benchmark for code generation is <InlineMath math="\text{pass@k}" />,
        which measures the probability that at least one of <InlineMath math="k" /> generated
        samples passes all unit tests. HumanEval and SWE-bench are widely used evaluation suites.
      </p>

      <ExampleBlock
        title="pass@k Metric"
        problem="If a model generates 10 code samples and 3 pass all tests, what is pass@1 and pass@10?"
        steps={[
          { formula: 'pass@k = 1 - \\binom{n-c}{k} / \\binom{n}{k}', explanation: 'Where n=total samples, c=correct samples, k=samples considered.' },
          { formula: 'pass@1 = 1 - \\binom{7}{1}/\\binom{10}{1} = 1 - 7/10 = 0.3', explanation: 'Expected probability that a single sample is correct: 30%.' },
          { formula: 'pass@10 = 1 - \\binom{7}{10}/\\binom{10}{10} = 1 - 0 = 1.0', explanation: 'With 10 samples and 3 correct, at least one will always be correct.' },
        ]}
        id="example-pass-at-k"
      />

      <PythonCode
        title="code_generation_prompting.py"
        code={`import anthropic

client = anthropic.Anthropic()

def generate_code(spec: str, language: str = "python") -> str:
    """Generate high-quality code from a specification."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=(
            f"You are an expert {language} developer. Generate clean, "
            f"well-documented, production-quality code. Include:\\n"
            f"- Type hints and docstrings\\n"
            f"- Error handling for edge cases\\n"
            f"- Efficient algorithms\\n"
            f"- Follow PEP 8 conventions\\n"
            f"Return ONLY the code, no explanations."
        ),
        messages=[{"role": "user", "content": spec}]
    )
    return response.content[0].text

# Simple generation
code = generate_code(
    "Write a function that finds all prime numbers up to n "
    "using the Sieve of Eratosthenes algorithm."
)
print(code)

# Structured generation with examples
def generate_with_examples(spec: str, examples: list[dict]) -> str:
    """Generate code with input/output examples for clarity."""
    examples_text = "\\n".join(
        f"  Input: {ex['input']} -> Output: {ex['output']}"
        for ex in examples
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": (
                f"Write a Python function for:\\n{spec}\\n\\n"
                f"Examples:\\n{examples_text}\\n\\n"
                f"Include comprehensive error handling and type hints."
            )
        }]
    )
    return response.content[0].text

code = generate_with_examples(
    "Convert a Roman numeral string to an integer",
    examples=[
        {"input": "'III'", "output": "3"},
        {"input": "'MCMXCIV'", "output": "1994"},
        {"input": "'XLII'", "output": "42"},
    ]
)`}
        id="code-generation-prompting"
      />

      <h2 className="text-2xl font-semibold">Iterative Refinement</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The best coding agents do not rely on a single generation pass. They generate,
        test, analyze failures, and iterate. This feedback loop dramatically improves
        the final code quality.
      </p>

      <PythonCode
        title="iterative_code_refinement.py"
        code={`import anthropic
import subprocess
import tempfile
import os

client = anthropic.Anthropic()

def run_python_code(code: str, test_code: str = "") -> dict:
    """Execute Python code and return results."""
    full_code = code + "\\n\\n" + test_code if test_code else code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True, text=True, timeout=10
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Timeout"}
        finally:
            os.unlink(f.name)

def iterative_generate(spec: str, tests: str, max_attempts: int = 3) -> str:
    """Generate code iteratively until tests pass."""
    code = ""
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            prompt = f"Write Python code for: {spec}"
        else:
            prompt = (
                f"The previous code failed tests.\\n"
                f"Code:\\n{code}\\n\\n"
                f"Error:\\n{result['stderr'][:500]}\\n\\n"
                f"Fix the code. Return only the corrected code."
            )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        code = response.content[0].text

        # Strip markdown code fences if present
        if code.startswith("\`\`\`"):
            code = "\\n".join(code.split("\\n")[1:-1])

        result = run_python_code(code, tests)
        print(f"Attempt {attempt}: {'PASS' if result['success'] else 'FAIL'}")

        if result["success"]:
            return code

    return code  # Return best effort

final_code = iterative_generate(
    spec="A function 'merge_sorted' that merges two sorted lists into one sorted list",
    tests="""
assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]
assert merge_sorted([], [1,2]) == [1,2]
assert merge_sorted([1], []) == [1]
assert merge_sorted([], []) == []
assert merge_sorted([1,1], [1,1]) == [1,1,1,1]
print("All tests passed!")
"""
)`}
        id="code-iterative-refinement"
      />

      <NoteBlock
        type="tip"
        title="Prompting for Better Code"
        content="Specify the programming language, style guide, and patterns you want. Provide type signatures or interfaces upfront. Include edge cases in your specification. Ask for error handling explicitly. Request that the model think through the algorithm before coding (CoT for code). These simple techniques significantly improve first-pass quality."
        id="note-better-prompts"
      />

      <WarningBlock
        title="Code Execution Security"
        content="Never execute LLM-generated code in an unsandboxed environment. Generated code may contain bugs that corrupt data, infinite loops that consume resources, or (in adversarial settings) malicious operations. Always use sandboxed execution: Docker containers, E2B sandboxes, or subprocess with strict resource limits and no network access."
        id="warning-code-security"
      />

      <NoteBlock
        type="note"
        title="SWE-bench: Real-World Code Evaluation"
        content="SWE-bench tests agents on real GitHub issues from popular open-source projects. Unlike HumanEval's isolated function problems, SWE-bench requires understanding existing codebases, navigating multiple files, and producing patches that pass existing test suites. As of early 2025, top agents solve roughly 50% of SWE-bench verified instances."
        id="note-swe-bench"
      />
    </div>
  )
}
