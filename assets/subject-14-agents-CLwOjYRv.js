import{j as e}from"./vendor-DWbzdFaj.js";import{r}from"./vendor-katex-BYl39Yo6.js";import{D as s,P as t,E as a,N as n,W as o}from"./subject-01-text-fundamentals-DG6tAvii.js";function i(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Chain-of-Thought Prompting"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Chain-of-Thought (CoT) prompting encourages LLMs to break down complex problems into intermediate reasoning steps before arriving at a final answer. This simple technique dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks."}),e.jsx(s,{title:"Chain-of-Thought Prompting",definition:"A prompting strategy where the model is instructed to produce a sequence of intermediate reasoning steps, mimicking human step-by-step problem solving, before providing a final answer.",id:"def-cot"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Few-Shot Chain-of-Thought"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The original CoT technique by Wei et al. (2022) provides exemplars that include reasoning traces. The model learns to mimic this pattern for new queries."}),e.jsx(t,{title:"few_shot_cot.py",code:`import anthropic

client = anthropic.Anthropic()

few_shot_cot_prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step.
Roger started with 5 balls. He bought 2 cans of 3 balls each.
2 cans * 3 balls = 6 balls. 5 + 6 = 11 balls.
The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
bought 6 more, how many apples do they have?
A: Let's think step by step.
The cafeteria started with 23 apples. They used 20, so 23 - 20 = 3.
They bought 6 more, so 3 + 6 = 9 apples.
The answer is 9.

Q: A store has 48 shirts. They sell 1/3 of them on Monday,
then receive a shipment of 20. How many shirts do they have?
A: Let's think step by step.
"""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": few_shot_cot_prompt}]
)
print(response.content[0].text)
# The store started with 48 shirts. They sold 1/3 of 48 = 16 shirts.
# 48 - 16 = 32. Then received 20 more: 32 + 20 = 52.
# The answer is 52.`,id:"code-few-shot-cot"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Zero-Shot CoT"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:`Kojima et al. (2022) discovered that simply appending "Let's think step by step" to a prompt activates reasoning without any exemplars. This zero-shot approach is surprisingly effective across diverse tasks.`}),e.jsx(a,{title:"Zero-Shot vs Standard Prompting",problem:"Compare standard prompting to zero-shot CoT on a multi-step reasoning problem.",steps:[{formula:'Standard: "What is 17 * 24 + 13?"',explanation:"The model may guess or make arithmetic errors without showing work."},{formula:`CoT: "What is 17 * 24 + 13? Let's think step by step."`,explanation:"The model decomposes: 17 * 24 = 408, then 408 + 13 = 421."},{formula:"Accuracy improvement: 17.7% → 78.7% (MultiArith)",explanation:"Wei et al. showed CoT can improve arithmetic accuracy by 4x on some benchmarks."}],id:"example-zero-shot-cot"}),e.jsx(t,{title:"zero_shot_cot.py",code:`import anthropic

client = anthropic.Anthropic()

def solve_with_cot(question: str) -> str:
    """Use zero-shot CoT to solve a reasoning problem."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"{question}\\n\\nLet's think step by step."
        }]
    )
    return response.content[0].text

# Multi-step word problem
answer = solve_with_cot(
    "A train travels at 60 mph for 2.5 hours, then at 80 mph "
    "for 1.5 hours. What is the total distance traveled?"
)
print(answer)
# Step 1: Distance at 60 mph = 60 * 2.5 = 150 miles
# Step 2: Distance at 80 mph = 80 * 1.5 = 120 miles
# Step 3: Total = 150 + 120 = 270 miles`,id:"code-zero-shot-cot"}),e.jsx(n,{type:"intuition",title:"Why Does CoT Work?",content:"CoT works because it allocates more computation to harder problems. Each reasoning token generated gives the model additional forward passes to process information. In effect, CoT converts a single-step mapping into a multi-step computation, letting the model 'think aloud' through intermediate states.",id:"note-why-cot-works"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Self-Consistency"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Self-consistency (Wang et al., 2023) samples multiple CoT reasoning paths and takes the majority vote on the final answer. This ensemble approach further improves accuracy by reducing variance from any single reasoning chain."}),e.jsx(t,{title:"self_consistency.py",code:`import anthropic
from collections import Counter

client = anthropic.Anthropic()

def self_consistency_solve(question: str, n_samples: int = 5) -> str:
    """Sample multiple CoT paths and take majority vote."""
    answers = []
    for _ in range(n_samples):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.7,  # Higher temp for diverse reasoning paths
            messages=[{
                "role": "user",
                "content": (
                    f"{question}\\n\\nThink step by step, then provide "
                    f"your final answer on the last line as: ANSWER: <value>"
                )
            }]
        )
        text = response.content[0].text
        # Extract final answer
        for line in text.strip().split("\\n")[::-1]:
            if "ANSWER:" in line:
                answers.append(line.split("ANSWER:")[-1].strip())
                break

    # Majority vote
    vote = Counter(answers).most_common(1)[0]
    print(f"Votes: {Counter(answers)}")
    print(f"Consensus answer: {vote[0]} ({vote[1]}/{n_samples} votes)")
    return vote[0]

result = self_consistency_solve(
    "If a shirt costs $25 after a 20% discount, what was the original price?"
)
# Votes: Counter({'$31.25': 5})
# Consensus answer: $31.25 (5/5 votes)`,id:"code-self-consistency"}),e.jsx(o,{title:"CoT Limitations",content:"Chain-of-thought does not guarantee correct reasoning. Models can produce plausible-sounding but incorrect chains. CoT also increases token usage (and therefore cost and latency). For simple tasks, CoT may actually hurt performance by over-thinking. Always validate reasoning chains against known correct answers during development.",id:"warning-cot-limits"}),e.jsx(n,{type:"historical",title:"Key CoT Papers",content:"Wei et al. (2022) introduced few-shot CoT. Kojima et al. (2022) discovered zero-shot CoT with 'Let's think step by step.' Wang et al. (2023) proposed self-consistency decoding. These techniques form the foundation for all modern agent reasoning, including the extended thinking capabilities in Claude and the chain-of-thought reasoning in OpenAI's o1/o3 models.",id:"note-cot-history"})]})}const T=Object.freeze(Object.defineProperty({__proto__:null,default:i},Symbol.toStringTag,{value:"Module"}));function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"ReAct: Reasoning + Acting"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ReAct (Yao et al., 2023) interleaves reasoning traces with action execution, allowing an LLM to think about what to do, take an action in the environment, observe the result, and repeat. This Thought-Action-Observation loop is the foundational pattern behind most modern AI agents."}),e.jsx(s,{title:"ReAct Framework",definition:"ReAct synergizes reasoning (chain-of-thought) and acting (tool use) by generating interleaved Thought, Action, and Observation steps. The model reasons about what information it needs, takes actions to obtain it, then reasons about the results to decide next steps.",id:"def-react"}),e.jsx(a,{title:"ReAct Trace Example",problem:"Answer: 'What is the population of the capital of France?'",steps:[{formula:"Thought: I need to find the capital of France, then its population.",explanation:"The model decomposes the question into sub-tasks."},{formula:'Action: search("capital of France")',explanation:"The model invokes a search tool."},{formula:"Observation: Paris is the capital of France.",explanation:"The environment returns the search result."},{formula:"Thought: Now I need the population of Paris.",explanation:"The model reasons about what to do next."},{formula:'Action: search("population of Paris")',explanation:"Another search action is taken."},{formula:"Observation: Paris has a population of ~2.1 million.",explanation:"Result returned from the environment."}],id:"example-react-trace"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Implementing a ReAct Loop"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"A ReAct agent follows a simple loop: prompt the model with the current context, parse out any actions, execute them, append the observations, and repeat until the model produces a final answer."}),e.jsx(t,{title:"react_loop.py",code:`import anthropic
import json

client = anthropic.Anthropic()

# Define available tools
tools = [
    {
        "name": "search",
        "description": "Search the web for current information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]

def execute_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return the result."""
    if name == "search":
        # Simulated search results
        return f"Search results for '{inputs['query']}': [simulated result]"
    elif name == "calculator":
        try:
            result = eval(inputs["expression"])
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    return "Unknown tool"

def react_agent(question: str, max_steps: int = 5) -> str:
    """Run a ReAct agent loop."""
    messages = [{"role": "user", "content": question}]
    system = (
        "You are a helpful assistant. Use the provided tools to answer "
        "questions. Think step by step about what you need to find out."
    )

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Check if the model wants to use a tool
        if response.stop_reason == "tool_use":
            # Collect all tool uses and results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    print(f"Step {step+1}: {block.name}({block.input})")
                    print(f"  -> {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Model produced a final answer
            final = next(b.text for b in response.content if b.type == "text")
            print(f"Final answer: {final}")
            return final

    return "Max steps reached without answer"

answer = react_agent("What is 15% of the population of Tokyo?")`,id:"code-react-loop"}),e.jsx(n,{type:"intuition",title:"ReAct vs Pure CoT vs Pure Acting",content:"Pure chain-of-thought reasons without access to external information, leading to hallucinations on factual questions. Pure acting (taking tool actions without reasoning) makes suboptimal decisions. ReAct combines both: reasoning grounds the actions, and observations ground the reasoning. This synergy is why ReAct agents outperform either approach alone.",id:"note-react-vs-cot"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"ReAct with Claude's Native Tool Use"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Claude's API natively supports the ReAct pattern through its tool_use capability. The model automatically interleaves reasoning with tool calls, and you simply need to handle tool execution in your application code."}),e.jsx(t,{title:"react_claude_native.py",code:`import anthropic

client = anthropic.Anthropic()

# Claude handles ReAct natively through tool_use
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["city"]
    }
}]

def get_weather(city: str, units: str = "celsius") -> dict:
    """Simulated weather API."""
    return {"city": city, "temp": 22, "units": units, "condition": "sunny"}

# The conversation loop handles the ReAct pattern
messages = [{"role": "user", "content": "Should I bring an umbrella to Paris today?"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Claude's reasoning is implicit in its tool choice
                result = get_weather(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
        messages.append({"role": "user", "content": tool_results})
    else:
        # Final answer with reasoning
        print(next(b.text for b in response.content if b.type == "text"))
        break`,id:"code-react-native"}),e.jsx(o,{title:"Infinite Loops and Runaway Agents",content:"ReAct agents can get stuck in loops, repeatedly calling the same tool or oscillating between actions. Always set a maximum step count, implement cost budgets, and add loop detection. In production, log every step for debugging and set hard timeouts on the overall agent execution.",id:"warning-react-loops"}),e.jsx(n,{type:"historical",title:"From ReAct to Modern Agents",content:"ReAct (Yao et al., 2023) unified two previously separate lines of work: reasoning (CoT) and tool-augmented LMs (like Toolformer). The Thought-Action-Observation pattern became the de facto standard for agent architectures. LangChain, Claude's tool_use API, and OpenAI's function calling all implement variants of this pattern.",id:"note-react-history"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Task Decomposition and Plan-and-Execute"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Complex tasks often require breaking a high-level goal into smaller, manageable sub-tasks before execution. Plan-and-Execute agents separate the planning phase from the execution phase, leading to more structured and reliable agent behavior."}),e.jsx(s,{title:"Task Decomposition",definition:"The process of breaking a complex task into a sequence of simpler sub-tasks that can be individually solved. In agent systems, an LLM generates a plan (ordered list of steps), then executes each step, potentially replanning as new information emerges.",id:"def-task-decomposition"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Plan-and-Execute Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Unlike ReAct which interleaves thinking and acting at each step, Plan-and-Execute first creates a complete plan, then executes it step by step. This mirrors how humans approach complex projects: plan first, then do."}),e.jsx(a,{title:"Plan-and-Execute for Research",problem:"Task: Write a comparison of React vs Vue for a technical blog post.",steps:[{formula:"Plan Step 1: Research React features and recent updates",explanation:"Identify the sub-tasks needed to complete the goal."},{formula:"Plan Step 2: Research Vue features and recent updates",explanation:"Each step is specific and actionable."},{formula:"Plan Step 3: Identify key comparison dimensions",explanation:"Performance, ecosystem, learning curve, etc."},{formula:"Plan Step 4: Write the comparison with evidence",explanation:"Synthesize findings into the final output."},{formula:"Execute: Run each step sequentially, passing context forward",explanation:"Each step builds on the results of previous steps."}],id:"example-plan-execute"}),e.jsx(t,{title:"plan_and_execute.py",code:`import anthropic
import json

client = anthropic.Anthropic()

def create_plan(task: str) -> list[str]:
    """Use the LLM to decompose a task into steps."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Break this task into 3-6 concrete, sequential steps.\\n"
                f"Task: {task}\\n\\n"
                f"Return a JSON array of step descriptions.\\n"
                f"Example: [\\"step 1 description\\", \\"step 2 description\\"]"
            )
        }]
    )
    text = response.content[0].text
    # Extract JSON from response
    start = text.index("[")
    end = text.rindex("]") + 1
    return json.loads(text[start:end])

def execute_step(step: str, context: str) -> str:
    """Execute a single plan step with accumulated context."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Previous context:\\n{context}\\n\\n"
                f"Current step: {step}\\n\\n"
                f"Complete this step thoroughly."
            )
        }]
    )
    return response.content[0].text

def plan_and_execute(task: str) -> str:
    """Full plan-and-execute agent."""
    print(f"Task: {task}\\n")

    # Phase 1: Planning
    plan = create_plan(task)
    print("Plan:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    # Phase 2: Execution
    context = ""
    for i, step in enumerate(plan, 1):
        print(f"\\nExecuting step {i}/{len(plan)}: {step}")
        result = execute_step(step, context)
        context += f"\\n## Step {i}: {step}\\n{result}\\n"
        print(f"  Done. ({len(result)} chars)")

    return context

result = plan_and_execute(
    "Analyze the trade-offs between SQL and NoSQL databases "
    "for a real-time analytics application"
)`,id:"code-plan-execute"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Adaptive Replanning"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Static plans often need adjustment as execution reveals new information or unexpected obstacles. Adaptive agents can replan mid-execution, combining the structure of planning with the flexibility of reactive approaches."}),e.jsx(t,{title:"adaptive_replanning.py",code:`import anthropic
import json

client = anthropic.Anthropic()

def should_replan(step_result: str, remaining_plan: list[str], goal: str) -> list[str] | None:
    """Ask the LLM if the plan needs adjustment based on new information."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Goal: {goal}\\n"
                f"Latest step result: {step_result[:500]}\\n"
                f"Remaining plan: {json.dumps(remaining_plan)}\\n\\n"
                f"Should the remaining plan be adjusted? If yes, return a new "
                f"JSON array of steps. If no, return exactly: NO_CHANGE"
            )
        }]
    )
    text = response.content[0].text.strip()
    if "NO_CHANGE" in text:
        return None
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None

def adaptive_plan_execute(task: str, max_replans: int = 3) -> str:
    """Plan-and-execute with adaptive replanning."""
    plan = create_plan(task)  # From previous example
    context = ""
    replans = 0

    i = 0
    while i < len(plan):
        step = plan[i]
        print(f"\\nStep {i+1}/{len(plan)}: {step}")
        result = execute_step(step, context)
        context += f"\\nStep: {step}\\nResult: {result}\\n"

        # Check if replanning is needed
        remaining = plan[i+1:]
        if remaining and replans < max_replans:
            new_plan = should_replan(result, remaining, task)
            if new_plan is not None:
                print(f"  Replanning! New remaining steps: {new_plan}")
                plan = plan[:i+1] + new_plan
                replans += 1
        i += 1

    return context`,id:"code-adaptive-replan"}),e.jsx(n,{type:"tip",title:"Planning Granularity",content:"Plans that are too detailed become brittle and hard to adjust. Plans that are too vague provide no useful structure. Aim for 3-7 steps where each step represents a meaningful unit of work. Let the executor handle the details within each step.",id:"note-planning-granularity"}),e.jsx(o,{title:"Planning Hallucination",content:"LLMs can generate plans that sound reasonable but contain impossible or nonsensical steps. Always validate plans against the agent's actual capabilities (available tools, permissions, API access). A plan step like 'access the production database' is useless if the agent has no database tool.",id:"warning-plan-hallucination"}),e.jsx(n,{type:"historical",title:"Plan-and-Execute in AI History",content:"Hierarchical task planning has deep roots in classical AI (STRIPS, 1971; HTN planners). The Plan-and-Execute pattern in LLM agents was popularized by frameworks like LangChain's PlanAndExecute agent and BabyAGI. Modern agents like Devin and Claude Code use sophisticated variants that continuously replan based on execution feedback.",id:"note-planning-history"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Reflexion and Self-Critique"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Self-reflection enables agents to evaluate their own outputs, identify mistakes, and iteratively improve. Reflexion (Shinn et al., 2023) formalizes this as a loop where an agent attempts a task, reflects on its performance, and retries with accumulated lessons learned stored in memory."}),e.jsx(s,{title:"Reflexion",definition:"An agent architecture where the model generates an output, evaluates it against criteria (self-critique), produces a verbal reflection summarizing what went wrong, stores this reflection in memory, and uses it to improve on the next attempt.",id:"def-reflexion"}),e.jsx(a,{title:"Reflexion Loop",problem:"An agent is asked to write a Python function that passes unit tests but fails on the first attempt.",steps:[{formula:"Attempt 1: Generate function → Run tests → 2/5 pass",explanation:"Initial attempt has bugs."},{formula:'Reflect: "I missed edge cases for empty lists and negative numbers"',explanation:"The agent critiques its own output."},{formula:"Attempt 2: Generate improved function using reflection → 4/5 pass",explanation:"Reflection guides the improvement."},{formula:'Reflect: "Off-by-one error in the loop boundary"',explanation:"More specific critique on remaining failure."},{formula:"Attempt 3: Final fix → 5/5 pass",explanation:"Iterative refinement converges to a correct solution."}],id:"example-reflexion"}),e.jsx(t,{title:"reflexion_agent.py",code:`import anthropic

client = anthropic.Anthropic()

def generate_solution(task: str, reflections: list[str]) -> str:
    """Generate a solution, incorporating past reflections."""
    reflection_context = ""
    if reflections:
        reflection_context = "\\nPrevious reflections (learn from these):\\n"
        for i, r in enumerate(reflections, 1):
            reflection_context += f"{i}. {r}\\n"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {task}\\n"
                f"{reflection_context}\\n"
                f"Provide your solution:"
            )
        }]
    )
    return response.content[0].text

def evaluate_solution(task: str, solution: str) -> dict:
    """Evaluate a solution and return pass/fail with feedback."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {task}\\n"
                f"Solution:\\n{solution}\\n\\n"
                f"Evaluate this solution. Is it correct and complete?\\n"
                f"Respond with JSON: {{\\"passed\\": bool, \\"feedback\\": str}}"
            )
        }]
    )
    import json
    text = response.content[0].text
    start = text.index("{")
    end = text.rindex("}") + 1
    return json.loads(text[start:end])

def reflect(task: str, solution: str, feedback: str) -> str:
    """Generate a reflection on what went wrong."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {task}\\n"
                f"Your solution:\\n{solution}\\n"
                f"Feedback: {feedback}\\n\\n"
                f"Reflect on what went wrong and what to do differently. "
                f"Be specific and actionable in 2-3 sentences."
            )
        }]
    )
    return response.content[0].text

def reflexion_loop(task: str, max_attempts: int = 3) -> str:
    """Run the full Reflexion loop."""
    reflections = []

    for attempt in range(1, max_attempts + 1):
        print(f"\\n--- Attempt {attempt} ---")
        solution = generate_solution(task, reflections)
        print(f"Solution generated ({len(solution)} chars)")

        evaluation = evaluate_solution(task, solution)
        print(f"Evaluation: {'PASS' if evaluation['passed'] else 'FAIL'}")
        print(f"Feedback: {evaluation['feedback']}")

        if evaluation["passed"]:
            return solution

        # Reflect and store for next attempt
        reflection = reflect(task, solution, evaluation["feedback"])
        reflections.append(reflection)
        print(f"Reflection: {reflection}")

    return solution  # Return best effort

result = reflexion_loop(
    "Write a Python function that finds the longest palindromic "
    "substring in a given string. Handle edge cases."
)`,id:"code-reflexion"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Self-Critique Patterns"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Self-critique can be applied without the full Reflexion loop. A simple pattern is to generate output, then ask the model to critique it, then revise."}),e.jsx(t,{title:"self_critique.py",code:`import anthropic

client = anthropic.Anthropic()

def generate_and_critique(task: str) -> str:
    """Generate, critique, and revise in a single flow."""

    # Step 1: Initial generation
    draft = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": task}]
    ).content[0].text

    # Step 2: Self-critique
    critique = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Critique the following response for accuracy, completeness, "
                f"and clarity. List specific issues.\\n\\n"
                f"Task: {task}\\n\\nResponse:\\n{draft}"
            )
        }]
    ).content[0].text

    # Step 3: Revise based on critique
    revised = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Original task: {task}\\n\\n"
                f"Draft response:\\n{draft}\\n\\n"
                f"Critique:\\n{critique}\\n\\n"
                f"Provide an improved response addressing the critique."
            )
        }]
    ).content[0].text

    return revised

# Constitutional AI-style self-critique
def constitutional_critique(response: str, principles: list[str]) -> str:
    """Critique against explicit principles (Constitutional AI pattern)."""
    principles_text = "\\n".join(f"- {p}" for p in principles)

    critique = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Evaluate this response against these principles:\\n"
                f"{principles_text}\\n\\nResponse:\\n{response}\\n\\n"
                f"For each principle, note if it is satisfied or violated."
            )
        }]
    ).content[0].text
    return critique

principles = [
    "The response is factually accurate",
    "The response does not make unsupported claims",
    "The response acknowledges uncertainty where appropriate",
    "The response is helpful and actionable",
]`,id:"code-self-critique"}),e.jsx(n,{type:"intuition",title:"Why Self-Reflection Works",content:"LLMs are often better at evaluating solutions than generating them on the first try. This is analogous to how it is easier to spot a bug in code review than to write bug-free code. Self-reflection exploits this asymmetry by using the model's evaluation capability to guide its generation capability.",id:"note-reflection-intuition"}),e.jsx(o,{title:"Reflection Can Be Wrong",content:"Models can generate confident but incorrect reflections, leading subsequent attempts further astray. This is especially dangerous when the model lacks knowledge needed to evaluate correctness. Use external validation (tests, type checkers, search) rather than relying solely on self-assessment when possible.",id:"warning-bad-reflection"}),e.jsx(n,{type:"note",title:"Reflexion vs Fine-Tuning",content:"Reflexion stores lessons as text in the context window (episodic memory). This is distinct from fine-tuning, which updates model weights. Reflexion is immediate and requires no training, but lessons are lost when the context resets. For persistent improvement, combine Reflexion with a long-term memory store or use the reflections as training data.",id:"note-reflexion-vs-finetuning"})]})}const R=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Function and Tool Schemas"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Tool use allows LLMs to interact with external systems by calling functions with structured inputs. Tools are defined using JSON Schema, which tells the model what tools are available, what parameters they accept, and what each parameter means. Well-designed tool schemas are critical for reliable agent behavior."}),e.jsx(s,{title:"Tool Definition",definition:"A structured description of an external function that an LLM can invoke. It includes a name, a natural language description, and an input schema (typically JSON Schema) specifying the parameters, their types, constraints, and descriptions.",id:"def-tool-definition"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"JSON Schema for Tools"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Both Claude and OpenAI use JSON Schema to define tool parameters. The schema tells the model exactly what structure of input to produce when calling a tool."}),e.jsx(t,{title:"tool_schema_anatomy.py",code:`# Anatomy of a Claude tool definition
weather_tool = {
    "name": "get_weather",               # Unique identifier
    "description": (                       # Natural language - crucial for the model
        "Get the current weather conditions for a specified city. "
        "Returns temperature, humidity, and weather condition. "
        "Use this when the user asks about weather or needs to "
        "plan outdoor activities."
    ),
    "input_schema": {                      # JSON Schema object
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco' or 'London, UK'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units. Default: celsius"
            }
        },
        "required": ["city"]               # Only city is required
    }
}

# OpenAI function calling format (slightly different wrapper)
openai_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {                    # Same JSON Schema, different key name
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}`,id:"code-tool-schema"}),e.jsx(a,{title:"Designing Effective Tool Descriptions",problem:"How should you write tool descriptions so the model reliably chooses the right tool?",steps:[{formula:'Bad: "Database tool"',explanation:"Too vague. The model cannot determine when to use this."},{formula:'Better: "Query the SQL database"',explanation:"Clearer, but still lacks guidance on when to use it."},{formula:'Best: "Execute a read-only SQL query against the users database. Use when you need to look up user information like email, name, or account status. Returns up to 100 rows."',explanation:"Describes capability, use cases, and constraints."}],id:"example-tool-descriptions"}),e.jsx(t,{title:"complex_tool_schemas.py",code:`import anthropic

client = anthropic.Anthropic()

# Real-world tool definitions with rich schemas
tools = [
    {
        "name": "search_documents",
        "description": (
            "Search the knowledge base for relevant documents. "
            "Returns ranked results with snippets. Use for factual "
            "questions about company policies, products, or procedures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters to narrow results",
                    "properties": {
                        "department": {
                            "type": "string",
                            "enum": ["engineering", "sales", "hr", "legal"]
                        },
                        "date_after": {
                            "type": "string",
                            "description": "ISO date string, e.g. '2024-01-01'"
                        }
                    }
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (1-20)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_ticket",
        "description": (
            "Create a support ticket in the issue tracker. Use when "
            "the user reports a bug or requests a feature. Always "
            "confirm details with the user before creating."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Brief ticket title (max 100 chars)"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the issue"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"]
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels like 'bug', 'feature', 'docs'"
                }
            },
            "required": ["title", "description", "priority"]
        }
    }
]

# The model uses schemas to produce structured tool calls
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "Find documents about our vacation policy from HR"
    }]
)

for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")
        # Tool: search_documents
        # Input: {"query": "vacation policy", "filters": {"department": "hr"}}`,id:"code-complex-schemas"}),e.jsx(n,{type:"tip",title:"Schema Design Best Practices",content:"Use descriptive enum values instead of codes (e.g., 'high' not '3'). Add examples in descriptions for ambiguous fields. Keep required fields minimal to give the model flexibility. Use default values where sensible. Test your schemas with edge cases to ensure the model fills them correctly.",id:"note-schema-practices"}),e.jsx(o,{title:"Schema Complexity Limits",content:"Extremely complex schemas with deep nesting, many optional fields, or complex conditional logic can confuse the model. If a tool needs more than 5-7 parameters, consider splitting it into multiple simpler tools. Models are also more reliable with tools when there are fewer than 20 total tools defined.",id:"warning-schema-complexity"}),e.jsx(n,{type:"note",title:"Tool Schemas Across Providers",content:"Claude uses 'input_schema' with 'tools' parameter. OpenAI uses 'parameters' wrapped in a 'function' object. Google Gemini uses a similar format to OpenAI. Despite these syntactic differences, the underlying JSON Schema definitions are interchangeable. Libraries like LiteLLM abstract these differences.",id:"note-cross-provider"})]})}const L=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Parallel vs Sequential Tool Calls"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Modern LLMs can request multiple tool calls in a single response. Understanding when tools should be called in parallel versus sequentially is essential for building fast and correct agents. Parallel execution reduces latency while sequential execution handles data dependencies."}),e.jsx(s,{title:"Parallel Tool Calls",definition:"When a model requests multiple tool invocations in a single response turn, and those invocations have no data dependencies between them, they can be executed concurrently. This reduces total latency to the duration of the slowest call rather than the sum of all calls.",id:"def-parallel-tools"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Parallel Tool Execution"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Claude can return multiple tool_use blocks in a single response. When the tools are independent, you should execute them concurrently for better performance."}),e.jsx(t,{title:"parallel_tool_execution.py",code:`import anthropic
import asyncio
import time

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_time",
        "description": "Get current time in a timezone",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string"}
            },
            "required": ["timezone"]
        }
    }
]

def execute_tool(name: str, inputs: dict) -> str:
    """Simulate tool execution with latency."""
    time.sleep(1)  # Simulated API latency
    if name == "get_weather":
        return f"Weather in {inputs['city']}: 22C, sunny"
    elif name == "get_time":
        return f"Time in {inputs['timezone']}: 14:30"
    return "Unknown"

# Model may request multiple tools at once
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "What's the weather and time in Tokyo and London?"
    }]
)

# Collect all tool_use blocks
tool_calls = [b for b in response.content if b.type == "tool_use"]
print(f"Model requested {len(tool_calls)} tool calls")

# SEQUENTIAL execution (slow): ~4 seconds for 4 calls
start = time.time()
sequential_results = []
for tc in tool_calls:
    result = execute_tool(tc.name, tc.input)
    sequential_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
print(f"Sequential: {time.time() - start:.1f}s")

# PARALLEL execution (fast): ~1 second for 4 calls
import concurrent.futures

start = time.time()
parallel_results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(execute_tool, tc.name, tc.input): tc
        for tc in tool_calls
    }
    for future in concurrent.futures.as_completed(futures):
        tc = futures[future]
        result = future.result()
        parallel_results.append({
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": result
        })
print(f"Parallel: {time.time() - start:.1f}s")`,id:"code-parallel-tools"}),e.jsx(a,{title:"When to Use Parallel vs Sequential",problem:"Decide execution strategy for different tool call patterns.",steps:[{formula:'Parallel: get_weather("Tokyo") + get_weather("London")',explanation:"Independent calls with no data dependency. Safe to parallelize."},{formula:'Sequential: search("CEO of Acme") → get_profile(ceo_name)',explanation:"Second call depends on the result of the first. Must be sequential."},{formula:'Mixed: [get_weather("Tokyo") || get_time("Tokyo")] → format_report()',explanation:"First two are parallel, then the report depends on both results."}],id:"example-parallel-vs-sequential"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Async Tool Execution"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"For production agents handling I/O-bound tool calls (API requests, database queries), async execution with asyncio provides the best performance."}),e.jsx(t,{title:"async_tool_execution.py",code:`import anthropic
import asyncio

async_client = anthropic.AsyncAnthropic()

async def execute_tool_async(name: str, inputs: dict) -> str:
    """Async tool execution for I/O-bound operations."""
    await asyncio.sleep(1)  # Simulated async API call
    if name == "get_weather":
        return f"Weather in {inputs['city']}: 22C, sunny"
    elif name == "get_time":
        return f"Time in {inputs['timezone']}: 14:30"
    return "Unknown"

async def run_agent(question: str, tools: list, max_turns: int = 5):
    """Async agent loop with parallel tool execution."""
    messages = [{"role": "user", "content": question}]

    for turn in range(max_turns):
        response = await async_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            # Execute ALL tool calls in parallel
            tasks = [
                execute_tool_async(tc.name, tc.input)
                for tc in tool_calls
            ]
            results = await asyncio.gather(*tasks)

            # Build tool results in the correct order
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                }
                for tc, result in zip(tool_calls, results)
            ]

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            return next(b.text for b in response.content if b.type == "text")

    return "Max turns reached"

# Run the async agent
# result = asyncio.run(run_agent("Compare weather in 5 cities", tools))`,id:"code-async-tools"}),e.jsx(n,{type:"tip",title:"Encouraging Parallel Tool Calls",content:"Models sometimes make sequential tool calls when parallel would be more efficient. You can encourage parallel calls by adding to the system prompt: 'When multiple independent pieces of information are needed, request all relevant tool calls at once rather than one at a time.' Claude tends to naturally parallelize when it recognizes independence.",id:"note-encourage-parallel"}),e.jsx(o,{title:"Race Conditions in Parallel Execution",content:"If parallel tool calls modify shared state (e.g., two database writes to the same record), you may encounter race conditions. Only parallelize truly independent operations. For tools that modify state, consider whether the order of execution matters and add appropriate safeguards.",id:"warning-race-conditions"}),e.jsx(n,{type:"note",title:"Batching and Rate Limits",content:"When executing many tools in parallel, be mindful of API rate limits on the tool backends. Implement rate limiting, connection pooling, and exponential backoff. For very high parallelism (10+ concurrent calls), consider batching into groups to avoid overwhelming downstream services.",id:"note-rate-limits"})]})}const P=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Tool Error Recovery"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Tools fail. APIs timeout, databases go down, inputs are malformed, and permissions get denied. Robust agents must gracefully handle tool errors, retry when appropriate, and communicate failures clearly rather than crashing or hallucinating results."}),e.jsx(s,{title:"Tool Error Recovery",definition:"The set of strategies an agent uses when a tool call fails, including returning error information to the model, retrying with modified inputs, falling back to alternative tools, or gracefully degrading the response to inform the user of the limitation.",id:"def-error-recovery"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Returning Errors to the Model"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The most important principle: always return error information to the model rather than crashing. Claude can reason about errors and adapt its approach."}),e.jsx(t,{title:"error_returning.py",code:`import anthropic
import json
import traceback

client = anthropic.Anthropic()

def execute_tool_safely(name: str, inputs: dict) -> dict:
    """Execute a tool with comprehensive error handling."""
    try:
        if name == "query_database":
            result = query_db(inputs["sql"])
            return {"type": "tool_result", "content": json.dumps(result)}

        elif name == "fetch_url":
            result = fetch(inputs["url"])
            return {"type": "tool_result", "content": result}

        else:
            return {
                "type": "tool_result",
                "content": f"Error: Unknown tool '{name}'",
                "is_error": True  # Claude API supports this flag
            }

    except TimeoutError:
        return {
            "type": "tool_result",
            "content": (
                f"Error: Tool '{name}' timed out after 30 seconds. "
                f"The service may be temporarily unavailable. "
                f"You can retry or try a different approach."
            ),
            "is_error": True
        }
    except PermissionError as e:
        return {
            "type": "tool_result",
            "content": f"Error: Permission denied - {e}. You do not have access to this resource.",
            "is_error": True
        }
    except Exception as e:
        return {
            "type": "tool_result",
            "content": (
                f"Error executing {name}: {type(e).__name__}: {str(e)}. "
                f"Please try a different approach or inform the user."
            ),
            "is_error": True
        }

def agent_loop_with_errors(question: str, tools: list, max_turns: int = 10):
    """Agent loop that handles tool errors gracefully."""
    messages = [{"role": "user", "content": question}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool_safely(block.name, block.input)
                    result["tool_use_id"] = block.id
                    tool_results.append(result)
            messages.append({"role": "user", "content": tool_results})
        else:
            return next(b.text for b in response.content if b.type == "text")`,id:"code-error-returning"}),e.jsx(a,{title:"Error Recovery Strategies",problem:"What should an agent do when a tool call fails?",steps:[{formula:"Strategy 1: Retry with same inputs",explanation:"For transient errors like timeouts or rate limits. Add exponential backoff."},{formula:"Strategy 2: Retry with modified inputs",explanation:"If the error suggests bad input (e.g., invalid date format), fix and retry."},{formula:"Strategy 3: Fall back to alternative tool",explanation:"If web search fails, try a different search API or cached data."},{formula:"Strategy 4: Inform the user gracefully",explanation:"When no recovery is possible, explain what failed and what they can do."}],id:"example-recovery-strategies"}),e.jsx(t,{title:"retry_with_backoff.py",code:`import anthropic
import time
import random

client = anthropic.Anthropic()

def execute_with_retry(
    name: str,
    inputs: dict,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> dict:
    """Execute a tool with exponential backoff retry."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = execute_tool(name, inputs)  # Your tool executor
            return {"content": result}
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  Retry {attempt+1}/{max_retries} in {delay:.1f}s: {e}")
                time.sleep(delay)

    # All retries exhausted
    return {
        "content": (
            f"Error after {max_retries} retries: {last_error}. "
            f"The service appears to be down."
        ),
        "is_error": True
    }

def execute_with_fallback(name: str, inputs: dict, fallbacks: list) -> dict:
    """Try primary tool, then fall back to alternatives."""
    tools_to_try = [name] + fallbacks

    for tool_name in tools_to_try:
        try:
            result = execute_tool(tool_name, inputs)
            if tool_name != name:
                result = f"[via {tool_name}] {result}"
            return {"content": result}
        except Exception as e:
            print(f"  {tool_name} failed: {e}")
            continue

    return {
        "content": f"All tools failed: {name} and fallbacks {fallbacks}",
        "is_error": True
    }

# Example: search with fallback
result = execute_with_fallback(
    "web_search",
    {"query": "latest AI news"},
    fallbacks=["cached_search", "knowledge_base_search"]
)`,id:"code-retry-fallback"}),e.jsx(n,{type:"tip",title:"The is_error Flag",content:"Claude's API supports an 'is_error' field in tool results. When set to True, it signals to Claude that the tool call failed. This helps the model distinguish between an error message that should trigger recovery and a successful result that happens to contain the word 'error'.",id:"note-is-error-flag"}),e.jsx(o,{title:"Retry Loops and Cost",content:"Unbounded retries can lead to runaway API costs and infinite loops. Always set a maximum retry count and a total cost/time budget for the agent. Log all tool errors for monitoring. If a tool consistently fails, it may indicate a systemic issue rather than a transient error.",id:"warning-retry-cost"}),e.jsx(n,{type:"note",title:"Input Validation Before Execution",content:"Many tool errors can be prevented by validating inputs before execution. Validate types, check required fields, enforce constraints (string length, numeric ranges), and sanitize inputs to prevent injection attacks. This is especially important for tools that execute code or SQL queries.",id:"note-input-validation"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Building Custom Tools"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"While many agent frameworks provide built-in tools, real-world applications require custom tools tailored to your domain. Building effective custom tools involves defining clear schemas, implementing reliable execution, handling authentication, and testing tool behavior with your LLM."}),e.jsx(s,{title:"Custom Tool",definition:"A user-defined function exposed to an LLM agent through a tool schema. Custom tools bridge the gap between the model's reasoning capabilities and your application's specific APIs, databases, and business logic.",id:"def-custom-tool"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Tool Implementation Pattern"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"A well-structured custom tool separates the schema definition (what the model sees) from the implementation (what actually runs). This makes tools testable, reusable, and easy to modify."}),e.jsx(t,{title:"custom_tool_pattern.py",code:`import anthropic
import json
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class Tool:
    """A custom tool with schema and implementation."""
    name: str
    description: str
    input_schema: dict
    execute: Callable[[dict], str]

def make_tool(name: str, description: str, properties: dict,
              required: list[str], handler: Callable) -> Tool:
    """Factory for creating tools with consistent structure."""
    return Tool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        execute=handler,
    )

# --- Custom tool implementations ---

def handle_db_query(inputs: dict) -> str:
    """Execute a database query (simulated)."""
    sql = inputs["query"]
    # In production: validate SQL, use parameterized queries
    if "DROP" in sql.upper() or "DELETE" in sql.upper():
        return "Error: Destructive queries are not allowed"
    # Simulated result
    return json.dumps([
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
    ])

def handle_send_email(inputs: dict) -> str:
    """Send an email (simulated with confirmation)."""
    return (
        f"Email draft prepared:\\n"
        f"To: {inputs['to']}\\n"
        f"Subject: {inputs['subject']}\\n"
        f"Body: {inputs['body'][:100]}...\\n"
        f"Status: PENDING_CONFIRMATION (user must approve)"
    )

# --- Register tools ---

db_tool = make_tool(
    name="query_database",
    description=(
        "Execute a read-only SQL SELECT query against the application database. "
        "Tables: users (id, name, email, created_at), "
        "orders (id, user_id, amount, status, created_at). "
        "Use this to look up user or order information."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "SQL SELECT query to execute"
        }
    },
    required=["query"],
    handler=handle_db_query,
)

email_tool = make_tool(
    name="send_email",
    description=(
        "Compose and send an email. The email will be held for "
        "user confirmation before actually sending."
    ),
    properties={
        "to": {"type": "string", "description": "Recipient email address"},
        "subject": {"type": "string", "description": "Email subject line"},
        "body": {"type": "string", "description": "Email body text"},
    },
    required=["to", "subject", "body"],
    handler=handle_send_email,
)

TOOLS = [db_tool, email_tool]`,id:"code-custom-tool-pattern"}),e.jsx(t,{title:"tool_registry_agent.py",code:`import anthropic

client = anthropic.Anthropic()

class ToolRegistry:
    """Registry that maps tool names to their implementations."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        """Get all tool schemas for the Claude API."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def execute(self, name: str, inputs: dict) -> str:
        """Execute a tool by name with error handling."""
        if name not in self.tools:
            return f"Error: Unknown tool '{name}'"
        try:
            return self.tools[name].execute(inputs)
        except Exception as e:
            return f"Error executing {name}: {e}"

# Build the registry
registry = ToolRegistry()
for tool in TOOLS:  # From previous example
    registry.register(tool)

# Run agent with registered tools
def run_agent(question: str, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": question}]
    schemas = registry.get_schemas()

    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=schemas,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = registry.execute(block.name, block.input)
                    print(f"  {block.name} -> {output[:80]}...")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
            messages.append({"role": "user", "content": results})
        else:
            return next(b.text for b in response.content if b.type == "text")

    return "Max turns reached"

answer = run_agent("Find all users and send a welcome email to Alice")`,id:"code-tool-registry"}),e.jsx(a,{title:"MCP: Model Context Protocol",problem:"How do you share custom tools across different agents and applications?",steps:[{formula:"MCP Server: Exposes tools via a standard JSON-RPC protocol",explanation:"Any tool can be served as an MCP server, making it accessible to any MCP-compatible client."},{formula:"MCP Client: Connects to servers and discovers available tools",explanation:"Claude Desktop, Claude Code, and other clients auto-discover tools from MCP servers."},{formula:"Benefit: Write once, use everywhere",explanation:"A database tool written as an MCP server works in any agent framework that supports MCP."}],id:"example-mcp"}),e.jsx(n,{type:"tip",title:"Human-in-the-Loop for Dangerous Tools",content:"Tools that modify state (send emails, write to databases, make purchases) should always include a confirmation step. Return a 'pending confirmation' status and require explicit user approval before executing the action. This prevents costly mistakes from model errors.",id:"note-human-in-loop"}),e.jsx(o,{title:"Security: Tool Injection Attacks",content:"If tool inputs include user-generated content, an attacker could craft prompts that cause the model to misuse tools (e.g., SQL injection through tool parameters). Always sanitize and validate tool inputs server-side. Never trust the model's output as safe input for sensitive operations.",id:"warning-tool-injection"}),e.jsx(n,{type:"note",title:"Testing Custom Tools",content:"Test tools at three levels: (1) Unit test the handler functions with various inputs including edge cases. (2) Integration test the tool with the LLM by verifying the model produces valid inputs for common queries. (3) End-to-end test the full agent loop to ensure tools compose correctly when used together.",id:"note-testing-tools"})]})}const q=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LangChain Agents"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LangChain is one of the most widely used frameworks for building LLM applications and agents. It provides abstractions for models, tools, memory, and agent orchestration, along with LangGraph for building stateful, multi-step agent workflows as graphs."}),e.jsx(s,{title:"LangChain Agent",definition:"An autonomous system built with the LangChain framework that uses an LLM to decide which tools to call, in what order, and with what inputs. LangChain agents follow the ReAct pattern and support customizable tool sets, memory, and execution strategies.",id:"def-langchain-agent"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Basic Agent Setup"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LangChain provides a high-level interface for creating agents with tools. The framework handles the ReAct loop, tool execution, and message formatting."}),e.jsx(t,{title:"langchain_basic_agent.py",code:`from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Initialize the model
model = ChatAnthropic(model="claude-sonnet-4-20250514")

# Define tools using the @tool decorator
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python math syntax.
    Examples: '2 + 2', '(10 * 5) / 3', '2 ** 10'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_word_length(word: str) -> int:
    """Get the length of a word. Use when asked about word lengths."""
    return len(word)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for information.
    Use for questions about policies, procedures, or products."""
    # Simulated search
    return f"Found 3 results for '{query}': [result summaries...]"

# Create the agent using LangGraph's prebuilt ReAct agent
tools = [calculator, get_word_length, search_knowledge_base]
agent = create_react_agent(model, tools)

# Run the agent
result = agent.invoke({
    "messages": [("user", "What is 15% of 847, and how many letters are in 'extraordinary'?")]
})

# Print the conversation
for message in result["messages"]:
    print(f"{message.type}: {message.content[:100]}...")`,id:"code-langchain-basic"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"LangGraph: Stateful Agent Workflows"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LangGraph extends LangChain with graph-based agent orchestration. It lets you define agents as state machines with explicit control flow, making complex multi-step workflows predictable and debuggable."}),e.jsx(t,{title:"langgraph_workflow.py",code:`from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

model = ChatAnthropic(model="claude-sonnet-4-20250514")

@tool
def search(query: str) -> str:
    """Search for information on the web."""
    return f"Search results for '{query}': ..."

@tool
def write_report(content: str, title: str) -> str:
    """Write a structured report with the given content."""
    return f"Report '{title}' written successfully ({len(content)} chars)"

tools = [search, write_report]
model_with_tools = model.bind_tools(tools)
tool_node = ToolNode(tools)

# Define the agent logic
def should_continue(state: MessagesState):
    """Decide whether to use tools or finish."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    """Invoke the model with the current messages."""
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Build the graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        SystemMessage(content="You are a research assistant."),
        HumanMessage(content="Research AI agents and write a brief report"),
    ]
})`,id:"code-langgraph"}),e.jsx(a,{title:"LangChain vs Raw API",problem:"When should you use LangChain versus the raw Claude/OpenAI API?",steps:[{formula:"Use LangChain: Rapid prototyping, multiple LLM providers, complex chains",explanation:"LangChain shines when you need to swap models, compose chains, or use built-in tools."},{formula:"Use raw API: Production systems, fine control, minimal dependencies",explanation:"Direct API calls give you full control and avoid framework overhead."},{formula:"Use LangGraph: Complex multi-step workflows with branching logic",explanation:"When your agent has explicit states, conditions, and parallel branches."}],id:"example-langchain-vs-raw"}),e.jsx(n,{type:"tip",title:"LangSmith for Debugging",content:"LangSmith (LangChain's observability platform) traces every step of agent execution, including model calls, tool invocations, and intermediate states. This is invaluable for debugging agents that produce unexpected results. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable tracing.",id:"note-langsmith"}),e.jsx(o,{title:"Abstraction Trade-offs",content:"LangChain's abstractions can hide important details. When debugging, you may need to understand the exact prompts being sent, token counts, and API parameters. Over-reliance on high-level abstractions can make it harder to optimize performance and costs. Always be prepared to drop down to the raw API level when needed.",id:"warning-langchain-abstractions"}),e.jsx(n,{type:"note",title:"LangChain Ecosystem",content:"The LangChain ecosystem includes: langchain-core (base abstractions), langchain-community (integrations), langchain-anthropic/openai (provider packages), LangGraph (graph orchestration), LangSmith (observability), and LangServe (deployment). Start with the specific provider package you need rather than installing the full langchain package.",id:"note-ecosystem"})]})}const N=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Claude Agent SDK"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The Claude Agent SDK (part of Anthropic's official Python SDK) provides a lightweight, opinionated framework for building agents powered by Claude. It handles the agentic loop, tool execution, and multi-turn conversations with minimal boilerplate while staying close to the raw API."}),e.jsx(s,{title:"Claude Agent SDK",definition:"Anthropic's official agent framework built on top of the Claude API. It provides an Agent class that manages the ReAct loop, tool registration, guardrails, handoffs between agents, and structured output extraction.",id:"def-claude-agent-sdk"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Basic Agent Setup"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The SDK uses a declarative approach: define an Agent with instructions, tools, and configuration, then run it with a query."}),e.jsx(t,{title:"claude_agent_basic.py",code:`import anthropic
from agents import Agent, Runner, function_tool

# Define tools using the @function_tool decorator
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to check weather for.
    """
    # In production, call a real weather API
    return f"Weather in {city}: 22°C, partly cloudy, 45% humidity"

@function_tool
def search_web(query: str) -> str:
    """Search the web for current information.

    Args:
        query: The search query string.
    """
    return f"Top results for '{query}': [simulated results]"

# Create the agent
agent = Agent(
    name="Assistant",
    instructions=(
        "You are a helpful assistant that can check weather and "
        "search the web. Always provide concise, accurate answers. "
        "Use tools when you need current information."
    ),
    tools=[get_weather, search_web],
    model="claude-sonnet-4-20250514",
)

# Run the agent
result = Runner.run_sync(agent, "What's the weather like in Tokyo?")
print(result.final_output)
# The weather in Tokyo is 22°C, partly cloudy with 45% humidity.`,id:"code-agent-sdk-basic"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Agent Handoffs"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The SDK supports handoffs between specialized agents. A triage agent can route queries to domain-specific agents, each with their own tools and instructions. This enables modular agent architectures."}),e.jsx(t,{title:"agent_handoffs.py",code:`from agents import Agent, Runner, function_tool

# Specialized agents for different domains
@function_tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its ID."""
    return f"Order {order_id}: Shipped, arriving March 20"

@function_tool
def check_balance(account_id: str) -> str:
    """Check account balance."""
    return f"Account {account_id}: Balance $1,234.56"

@function_tool
def reset_password(email: str) -> str:
    """Send a password reset email."""
    return f"Password reset sent to {email}"

# Domain-specific agents
orders_agent = Agent(
    name="Orders Agent",
    instructions="You handle order-related queries. Look up orders and provide shipping info.",
    tools=[lookup_order],
    model="claude-sonnet-4-20250514",
)

billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle billing and account balance queries.",
    tools=[check_balance],
    model="claude-sonnet-4-20250514",
)

account_agent = Agent(
    name="Account Agent",
    instructions="You handle account management: password resets, profile updates.",
    tools=[reset_password],
    model="claude-sonnet-4-20250514",
)

# Triage agent that routes to specialists
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are the first point of contact. Determine the user's need "
        "and hand off to the appropriate specialist agent. "
        "Orders -> Orders Agent, Billing -> Billing Agent, "
        "Account issues -> Account Agent."
    ),
    handoffs=[orders_agent, billing_agent, account_agent],
    model="claude-sonnet-4-20250514",
)

# The triage agent routes automatically
result = Runner.run_sync(triage_agent, "Where is my order #12345?")
print(result.final_output)
# Routes to orders_agent, which looks up the order`,id:"code-agent-handoffs"}),e.jsx(a,{title:"SDK vs Raw API Agent Loop",problem:"What does the Agent SDK handle that you would otherwise write manually?",steps:[{formula:"Tool execution loop: automatic retry on tool_use stop reason",explanation:"The SDK manages the back-and-forth of tool calls and results."},{formula:"Tool schema generation: decorators auto-generate JSON Schema",explanation:"No need to manually write input_schema objects."},{formula:"Guardrails: input/output validation hooks",explanation:"The SDK supports pre/post processing validators."},{formula:"Handoffs: seamless routing between specialized agents",explanation:"Multi-agent orchestration built in."}],id:"example-sdk-benefits"}),e.jsx(t,{title:"agent_with_guardrails.py",code:`from agents import Agent, Runner, function_tool, InputGuardrail, GuardrailFunctionOutput

# Define a guardrail that checks for harmful requests
async def safety_check(ctx, agent, input_text):
    """Check if the input is appropriate."""
    # In production, use a classifier or content filter
    blocked_terms = ["hack", "exploit", "steal"]
    for term in blocked_terms:
        if term in input_text.lower():
            return GuardrailFunctionOutput(
                output_info={"blocked_term": term},
                tripwire_triggered=True,
            )
    return GuardrailFunctionOutput(
        output_info={"status": "safe"},
        tripwire_triggered=False,
    )

@function_tool
def execute_code(code: str) -> str:
    """Execute Python code in a sandbox.

    Args:
        code: Python code to execute safely.
    """
    # In production: use a sandboxed executor
    return f"Executed successfully. Output: [simulated]"

safe_agent = Agent(
    name="Safe Coding Agent",
    instructions="Help users write and test Python code.",
    tools=[execute_code],
    input_guardrails=[
        InputGuardrail(guardrail_function=safety_check),
    ],
    model="claude-sonnet-4-20250514",
)

# Safe request goes through
result = Runner.run_sync(safe_agent, "Write a function to sort a list")
print(result.final_output)

# Blocked request triggers guardrail
try:
    result = Runner.run_sync(safe_agent, "Help me hack a website")
except Exception as e:
    print(f"Guardrail triggered: {e}")`,id:"code-guardrails"}),e.jsx(n,{type:"tip",title:"Choosing Between Frameworks",content:"Use the Claude Agent SDK when building Claude-first applications that need a lightweight, well-integrated agent framework. Use LangChain/LangGraph when you need multi-provider support, complex graph workflows, or the extensive LangChain ecosystem of integrations. Use the raw API when you need maximum control and minimal dependencies.",id:"note-framework-choice"}),e.jsx(o,{title:"SDK Versioning",content:"Agent frameworks evolve rapidly. The Claude Agent SDK API may change between versions. Pin your dependency versions in production and review changelogs before upgrading. The patterns shown here reflect the SDK's architecture; consult the latest documentation for current API specifics.",id:"warning-sdk-versioning"})]})}const I=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"AutoGPT and BabyAGI Patterns"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Autonomous agents like AutoGPT and BabyAGI represented early attempts at fully self-directed LLM agents. These systems maintain persistent goals, create and prioritize tasks, and execute indefinitely with minimal human oversight. While often unreliable in practice, they introduced important architectural patterns still used in modern agents."}),e.jsx(s,{title:"Autonomous Agent",definition:"An LLM-powered system that operates with a high-level goal, autonomously generating sub-tasks, executing them using tools, maintaining memory across steps, and iterating until the goal is achieved or a stopping condition is met, with minimal or no human intervention between steps.",id:"def-autonomous-agent"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The BabyAGI Pattern"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"BabyAGI introduced a clean three-component architecture: a task creation agent, a task prioritization agent, and a task execution agent, all sharing a task queue."}),e.jsx(t,{title:"babyagi_pattern.py",code:`import anthropic
import json
from collections import deque

client = anthropic.Anthropic()

class BabyAGI:
    """Simplified BabyAGI-style autonomous agent."""

    def __init__(self, objective: str):
        self.objective = objective
        self.task_queue: deque[dict] = deque()
        self.completed_tasks: list[dict] = []
        self.task_id = 0

    def create_tasks(self, last_result: str) -> list[str]:
        """Generate new tasks based on the objective and recent results."""
        completed = [t["name"] for t in self.completed_tasks[-5:]]
        pending = [t["name"] for t in self.task_queue]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Objective: {self.objective}\\n"
                    f"Recently completed: {completed}\\n"
                    f"Pending tasks: {pending}\\n"
                    f"Last result: {last_result[:300]}\\n\\n"
                    f"Generate 1-3 NEW tasks that would help achieve the "
                    f"objective. Do not duplicate existing tasks.\\n"
                    f"Return as JSON array of strings."
                )
            }]
        )
        text = response.content[0].text
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

    def prioritize_tasks(self):
        """Reorder the task queue by priority."""
        if len(self.task_queue) <= 1:
            return
        tasks = [t["name"] for t in self.task_queue]
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"Objective: {self.objective}\\n"
                    f"Tasks: {json.dumps(tasks)}\\n\\n"
                    f"Reorder these tasks by priority (most important first). "
                    f"Return as JSON array of strings."
                )
            }]
        )
        text = response.content[0].text
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            ordered = json.loads(text[start:end])
            task_map = {t["name"]: t for t in self.task_queue}
            self.task_queue = deque(
                task_map[name] for name in ordered if name in task_map
            )
        except (ValueError, json.JSONDecodeError):
            pass

    def execute_task(self, task: dict) -> str:
        """Execute a single task."""
        context = "\\n".join(
            f"- {t['name']}: {t['result'][:100]}"
            for t in self.completed_tasks[-3:]
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": (
                    f"Objective: {self.objective}\\n"
                    f"Previous work:\\n{context}\\n\\n"
                    f"Current task: {task['name']}\\n"
                    f"Complete this task thoroughly."
                )
            }]
        )
        return response.content[0].text

    def run(self, initial_task: str, max_iterations: int = 5):
        """Run the autonomous agent loop."""
        self.task_id += 1
        self.task_queue.append({"id": self.task_id, "name": initial_task})

        for i in range(max_iterations):
            if not self.task_queue:
                print("All tasks completed!")
                break

            task = self.task_queue.popleft()
            print(f"\\n--- Iteration {i+1}: {task['name']} ---")

            result = self.execute_task(task)
            task["result"] = result
            self.completed_tasks.append(task)
            print(f"Result: {result[:150]}...")

            new_tasks = self.create_tasks(result)
            for name in new_tasks:
                self.task_id += 1
                self.task_queue.append({"id": self.task_id, "name": name})
                print(f"  New task: {name}")

            self.prioritize_tasks()

# agent = BabyAGI("Create a comprehensive guide to Python async programming")
# agent.run("Outline the main topics for async programming")`,id:"code-babyagi"}),e.jsx(a,{title:"AutoGPT Architecture",problem:"What are the key components of the AutoGPT pattern?",steps:[{formula:"Goal: A persistent high-level objective",explanation:"The agent works toward this goal across many iterations."},{formula:"Memory: Vector store for long-term recall",explanation:"Results and observations are embedded and stored for later retrieval."},{formula:"Tools: File I/O, web search, code execution, browsing",explanation:"A rich set of tools for interacting with the real world."},{formula:"Self-prompting: The agent generates its own next prompt",explanation:"Each loop iteration, the agent decides its own next action."}],id:"example-autogpt"}),e.jsx(n,{type:"intuition",title:"Why Autonomous Agents Often Fail",content:"Autonomous agents suffer from error compounding: each step has a probability of error, and over many steps these errors multiply. A 95% accurate step becomes only 60% reliable over 10 steps (0.95^10). Without human correction, agents drift off-course, get stuck in loops, or pursue irrelevant sub-tasks. This is why modern production agents favor human-in-the-loop designs.",id:"note-why-fail"}),e.jsx(o,{title:"Cost and Safety Risks",content:"Autonomous agents can consume large amounts of API tokens with no guarantee of useful output. AutoGPT-style agents have been observed spending hundreds of dollars on API calls while accomplishing nothing. Always set hard budget limits, step limits, and time limits. Never give autonomous agents access to production systems or real money without human approval gates.",id:"warning-autonomous-cost"}),e.jsx(n,{type:"historical",title:"The 2023 Autonomous Agent Wave",content:"AutoGPT (March 2023) became the fastest-growing GitHub repo at the time, capturing enormous public interest. BabyAGI followed shortly after with a cleaner architecture. While neither proved reliable for production use, they catalyzed research into agent architectures and demonstrated the potential of LLMs as autonomous reasoners. Modern agents like Claude Code and Cursor use more constrained versions of these patterns with human oversight.",id:"note-history"})]})}const M=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multi-Agent Collaboration"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Multi-agent systems use multiple specialized LLM agents that communicate and collaborate to solve complex tasks. Each agent has a distinct role, expertise, and set of tools. This mirrors how human teams work: a researcher gathers information, a writer produces content, a reviewer provides feedback."}),e.jsx(s,{title:"Multi-Agent System",definition:"An architecture where multiple LLM-powered agents with distinct roles, instructions, and tools collaborate on a shared task. Agents communicate by passing messages, sharing artifacts, or through a central orchestrator that coordinates their work.",id:"def-multi-agent"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Orchestrator Pattern"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The most common multi-agent pattern uses a central orchestrator agent that delegates sub-tasks to specialist agents and synthesizes their results."}),e.jsx(t,{title:"multi_agent_orchestrator.py",code:`import anthropic

client = anthropic.Anthropic()

class SpecialistAgent:
    """A specialist agent with a specific role and instructions."""

    def __init__(self, name: str, role: str, instructions: str):
        self.name = name
        self.role = role
        self.instructions = instructions

    def run(self, task: str, context: str = "") -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"You are {self.name}, a {self.role}. {self.instructions}",
            messages=[{
                "role": "user",
                "content": f"Context:\\n{context}\\n\\nTask: {task}" if context else task
            }]
        )
        return response.content[0].text

# Define specialist agents
researcher = SpecialistAgent(
    name="Research Agent",
    role="technical researcher",
    instructions=(
        "You gather and synthesize technical information. "
        "Provide detailed, factual analysis with specific examples."
    )
)

writer = SpecialistAgent(
    name="Writer Agent",
    role="technical writer",
    instructions=(
        "You write clear, engaging technical content. "
        "Transform research into well-structured prose."
    )
)

reviewer = SpecialistAgent(
    name="Reviewer Agent",
    role="critical reviewer",
    instructions=(
        "You review technical content for accuracy, clarity, and completeness. "
        "Provide specific, actionable feedback. Be constructive but thorough."
    )
)

class Orchestrator:
    """Coordinates multiple specialist agents."""

    def __init__(self, agents: dict[str, SpecialistAgent]):
        self.agents = agents

    def run_pipeline(self, task: str) -> dict:
        results = {}

        # Step 1: Research
        print("Phase 1: Research...")
        research = self.agents["researcher"].run(
            f"Research the following topic thoroughly: {task}"
        )
        results["research"] = research
        print(f"  Done ({len(research)} chars)")

        # Step 2: Writing
        print("Phase 2: Writing...")
        draft = self.agents["writer"].run(
            f"Write a comprehensive article about: {task}",
            context=f"Research findings:\\n{research}"
        )
        results["draft"] = draft
        print(f"  Done ({len(draft)} chars)")

        # Step 3: Review
        print("Phase 3: Review...")
        review = self.agents["reviewer"].run(
            "Review this article for accuracy and quality.",
            context=f"Article:\\n{draft}"
        )
        results["review"] = review
        print(f"  Done ({len(review)} chars)")

        # Step 4: Revision
        print("Phase 4: Revision...")
        final = self.agents["writer"].run(
            "Revise the article based on this feedback.",
            context=f"Original:\\n{draft}\\n\\nFeedback:\\n{review}"
        )
        results["final"] = final
        print(f"  Done ({len(final)} chars)")

        return results

orchestrator = Orchestrator({
    "researcher": researcher,
    "writer": writer,
    "reviewer": reviewer,
})
# results = orchestrator.run_pipeline("WebAssembly and its impact on web development")`,id:"code-orchestrator"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Debate Pattern"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"In the debate pattern, multiple agents with different perspectives argue a point, and a judge agent synthesizes the best arguments. This can improve reasoning quality by exploring multiple viewpoints."}),e.jsx(t,{title:"multi_agent_debate.py",code:`import anthropic

client = anthropic.Anthropic()

def debate(question: str, rounds: int = 2) -> str:
    """Multi-agent debate for exploring complex questions."""

    agents = [
        {"name": "Optimist", "bias": "Focus on benefits, opportunities, and positive outcomes."},
        {"name": "Skeptic", "bias": "Focus on risks, limitations, and potential problems."},
        {"name": "Pragmatist", "bias": "Focus on practical implications and trade-offs."},
    ]

    debate_history = []

    for round_num in range(1, rounds + 1):
        print(f"\\n=== Round {round_num} ===")
        round_arguments = []

        for agent in agents:
            context = ""
            if debate_history:
                context = "Previous arguments:\\n" + "\\n".join(
                    f"- {a['agent']}: {a['argument'][:200]}..."
                    for a in debate_history[-len(agents):]
                )

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=(
                    f"You are the {agent['name']}. {agent['bias']} "
                    f"Provide a concise, well-reasoned argument (150 words max). "
                    f"If responding to others, address their points directly."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Question: {question}\\n{context}\\n\\n"
                        f"Your argument (Round {round_num}):"
                    )
                }]
            )

            argument = response.content[0].text
            round_arguments.append({"agent": agent["name"], "argument": argument})
            print(f"\\n{agent['name']}: {argument[:150]}...")

        debate_history.extend(round_arguments)

    # Judge synthesizes
    all_arguments = "\\n\\n".join(
        f"{a['agent']}: {a['argument']}" for a in debate_history
    )

    synthesis = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are an impartial judge. Synthesize the debate into a balanced conclusion.",
        messages=[{
            "role": "user",
            "content": f"Question: {question}\\n\\nDebate:\\n{all_arguments}\\n\\nSynthesis:"
        }]
    )

    return synthesis.content[0].text

# result = debate("Should companies adopt AI coding assistants for all developers?")`,id:"code-debate"}),e.jsx(a,{title:"Multi-Agent Topologies",problem:"What are the common patterns for organizing multiple agents?",steps:[{formula:"Pipeline: Agent A → Agent B → Agent C",explanation:"Sequential chain where each agent processes the output of the previous one."},{formula:"Hub-and-spoke: Orchestrator delegates to specialists",explanation:"Central coordinator assigns tasks and collects results."},{formula:"Debate/consensus: Agents argue, judge decides",explanation:"Multiple perspectives improve quality on subjective or complex questions."},{formula:"Hierarchical: Manager agents supervise worker agents",explanation:"Multi-level delegation for complex projects with many sub-tasks."}],id:"example-topologies"}),e.jsx(n,{type:"tip",title:"When Multi-Agent Is Worth the Complexity",content:"Multi-agent systems add latency, cost, and complexity. Use them when: (1) the task genuinely requires different types of expertise, (2) quality improves measurably vs a single agent, (3) you need built-in checks and balances (reviewer agent catches writer errors). For most tasks, a well-prompted single agent with good tools outperforms a poorly designed multi-agent system.",id:"note-when-multi-agent"}),e.jsx(o,{title:"Coordination Overhead",content:"Multi-agent systems multiply API costs (each agent makes its own LLM calls) and introduce coordination challenges. Agents may produce contradictory outputs, lose context across handoffs, or enter infinite feedback loops (reviewer keeps finding issues, writer keeps revising). Set clear stopping conditions and monitor total token usage across all agents.",id:"warning-coordination"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Code Generation Quality"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LLM-powered code generation has evolved from simple autocomplete to producing entire functions, classes, and modules. Understanding what drives code quality in LLM outputs, how to measure it, and how to improve it through prompting techniques is essential for building effective coding agents."}),e.jsx(s,{title:"Code Generation",definition:"The task of producing syntactically correct, functionally accurate source code from a natural language specification or partial code context. Quality is measured across dimensions: correctness (passes tests), readability, efficiency, security, and adherence to conventions.",id:"def-code-generation"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Benchmarking Code Quality"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The standard benchmark for code generation is ",e.jsx(r.InlineMath,{math:"\\text{pass@k}"}),", which measures the probability that at least one of ",e.jsx(r.InlineMath,{math:"k"})," generated samples passes all unit tests. HumanEval and SWE-bench are widely used evaluation suites."]}),e.jsx(a,{title:"pass@k Metric",problem:"If a model generates 10 code samples and 3 pass all tests, what is pass@1 and pass@10?",steps:[{formula:"pass@k = 1 - \\binom{n-c}{k} / \\binom{n}{k}",explanation:"Where n=total samples, c=correct samples, k=samples considered."},{formula:"pass@1 = 1 - \\binom{7}{1}/\\binom{10}{1} = 1 - 7/10 = 0.3",explanation:"Expected probability that a single sample is correct: 30%."},{formula:"pass@10 = 1 - \\binom{7}{10}/\\binom{10}{10} = 1 - 0 = 1.0",explanation:"With 10 samples and 3 correct, at least one will always be correct."}],id:"example-pass-at-k"}),e.jsx(t,{title:"code_generation_prompting.py",code:`import anthropic

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
)`,id:"code-generation-prompting"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Iterative Refinement"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The best coding agents do not rely on a single generation pass. They generate, test, analyze failures, and iterate. This feedback loop dramatically improves the final code quality."}),e.jsx(t,{title:"iterative_code_refinement.py",code:`import anthropic
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
)`,id:"code-iterative-refinement"}),e.jsx(n,{type:"tip",title:"Prompting for Better Code",content:"Specify the programming language, style guide, and patterns you want. Provide type signatures or interfaces upfront. Include edge cases in your specification. Ask for error handling explicitly. Request that the model think through the algorithm before coding (CoT for code). These simple techniques significantly improve first-pass quality.",id:"note-better-prompts"}),e.jsx(o,{title:"Code Execution Security",content:"Never execute LLM-generated code in an unsandboxed environment. Generated code may contain bugs that corrupt data, infinite loops that consume resources, or (in adversarial settings) malicious operations. Always use sandboxed execution: Docker containers, E2B sandboxes, or subprocess with strict resource limits and no network access.",id:"warning-code-security"}),e.jsx(n,{type:"note",title:"SWE-bench: Real-World Code Evaluation",content:"SWE-bench tests agents on real GitHub issues from popular open-source projects. Unlike HumanEval's isolated function problems, SWE-bench requires understanding existing codebases, navigating multiple files, and producing patches that pass existing test suites. As of early 2025, top agents solve roughly 50% of SWE-bench verified instances.",id:"note-swe-bench"})]})}const D=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Codebase Navigation and Understanding"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Real-world coding tasks rarely involve writing code from scratch. More often, an agent must understand an existing codebase, find relevant files, trace dependencies, and make targeted changes. Codebase navigation is the foundation of effective coding agents."}),e.jsx(s,{title:"Codebase Navigation",definition:"The ability of a coding agent to explore, search, and understand the structure, conventions, and dependencies of an existing software repository. This includes finding relevant files, understanding module relationships, identifying patterns, and locating the right place to make changes.",id:"def-repo-navigation"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Search-Based Navigation"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Coding agents navigate repositories primarily through search tools: grep for content, find/glob for files, and AST analysis for structure. The quality of an agent's search strategy determines how quickly it finds relevant code."}),e.jsx(t,{title:"repo_navigation_tools.py",code:`import os
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
# nav.get_definitions("src/auth/handlers.py")`,id:"code-repo-navigation"}),e.jsx(a,{title:"Navigation Strategy for Bug Fixing",problem:"Agent receives: 'Fix the login timeout bug reported in issue #234'",steps:[{formula:'Step 1: Search for "login" and "timeout" in the codebase',explanation:"Cast a wide net to find relevant files."},{formula:"Step 2: Identify the authentication module structure",explanation:"Read directory tree around auth-related files."},{formula:"Step 3: Read the login handler and timeout configuration",explanation:"Focus on the most likely locations for the bug."},{formula:"Step 4: Trace the call chain from login to session creation",explanation:"Follow the code path to find where timeout is set."},{formula:"Step 5: Read related tests to understand expected behavior",explanation:"Tests reveal the intended behavior and edge cases."}],id:"example-nav-strategy"}),e.jsx(t,{title:"agent_codebase_exploration.py",code:`import anthropic

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

# The agent would iterate: search -> read -> search -> read -> summarize`,id:"code-agent-exploration"}),e.jsx(n,{type:"intuition",title:"Context Window as Working Memory",content:"A coding agent's context window is its working memory. It can only reason about code it has read into context. Effective agents are strategic about what they read: they start with high-level structure (directory listing, file outlines), then drill into specific files based on relevance. Poor agents read files sequentially and run out of context before finding the relevant code.",id:"note-context-window"}),e.jsx(o,{title:"Large Codebases Exceed Context",content:"Production repositories can have millions of lines of code. No LLM can fit an entire codebase in its context window. Agents must use search tools strategically, read only relevant sections, and maintain a mental model of the codebase structure. Retrieval-augmented approaches (embedding the codebase and searching semantically) help but are not a complete solution.",id:"warning-large-codebases"}),e.jsx(n,{type:"tip",title:"Indexing for Faster Navigation",content:"Pre-index the codebase with tools like tree-sitter (for AST parsing), ctags (for symbol definitions), or embedding-based search (for semantic similarity). An indexed codebase lets the agent find relevant code in milliseconds rather than running grep over the entire repository for each search.",id:"note-indexing"})]})}const G=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Test-Driven Development with LLMs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Test-driven development (TDD) pairs naturally with LLM coding agents. Tests provide an objective, automated way to verify generated code. The TDD loop (write tests, generate code, run tests, fix failures) gives the agent concrete feedback at each step, dramatically improving code reliability."}),e.jsx(s,{title:"LLM-Assisted TDD",definition:"A development workflow where an LLM agent participates in the test-driven development cycle: either generating tests from specifications, generating code to pass existing tests, or both. The test suite serves as an automated oracle that guides the agent toward correct implementations.",id:"def-tdd-llm"}),e.jsx(a,{title:"TDD Cycle with an LLM Agent",problem:"Build a URL shortener service using TDD with an LLM.",steps:[{formula:"Red: Agent generates tests from the specification",explanation:"Tests define the expected behavior before any implementation exists."},{formula:"Green: Agent generates minimal code to pass tests",explanation:"The agent focuses on making tests pass, not on elegance."},{formula:"Refactor: Agent improves code quality while keeping tests green",explanation:"With passing tests as a safety net, the agent can optimize."},{formula:"Iterate: Add more tests for edge cases and new features",explanation:"The cycle repeats, building up coverage incrementally."}],id:"example-tdd-cycle"}),e.jsx(t,{title:"tdd_agent.py",code:`import anthropic
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
""")`,id:"code-tdd-agent"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Test Generation Strategies"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LLMs can generate tests from multiple sources: natural language specs, existing code (for regression tests), docstrings, or even by analyzing code paths and generating tests for each branch."}),e.jsx(t,{title:"test_generation_strategies.py",code:`import anthropic

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
    return response.content[0].text`,id:"code-test-strategies"}),e.jsx(n,{type:"intuition",title:"Tests as Specifications",content:"In LLM-assisted TDD, tests serve a dual purpose: they verify correctness and they communicate intent to the model. A well-written test suite is effectively a formal specification that the model can target. The more precise and comprehensive the tests, the better the generated code. This inverts the traditional complaint that 'tests are extra work' -- with LLMs, tests are the most valuable artifact you can write.",id:"note-tests-as-specs"}),e.jsx(o,{title:"LLM-Generated Tests May Be Wrong",content:"Tests generated by an LLM can contain the same misconceptions as LLM-generated code. A test that encodes incorrect behavior will lead the agent to produce incorrect code that 'passes' the tests. Always review generated tests for correctness, especially the expected values in assertions. Consider having a separate model or human review the test suite.",id:"warning-wrong-tests"}),e.jsx(n,{type:"tip",title:"Incremental TDD",content:"Instead of generating all tests at once, generate tests incrementally: start with the simplest case, get it passing, then add complexity. This mirrors human TDD practice and gives the model smaller, more tractable problems to solve at each step.",id:"note-incremental-tdd"})]})}const W=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Claude Code Capabilities"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Claude Code is Anthropic's agentic coding tool that operates directly in your terminal. It combines Claude's reasoning with the ability to read files, write code, run commands, search codebases, and interact with git. It represents a practical implementation of the agent patterns covered throughout this subject."}),e.jsx(s,{title:"Claude Code",definition:"An agentic coding assistant that runs as a CLI tool in your terminal. It can read and edit files, execute shell commands, search codebases, run tests, create commits, and manage git workflows. It operates in a ReAct-style loop with human-in-the-loop approval for potentially dangerous operations.",id:"def-claude-code"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture and Tool Set"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Claude Code implements the agent patterns we have studied: a ReAct loop with specialized tools for coding tasks, self-reflection for quality, and plan-and-execute for complex multi-step changes."}),e.jsx(a,{title:"Claude Code's Tool Set",problem:"What tools does Claude Code use internally to accomplish coding tasks?",steps:[{formula:"Read: Read files with line numbers and ranges",explanation:"Equivalent to cat -n, supports reading specific line ranges for large files."},{formula:"Edit: Make targeted string replacements in files",explanation:"Precise edits rather than rewriting entire files, reducing errors."},{formula:"Bash: Execute shell commands (git, npm, python, etc.)",explanation:"Full access to the development environment with user approval."},{formula:"Grep: Search file contents with regex patterns",explanation:"Fast codebase search powered by ripgrep."},{formula:"Glob: Find files by name pattern",explanation:"Locate files across the project structure."},{formula:"Write: Create new files or complete rewrites",explanation:"For new files that do not exist yet."}],id:"example-claude-code-tools"}),e.jsx(t,{title:"claude_code_workflow.py",code:`# Claude Code workflow: what happens under the hood
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
# This fixes a timezone bug with a single-line change`,id:"code-claude-code-workflow"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Effective Usage Patterns"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Getting the most out of Claude Code involves understanding how to give it context, structure requests, and leverage its agentic capabilities."}),e.jsx(t,{title:"effective_claude_code_usage.py",code:`# Patterns for effective Claude Code usage

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
# claude -p "Run the test suite and fix any failures" --allowedTools Edit,Bash,Read`,id:"code-effective-usage"}),e.jsx(n,{type:"tip",title:"CLAUDE.md for Project Context",content:"Create a CLAUDE.md file at the root of your repository with project architecture, conventions, common commands, and important context. Claude Code reads this file automatically and uses it to make better decisions about code style, project structure, and tooling. This is far more effective than repeating context in every prompt.",id:"note-claude-md"}),e.jsx(n,{type:"note",title:"Safety Model: Human-in-the-Loop",content:"Claude Code implements a layered safety model. Read-only operations (searching, reading files) run automatically. Write operations (editing files) proceed with notification. Potentially dangerous operations (running arbitrary shell commands, git push) require explicit user approval. This balances productivity with safety, following the principle of least privilege.",id:"note-safety-model"}),e.jsx(o,{title:"Agent Limitations",content:"Even the best coding agents have limitations. They can struggle with: large-scale architectural refactors spanning dozens of files, deeply domain-specific logic requiring expert knowledge, performance optimization requiring profiling data, and security-sensitive code where subtle bugs have outsized impact. Use coding agents as powerful assistants, not as replacements for engineering judgment.",id:"warning-limitations"}),e.jsx(n,{type:"historical",title:"Evolution of Coding Agents",content:"Coding assistants evolved from autocomplete (GitHub Copilot, 2021) to chat-based helpers (ChatGPT, 2022) to agentic tools (Devin, Claude Code, Cursor, 2024-2025). Each generation added more autonomy: autocomplete suggests lines, chat produces functions, and agents navigate repos, run tests, and create PRs. The trend is toward greater autonomy with better safety guardrails.",id:"note-evolution"})]})}const F=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));export{S as a,C as b,R as c,L as d,P as e,E as f,q as g,N as h,I as i,M as j,O as k,D as l,G as m,W as n,F as o,T as s};
