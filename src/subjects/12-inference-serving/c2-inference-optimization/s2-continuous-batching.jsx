import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ContinuousBatching() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Continuous Batching</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Static batching wastes GPU cycles waiting for the longest sequence in a batch to finish.
        Continuous (or dynamic) batching allows new requests to join and completed requests to leave
        the batch at every iteration step, maximizing GPU utilization.
      </p>

      <DefinitionBlock
        title="Static Batching"
        definition="In static batching, a batch of requests starts together and the GPU waits for all sequences to reach their max length or EOS token. If one request generates 10 tokens and another generates 500, the GPU is idle for the short request during 490 steps."
        id="def-static-batching"
      />

      <DefinitionBlock
        title="Continuous Batching"
        definition="Continuous batching (iteration-level scheduling) checks after every forward pass whether any sequence has finished. Completed sequences are evicted and new requests from the queue are inserted immediately. The batch size fluctuates dynamically."
        id="def-continuous-batching"
      />

      <ExampleBlock
        title="Throughput Comparison"
        problem="Compare static vs continuous batching with 4 requests: lengths 20, 50, 100, 200 tokens."
        steps={[
          { formula: 'Static: all wait for 200 steps = 800 total slot-steps', explanation: 'Each of 4 slots runs for 200 steps regardless of actual output length.' },
          { formula: 'Useful work: 20+50+100+200 = 370 slot-steps', explanation: 'Only 370/800 = 46% GPU utilization.' },
          { formula: 'Continuous: slot freed at step 20, new request enters', explanation: 'As soon as the 20-token request finishes, a new request fills its slot.' },
          { formula: 'Continuous achieves 85-95% utilization', explanation: 'The GPU is almost always doing useful work.' },
        ]}
        id="example-throughput"
      />

      <PythonCode
        title="continuous_batching_sim.py"
        code={`import random
import time
from collections import deque

class Request:
    def __init__(self, id, prompt_len, output_len):
        self.id = id
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.tokens_generated = 0
        self.start_time = None

    @property
    def done(self):
        return self.tokens_generated >= self.output_len

def simulate_batching(requests, max_batch_size, mode="continuous"):
    queue = deque(requests)
    batch = []
    step = 0
    completed = []

    while queue or batch:
        # Fill batch from queue
        while len(batch) < max_batch_size and queue:
            req = queue.popleft()
            req.start_time = step
            batch.append(req)

        # One forward pass -- generate one token per request
        step += 1
        for req in batch:
            req.tokens_generated += 1

        if mode == "continuous":
            # Remove finished requests immediately
            finished = [r for r in batch if r.done]
            batch = [r for r in batch if not r.done]
            for r in finished:
                r.end_time = step
                completed.append(r)
        else:  # static
            if all(r.done for r in batch):
                for r in batch:
                    r.end_time = step
                    completed.extend(batch)
                batch = []

    return completed, step

# Generate 20 requests with varying output lengths
random.seed(42)
reqs_cont = [Request(i, 10, random.randint(10, 200)) for i in range(20)]
reqs_stat = [Request(i, 10, reqs_cont[i].output_len) for i in range(20)]

completed_c, steps_c = simulate_batching(reqs_cont, max_batch_size=4, mode="continuous")
completed_s, steps_s = simulate_batching(reqs_stat, max_batch_size=4, mode="static")

avg_latency_c = sum(r.end_time - r.start_time for r in completed_c) / len(completed_c)
avg_latency_s = sum(r.end_time - r.start_time for r in completed_s) / len(completed_s)

print(f"Static batching:     {steps_s} steps, avg latency {avg_latency_s:.0f}")
print(f"Continuous batching: {steps_c} steps, avg latency {avg_latency_c:.0f}")
print(f"Throughput improvement: {steps_s/steps_c:.2f}x")`}
        id="code-simulation"
      />

      <PythonCode
        title="Terminal"
        code={`# vLLM uses continuous batching by default
# Start with specific batch settings:
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --max-num-seqs 256 \\
    --max-num-batched-tokens 8192

# TGI also supports continuous batching:
docker run --gpus all -p 8080:80 \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.1-8B-Instruct \\
    --max-batch-total-tokens 8192 \\
    --max-concurrent-requests 128`}
        id="code-frameworks"
      />

      <NoteBlock
        type="note"
        title="Prefill vs Decode Phases"
        content="Each request has two phases: prefill (processing the full prompt in parallel) and decode (generating tokens one by one). Prefill is compute-bound while decode is memory-bound. Advanced schedulers like chunked prefill interleave these phases to balance GPU compute and memory bandwidth."
        id="note-prefill-decode"
      />

      <WarningBlock
        title="Batch Size vs Latency Tradeoff"
        content="Larger batches improve throughput but increase per-request latency because each forward pass takes longer. Monitor time-to-first-token (TTFT) and inter-token latency (ITL) alongside throughput when tuning batch sizes."
        id="warning-latency"
      />
    </div>
  )
}
