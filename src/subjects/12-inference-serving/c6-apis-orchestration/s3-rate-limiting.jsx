import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RateLimiting() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Rate Limiting & Load Balancing</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Production LLM deployments must handle variable traffic without letting any single
        user overwhelm the system. Rate limiting controls how many requests each client can
        make, while load balancing distributes requests across multiple model replicas to
        maximize throughput and availability.
      </p>

      <DefinitionBlock
        title="Rate Limiting"
        definition="Rate limiting restricts the number of API requests a client can make within a time window. For LLM APIs, limits are typically expressed as requests per minute (RPM) and tokens per minute (TPM). The token-based limit is critical because a single request generating 4096 tokens consumes far more GPU time than one generating 10 tokens."
        id="def-rate-limiting"
      />

      <PythonCode
        title="Terminal"
        code={`# Nginx rate limiting for an LLM API
cat > /etc/nginx/conf.d/llm-proxy.conf << 'EOF'
# Define rate limit zones
limit_req_zone $binary_remote_addr zone=llm_per_ip:10m rate=10r/s;
limit_req_zone $http_authorization zone=llm_per_key:10m rate=30r/s;

upstream llm_backends {
    least_conn;                    # Route to least-busy server
    server 10.0.0.1:8000 weight=3; # 3x A100 GPU
    server 10.0.0.2:8000 weight=1; # 1x A100 GPU
    server 10.0.0.3:8000 weight=1 backup; # Failover only
}

server {
    listen 443 ssl;
    server_name api.example.com;

    # Rate limit: 10 req/s per IP, burst up to 20
    location /v1/ {
        limit_req zone=llm_per_ip burst=20 nodelay;
        limit_req zone=llm_per_key burst=50 nodelay;

        # Disable buffering for streaming
        proxy_buffering off;
        proxy_set_header X-Accel-Buffering no;

        proxy_pass http://llm_backends;
        proxy_read_timeout 300s;  # LLM responses can be slow
    }
}
EOF

# Reload nginx
nginx -t && nginx -s reload`}
        id="code-nginx"
      />

      <PythonCode
        title="rate_limiter.py"
        code={`import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class TokenBucket:
    """Token bucket rate limiter for LLM APIs."""
    rpm: int = 60          # Requests per minute
    tpm: int = 100_000     # Tokens per minute
    _requests: list = field(default_factory=list)
    _tokens: list = field(default_factory=list)

    def _cleanup(self, bucket, window=60):
        now = time.time()
        while bucket and bucket[0] < now - window:
            bucket.pop(0)

    def check(self, estimated_tokens=500):
        """Check if request is allowed. Returns (allowed, retry_after_s)."""
        now = time.time()
        self._cleanup(self._requests)
        self._cleanup(self._tokens)
        if len(self._requests) >= self.rpm:
            retry = self._requests[0] + 60 - now
            return False, retry
        token_sum = sum(t for _, t in self._tokens) if self._tokens else 0
        if token_sum + estimated_tokens > self.tpm:
            return False, 5.0
        return True, 0

    def record(self, tokens_used):
        now = time.time()
        self._requests.append(now)
        self._tokens.append((now, tokens_used))

# Per-API-key rate limiting
limiters = defaultdict(lambda: TokenBucket(rpm=30, tpm=50_000))

def handle_request(api_key, estimated_tokens=500):
    limiter = limiters[api_key]
    allowed, retry_after = limiter.check(estimated_tokens)
    if not allowed:
        print(f"Rate limited: retry after {retry_after:.1f}s")
        return 429, {"retry_after": retry_after}
    # ... process request ...
    tokens_used = 150  # actual tokens from response
    limiter.record(tokens_used)
    return 200, {"tokens": tokens_used}

# Simulate traffic
for i in range(40):
    status, data = handle_request("key-user-1")
    print(f"Request {i+1}: {status} {data}")`}
        id="code-rate-limiter"
      />

      <ExampleBlock
        title="Load Balancing Strategies"
        problem="How should you distribute requests across LLM replicas?"
        steps={[
          { formula: 'Least-connections: route to least-busy server', explanation: 'Best for LLMs since request durations vary widely based on output length.' },
          { formula: 'Weighted round-robin: allocate by GPU capacity', explanation: 'Give more traffic to servers with more/faster GPUs.' },
          { formula: 'Session affinity: same user to same server', explanation: 'Helps with prefix caching since the KV cache stays on one server.' },
          { formula: 'Health-check based: skip unhealthy servers', explanation: 'Use /health endpoints to detect and route around GPU OOM or crashes.' },
        ]}
        id="example-lb"
      />

      <NoteBlock
        type="tip"
        title="Client-Side Rate Limit Handling"
        content="Well-behaved clients should respect 429 responses and Retry-After headers. The OpenAI Python SDK has built-in retry logic with exponential backoff. When building clients for rate-limited APIs, implement backoff: wait, double the wait on each retry, add jitter, and cap at a maximum delay."
        id="note-client-retry"
      />

      <WarningBlock
        title="Token Counting Before Generation"
        content="Rate limiting by tokens is challenging because you do not know the output token count before generation starts. Use the max_tokens request parameter as an upper bound estimate, or implement post-hoc accounting that deducts actual tokens used after the response completes. Over-estimating is safer than under-estimating."
        id="warning-token-counting"
      />
    </div>
  )
}
