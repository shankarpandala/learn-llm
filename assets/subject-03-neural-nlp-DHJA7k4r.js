import{j as e}from"./vendor-DWbzdFaj.js";import{r as t}from"./vendor-katex-BYl39Yo6.js";import{D as i,E as o,P as n,N as a,W as s,T as r}from"./subject-01-text-fundamentals-DG6tAvii.js";function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Recurrent Neural Networks (RNN)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Recurrent Neural Networks introduced the idea of maintaining a hidden state that gets updated at each time step, allowing neural networks to process sequences of arbitrary length. This section covers the core RNN architecture, its forward equations, and practical implementation in PyTorch."}),e.jsx(i,{title:"Recurrent Neural Network",definition:"An RNN is a neural network that processes sequential input $x_1, x_2, \\ldots, x_T$ by maintaining a hidden state $h_t$ that is updated at each time step according to $h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.",notation:"$h_t \\in \\mathbb{R}^d$ is the hidden state, $W_{hh} \\in \\mathbb{R}^{d \\times d}$ is the recurrent weight matrix, $W_{xh} \\in \\mathbb{R}^{d \\times n}$ maps input to hidden space.",id:"def-rnn"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"RNN Forward Equations"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["At each time step ",e.jsx(t.InlineMath,{math:"t"}),", the RNN computes:"]}),e.jsx(t.BlockMath,{math:"h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)"}),e.jsx(t.BlockMath,{math:"y_t = W_{hy} h_t + b_y"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The hidden state ",e.jsx(t.InlineMath,{math:"h_t"})," serves as the network's memory. It encodes information about all previous inputs ",e.jsx(t.InlineMath,{math:"x_1, \\ldots, x_t"})," seen so far. The output ",e.jsx(t.InlineMath,{math:"y_t"})," is a linear projection used for the task at hand (classification, next-token prediction, etc.)."]}),e.jsx(o,{title:"Hidden State Dimensions",problem:"An RNN has hidden size 256 and processes word embeddings of dimension 100. What are the shapes of the weight matrices?",steps:[{formula:"W_{xh} \\in \\mathbb{R}^{256 \\times 100}",explanation:"Maps 100-dim input embeddings to 256-dim hidden space."},{formula:"W_{hh} \\in \\mathbb{R}^{256 \\times 256}",explanation:"Recurrent weights connecting previous hidden state to current."},{formula:"b_h \\in \\mathbb{R}^{256}",explanation:"Bias vector for the hidden state update."},{formula:"\\text{Total params} = 256 \\times 100 + 256 \\times 256 + 256 = 91{,}392",explanation:"Parameter count for the recurrent cell alone (excluding output layer)."}],id:"example-rnn-dims"}),e.jsx(n,{title:"rnn_from_scratch.py",code:`import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """Vanilla RNN implemented from scratch."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h_t = h_0
        outputs = []
        for t in range(seq_len):
            h_t = torch.tanh(self.W_xh(x[:, t]) + self.W_hh(h_t))
            outputs.append(h_t)

        # Stack all hidden states: (batch, seq_len, hidden_size)
        hidden_states = torch.stack(outputs, dim=1)
        # Output projection on final hidden state
        out = self.W_hy(h_t)
        return out, hidden_states

# Usage
model = SimpleRNN(input_size=100, hidden_size=256, output_size=10)
x = torch.randn(32, 20, 100)  # batch=32, seq_len=20, embed=100
logits, all_h = model(x)
print(f"Output shape: {logits.shape}")       # (32, 10)
print(f"Hidden states: {all_h.shape}")       # (32, 20, 256)`,id:"code-rnn-scratch"}),e.jsx(n,{title:"rnn_pytorch_builtin.py",code:`import torch
import torch.nn as nn

# PyTorch built-in RNN for text classification
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        output, h_n = self.rnn(emb)      # h_n: (1, batch, hidden_dim)
        logits = self.fc(h_n.squeeze(0)) # (batch, num_classes)
        return logits

model = RNNClassifier(vocab_size=10000, embed_dim=128,
                      hidden_dim=256, num_classes=4)
tokens = torch.randint(0, 10000, (16, 50))  # batch=16, seq_len=50
print(model(tokens).shape)  # (16, 4)`,id:"code-rnn-pytorch"}),e.jsx(a,{type:"intuition",title:"The Hidden State as Memory",content:"Think of the hidden state as a fixed-size summary of everything the network has read so far. At each step, the RNN must decide what to keep from its current memory and what to incorporate from the new input. This compression into a fixed-size vector is both the RNN's strength (constant memory) and its weakness (information bottleneck).",id:"note-hidden-state"}),e.jsx(s,{title:"Vanilla RNNs Struggle with Long Sequences",content:"In practice, vanilla RNNs have difficulty learning dependencies that span more than 10-20 time steps. The repeated matrix multiplication in the recurrence causes gradients to either vanish or explode during backpropagation through time (BPTT). This is addressed by LSTM and GRU architectures.",id:"warning-rnn-limits"}),e.jsx(a,{type:"historical",title:"Origins of Recurrent Networks",content:"The Elman network (1990) introduced the simple recurrent architecture with a hidden state fed back as input. Jordan networks (1986) instead fed the output back. Backpropagation Through Time (BPTT), the algorithm for training RNNs, was formalized by Werbos (1990), though the idea dates to the 1980s.",id:"note-rnn-history"})]})}const z=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Vanishing and Exploding Gradients"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The vanishing and exploding gradient problem is the central obstacle to training deep recurrent networks. Understanding why gradients decay or blow up through time is essential for appreciating why LSTM and GRU architectures were invented, and why gradient clipping became standard practice."}),e.jsx(i,{title:"Vanishing Gradient Problem",definition:"When training an RNN via backpropagation through time (BPTT), the gradient of the loss with respect to early hidden states involves a product of Jacobians $\\prod_{k=t+1}^{T} \\frac{\\partial h_k}{\\partial h_{k-1}}$. If the spectral radius of $W_{hh}$ is less than 1, this product shrinks exponentially, making it impossible to learn long-range dependencies.",notation:"$\\frac{\\partial \\mathcal{L}}{\\partial h_t} = \\frac{\\partial \\mathcal{L}}{\\partial h_T} \\prod_{k=t+1}^{T} \\text{diag}(\\sigma'(z_k)) W_{hh}$",id:"def-vanishing-gradient"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Gradient Flow Through Time"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["For a vanilla RNN with ",e.jsx(t.InlineMath,{math:"h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)"}),", the Jacobian of ",e.jsx(t.InlineMath,{math:"h_t"})," with respect to ",e.jsx(t.InlineMath,{math:"h_{t-1}"})," is:"]}),e.jsx(t.BlockMath,{math:"\\frac{\\partial h_t}{\\partial h_{t-1}} = \\text{diag}(1 - h_t^2) \\cdot W_{hh}"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Since ",e.jsx(t.InlineMath,{math:"\\tanh'(x) = 1 - \\tanh^2(x) \\leq 1"}),", and the diagonal entries are often much less than 1 for saturated units, repeated multiplication causes the gradient to shrink exponentially with the number of time steps."]}),e.jsx(r,{title:"Gradient Bound for Vanilla RNNs",statement:"For a vanilla RNN, if $\\|W_{hh}\\| < 1 / \\gamma$ where $\\gamma = \\max_t \\|\\text{diag}(1 - h_t^2)\\|$, then $\\left\\|\\frac{\\partial h_T}{\\partial h_t}\\right\\| \\leq (\\gamma \\|W_{hh}\\|)^{T-t} \\to 0$ as $T - t \\to \\infty$.",corollaries:["Long-range gradients vanish exponentially, making it impossible to credit-assign across many time steps.","Conversely, if $\\gamma \\|W_{hh}\\| > 1$, gradients explode exponentially."],id:"thm-gradient-bound"}),e.jsx(o,{title:"Sigmoid Saturation",problem:"Show why sigmoid and tanh activations cause gradient vanishing.",steps:[{formula:"\\sigma'(x) = \\sigma(x)(1 - \\sigma(x)) \\leq 0.25",explanation:"The sigmoid derivative peaks at 0.25 when x=0 and approaches 0 at the extremes."},{formula:"\\tanh'(x) = 1 - \\tanh^2(x) \\leq 1",explanation:"The tanh derivative equals 1 only at x=0 and decays rapidly for |x| > 2."},{formula:"\\text{After } k \\text{ steps: } (0.25)^k \\text{ for sigmoid}",explanation:"With sigmoid, after just 10 steps the gradient is attenuated by a factor of ~10^{-6}."},{formula:"\\text{After 20 steps: } \\approx 10^{-12}",explanation:"Gradients become numerically indistinguishable from zero in float32."}],id:"example-saturation"}),e.jsx(n,{title:"visualize_gradient_flow.py",code:`import torch
import torch.nn as nn

def measure_gradient_flow(seq_len=50, hidden_size=128):
    """Measure how gradients decay through an RNN."""
    rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    # Random input sequence
    x = torch.randn(1, seq_len, hidden_size, requires_grad=True)
    h0 = torch.zeros(1, 1, hidden_size)

    output, _ = rnn(x, h0)

    # Compute gradient of final output w.r.t. input at each step
    loss = output[0, -1].sum()  # scalar from last time step
    loss.backward()

    # Gradient magnitude at each time step
    grad_norms = x.grad[0].norm(dim=1).detach()
    print(f"Gradient at step 0 (earliest): {grad_norms[0]:.6f}")
    print(f"Gradient at step {seq_len//2}: {grad_norms[seq_len//2]:.6f}")
    print(f"Gradient at step {seq_len-1} (latest): {grad_norms[-1]:.6f}")
    print(f"Ratio (first/last): {grad_norms[0]/grad_norms[-1]:.6f}")

measure_gradient_flow(seq_len=50)
# Typical output: gradient at step 0 is orders of magnitude
# smaller than at step 49, confirming vanishing gradients.`,id:"code-grad-flow"}),e.jsx(n,{title:"exploding_gradient_demo.py",code:`import torch
import torch.nn as nn

# Demonstrate exploding gradients with large weights
rnn = nn.RNN(64, 64, batch_first=True)
# Artificially scale recurrent weights to cause explosion
with torch.no_grad():
    rnn.weight_hh_l0.mul_(2.0)

x = torch.randn(1, 100, 64, requires_grad=True)
output, _ = rnn(x)
loss = output[0, -1].sum()

try:
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"Gradient norm: {grad_norm}")
    if grad_norm > 1e6:
        print("Gradients exploded!")
except RuntimeError as e:
    print(f"Numerical error: {e}")

# Solution: gradient clipping
torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
print("After clipping, training can proceed stably.")`,id:"code-exploding"}),e.jsx(a,{type:"intuition",title:"Why This Matters for Language",content:"Consider the sentence: 'The cat, which sat on the mat and watched the birds outside the window for hours, was hungry.' The RNN needs to connect 'cat' to 'was hungry' across ~15 tokens. Vanishing gradients mean the model cannot learn that 'cat' determines the verb form. This is why vanilla RNNs fail at subject-verb agreement over long distances.",id:"note-language-impact"}),e.jsx(s,{title:"Gradient Clipping Is Not a Full Solution",content:"While gradient clipping prevents exploding gradients, it does nothing for vanishing gradients. Clipping rescales large gradients but cannot amplify small ones. Architectural changes like LSTM gates are needed to create gradient highways that allow information to flow unimpeded across many time steps.",id:"warning-clipping-limits"})]})}const L=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LSTM Architecture"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem by introducing a gating mechanism that controls information flow. The cell state acts as a conveyor belt, allowing gradients to flow across many time steps with minimal decay. LSTMs were the dominant sequence model from 1997 until transformers emerged in 2017."}),e.jsx(i,{title:"LSTM Cell",definition:"An LSTM cell maintains two state vectors: the hidden state $h_t$ and the cell state $c_t$. Three gates (forget, input, output) regulate information flow. The cell state update is additive rather than multiplicative, creating a gradient highway.",notation:"$f_t, i_t, o_t \\in (0,1)^d$ are the forget, input, and output gates; $c_t \\in \\mathbb{R}^d$ is the cell state.",id:"def-lstm"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"LSTM Gate Equations"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The LSTM computes four quantities at each time step:"}),e.jsx(t.BlockMath,{math:"f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f) \\quad \\text{(forget gate)}"}),e.jsx(t.BlockMath,{math:"i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i) \\quad \\text{(input gate)}"}),e.jsx(t.BlockMath,{math:"\\tilde{c}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c) \\quad \\text{(candidate cell)}"}),e.jsx(t.BlockMath,{math:"c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t \\quad \\text{(cell update)}"}),e.jsx(t.BlockMath,{math:"o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o) \\quad \\text{(output gate)}"}),e.jsx(t.BlockMath,{math:"h_t = o_t \\odot \\tanh(c_t) \\quad \\text{(hidden state)}"}),e.jsx(o,{title:"Gate Roles in Practice",problem:"Explain what each gate does when processing the sentence: 'The cat sat on the mat. It was happy.'",steps:[{formula:"f_t \\approx 1 \\text{ (keep)}, f_t \\approx 0 \\text{ (forget)}",explanation:'The forget gate decides to keep "cat" info through "sat on the mat" (f near 1) and might clear some info at the period (f near 0).'},{formula:"i_t \\approx 1 \\text{ when new info is relevant}",explanation:'The input gate opens when processing "happy" to write the sentiment into the cell state.'},{formula:"o_t \\text{ controls what is exposed}",explanation:'The output gate selects which cell dimensions are relevant for the current prediction. At "It", it exposes the subject info to resolve the pronoun.'}],id:"example-gates"}),e.jsx(a,{type:"intuition",title:"Why the Cell State Fixes Vanishing Gradients",content:"The key insight is the cell state update: c_t = f_t * c_{t-1} + i_t * c_tilde. This is an additive update, not a multiplicative one. When the forget gate is close to 1, the gradient flows through the cell state almost unchanged, like a skip connection. The gradient of c_T with respect to c_t is the product of forget gates, which can stay close to 1.",id:"note-gradient-highway"}),e.jsx(n,{title:"lstm_from_scratch.py",code:`import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """LSTM cell implemented from scratch."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined linear for all 4 gates (efficiency)
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, state):
        h_prev, c_prev = state
        combined = torch.cat([h_prev, x_t], dim=-1)
        gates = self.gates(combined)

        # Split into 4 gate activations
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        g = torch.tanh(g)     # candidate cell
        o = torch.sigmoid(o)  # output gate

        c_t = f * c_prev + i * g   # cell state update
        h_t = o * torch.tanh(c_t)  # hidden state
        return h_t, c_t

# Test
cell = LSTMCell(input_size=100, hidden_size=256)
h = torch.zeros(32, 256)
c = torch.zeros(32, 256)
x = torch.randn(32, 100)
h_new, c_new = cell(x, (h, c))
print(f"h: {h_new.shape}, c: {c_new.shape}")  # (32, 256), (32, 256)`,id:"code-lstm-scratch"}),e.jsx(n,{title:"lstm_text_classifier.py",code:`import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        # output: (batch, seq_len, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        output, (h_n, c_n) = self.lstm(emb)
        # Use final layer's hidden state
        logits = self.fc(self.dropout(h_n[-1]))
        return logits

model = LSTMClassifier(
    vocab_size=30000, embed_dim=128,
    hidden_dim=256, num_classes=5
)
x = torch.randint(0, 30000, (16, 100))
print(f"Output: {model(x).shape}")  # (16, 5)

# Parameter count comparison with vanilla RNN
rnn_params = 128*256 + 256*256 + 256      # ~98K
lstm_params = 4 * (128*256 + 256*256 + 256)  # ~394K
print(f"RNN params: {rnn_params:,}")
print(f"LSTM params: {lstm_params:,}")  # 4x more due to 4 gates`,id:"code-lstm-classifier"}),e.jsx(s,{title:"LSTM Parameter Count",content:"LSTMs have 4x the parameters of a vanilla RNN with the same hidden size because they compute four gate/candidate values. For hidden_size=d and input_size=n, an LSTM cell has 4(n*d + d*d + d) parameters. This makes them slower to train but far more capable at capturing long-range dependencies.",id:"warning-lstm-params"}),e.jsx(a,{type:"historical",title:"LSTM Timeline",content:"LSTMs were introduced by Hochreiter and Schmidhuber in 1997. The forget gate was added by Gers et al. in 2000 (the original had no forget gate). Peephole connections (letting gates see the cell state directly) were introduced in 2002 but are rarely used today. LSTMs dominated NLP from roughly 2014-2017, powering Google Translate, speech recognition, and state-of-the-art language models.",id:"note-lstm-history"})]})}const M=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GRU and Bidirectional RNNs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The Gated Recurrent Unit (GRU) simplifies the LSTM by merging the forget and input gates into a single update gate, and combining the cell and hidden states. Bidirectional RNNs process sequences in both directions to capture context from both past and future tokens."}),e.jsx(i,{title:"Gated Recurrent Unit (GRU)",definition:"A GRU uses two gates: an update gate $z_t$ and a reset gate $r_t$. The update gate interpolates between the previous hidden state and a candidate, while the reset gate controls how much past state enters the candidate computation.",notation:"$z_t = \\sigma(W_z [h_{t-1}, x_t])$, $r_t = \\sigma(W_r [h_{t-1}, x_t])$, $\\tilde{h}_t = \\tanh(W [r_t \\odot h_{t-1}, x_t])$, $h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$",id:"def-gru"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"GRU Equations"}),e.jsx(t.BlockMath,{math:"z_t = \\sigma(W_z x_t + U_z h_{t-1} + b_z) \\quad \\text{(update gate)}"}),e.jsx(t.BlockMath,{math:"r_t = \\sigma(W_r x_t + U_r h_{t-1} + b_r) \\quad \\text{(reset gate)}"}),e.jsx(t.BlockMath,{math:"\\tilde{h}_t = \\tanh(W_h x_t + U_h (r_t \\odot h_{t-1}) + b_h)"}),e.jsx(t.BlockMath,{math:"h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["When ",e.jsx(t.InlineMath,{math:"z_t \\approx 0"}),", the hidden state is copied forward unchanged (like a skip connection). When ",e.jsx(t.InlineMath,{math:"z_t \\approx 1"}),", the state is fully replaced by the candidate. The reset gate ",e.jsx(t.InlineMath,{math:"r_t"})," controls how much history enters the candidate computation."]}),e.jsx(o,{title:"GRU vs LSTM Parameter Count",problem:"Compare parameter counts for GRU and LSTM with input_size=128, hidden_size=256.",steps:[{formula:"\\text{LSTM: } 4 \\times (128 \\times 256 + 256 \\times 256 + 256) = 394{,}240",explanation:"LSTM has 4 weight matrices (forget, input, candidate, output)."},{formula:"\\text{GRU: } 3 \\times (128 \\times 256 + 256 \\times 256 + 256) = 295{,}680",explanation:"GRU has 3 weight matrices (update, reset, candidate) -- 25% fewer parameters."},{formula:"\\text{Ratio: } 295{,}680 / 394{,}240 = 0.75",explanation:"GRU uses 75% of the parameters of an LSTM, leading to faster training with comparable performance on many tasks."}],id:"example-gru-params"}),e.jsx(n,{title:"gru_implementation.py",code:`import torch
import torch.nn as nn

class GRUCell(nn.Module):
    """GRU cell from scratch."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=-1)
        z = torch.sigmoid(self.W_z(combined))  # update gate
        r = torch.sigmoid(self.W_r(combined))  # reset gate

        combined_r = torch.cat([x_t, r * h_prev], dim=-1)
        h_candidate = torch.tanh(self.W_h(combined_r))

        h_t = (1 - z) * h_prev + z * h_candidate
        return h_t

# Quick test
cell = GRUCell(128, 256)
h = torch.zeros(32, 256)
x = torch.randn(32, 128)
h_new = cell(x, h)
print(f"New hidden state: {h_new.shape}")  # (32, 256)`,id:"code-gru-scratch"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Bidirectional RNNs"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["A unidirectional RNN only sees past context when computing ",e.jsx(t.InlineMath,{math:"h_t"}),". For many NLP tasks (named entity recognition, sentiment analysis, machine comprehension), future context is equally informative. Bidirectional RNNs run two separate RNNs: one forward and one backward, then concatenate their hidden states."]}),e.jsx(t.BlockMath,{math:"\\overrightarrow{h_t} = \\text{RNN}_{\\text{fwd}}(x_t, \\overrightarrow{h_{t-1}})"}),e.jsx(t.BlockMath,{math:"\\overleftarrow{h_t} = \\text{RNN}_{\\text{bwd}}(x_t, \\overleftarrow{h_{t+1}})"}),e.jsx(t.BlockMath,{math:"h_t = [\\overrightarrow{h_t}; \\overleftarrow{h_t}] \\in \\mathbb{R}^{2d}"}),e.jsx(n,{title:"bidirectional_lstm.py",code:`import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,  # <-- key change
            dropout=0.3,
        )
        # hidden_dim * 2 because bidirectional concatenates
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        output, (h_n, _) = self.lstm(emb)
        # h_n shape: (num_layers*2, batch, hidden_dim)
        # Concatenate final forward and backward hidden states
        fwd = h_n[-2]  # last forward layer
        bwd = h_n[-1]  # last backward layer
        combined = torch.cat([fwd, bwd], dim=-1)
        return self.fc(combined)

model = BiLSTMClassifier(20000, 128, 256, 3)
x = torch.randint(0, 20000, (16, 80))
print(f"Output: {model(x).shape}")  # (16, 3)

# Bidirectional doubles the representation size at each step
output, _ = model.lstm(model.embedding(x))
print(f"BiLSTM output: {output.shape}")  # (16, 80, 512)`,id:"code-bilstm"}),e.jsx(a,{type:"tip",title:"When to Use Bidirectional",content:"Use bidirectional RNNs for encoding tasks where the full sequence is available (classification, tagging, question answering). Do NOT use them for autoregressive generation (language modeling, machine translation decoding), since the model would be cheating by looking at future tokens it is supposed to predict.",id:"note-when-bidir"}),e.jsx(s,{title:"Bidirectional Doubles Memory and Compute",content:"A bidirectional LSTM with hidden_size=256 produces 512-dimensional representations and has twice the parameters of a unidirectional one. For very long sequences, this can be a significant memory burden. Also, bidirectional models cannot be used for streaming/online inference since they require the complete input.",id:"warning-bidir-cost"}),e.jsx(a,{type:"historical",title:"GRU and Bidirectional Origins",content:"The GRU was proposed by Cho et al. (2014) in the context of neural machine translation. Empirical studies (Chung et al., 2014; Jozefowicz et al., 2015) found GRU and LSTM perform comparably, with GRU sometimes better on smaller datasets. Bidirectional RNNs were introduced by Schuster and Paliwal (1997) and became standard in NLP with ELMo (Peters et al., 2018).",id:"note-gru-history"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Encoder-Decoder Architecture"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The encoder-decoder (seq2seq) architecture maps variable-length input sequences to variable-length output sequences. The encoder compresses the input into a fixed-size context vector, and the decoder generates the output one token at a time, conditioned on this context. This architecture powered the first neural machine translation systems."}),e.jsx(i,{title:"Sequence-to-Sequence Model",definition:"A seq2seq model consists of an encoder RNN that reads the input sequence $x_1, \\ldots, x_S$ and produces a context vector $c = h_S^{\\text{enc}}$, and a decoder RNN that generates the output sequence $y_1, \\ldots, y_T$ conditioned on $c$.",notation:"$P(y_1, \\ldots, y_T | x_1, \\ldots, x_S) = \\prod_{t=1}^{T} P(y_t | y_{<t}, c)$",id:"def-seq2seq"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Information Bottleneck"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The context vector ",e.jsx(t.InlineMath,{math:"c \\in \\mathbb{R}^d"})," must encode all information from the source sequence into a single fixed-size vector. For a 50-word sentence compressed into a 512-dimensional vector, each dimension must encode roughly 0.1 words worth of information. This bottleneck degrades performance on long sequences."]}),e.jsx(t.BlockMath,{math:"c = h_S^{\\text{enc}} = f_{\\text{enc}}(x_1, x_2, \\ldots, x_S)"}),e.jsx(t.BlockMath,{math:"h_t^{\\text{dec}} = f_{\\text{dec}}(y_{t-1}, h_{t-1}^{\\text{dec}}, c)"}),e.jsx(t.BlockMath,{math:"P(y_t | y_{<t}, c) = \\text{softmax}(W_o h_t^{\\text{dec}})"}),e.jsx(o,{title:"Seq2Seq for Translation",problem:"Trace through a seq2seq model translating 'I love cats' to 'J'aime les chats'.",steps:[{formula:'h_1 = \\text{enc}(\\text{emb}(\\text{"I"}), h_0)',explanation:"Encoder processes first token, updating hidden state."},{formula:'h_2 = \\text{enc}(\\text{emb}(\\text{"love"}), h_1)',explanation:"Second token encoded, hidden state accumulates meaning."},{formula:'c = h_3 = \\text{enc}(\\text{emb}(\\text{"cats"}), h_2)',explanation:"Final encoder hidden state becomes the context vector."},{formula:`P(y_1|c) \\to \\text{"J'aime"}`,explanation:"Decoder generates first target token from context vector."},{formula:'P(y_2|y_1, c) \\to \\text{"les"}',explanation:"Each subsequent token is conditioned on previous outputs and context."}],id:"example-translation"}),e.jsx(n,{title:"seq2seq_model.py",code:`import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        # hidden: (num_layers, batch, hidden_dim) -- context vector
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, hidden, cell):
        # tgt_token: (batch, 1) -- single token
        embedded = self.embedding(tgt_token)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size,
                              device=self.device)

        hidden, cell = self.encoder(src)
        input_tok = tgt[:, 0:1]  # <SOS> token

        for t in range(1, tgt_len):
            pred, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t] = pred
            # Teacher forcing: use ground truth or model prediction
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_tok = tgt[:, t:t+1]
            else:
                input_tok = pred.argmax(dim=-1, keepdim=True)
        return outputs

# Build model
device = torch.device('cpu')
enc = Encoder(vocab_size=10000, embed_dim=256, hidden_dim=512)
dec = Decoder(vocab_size=8000, embed_dim=256, hidden_dim=512)
model = Seq2Seq(enc, dec, device)
src = torch.randint(0, 10000, (4, 20))
tgt = torch.randint(0, 8000, (4, 15))
out = model(src, tgt)
print(f"Output: {out.shape}")  # (4, 15, 8000)`,id:"code-seq2seq"}),e.jsx(s,{title:"The Bottleneck Problem Is Real",content:"Cho et al. (2014) showed that seq2seq performance degrades sharply for sentences longer than ~20 tokens. The fixed-size context vector simply cannot retain all necessary information from long inputs. This was the primary motivation for introducing attention mechanisms (Bahdanau et al., 2015).",id:"warning-bottleneck"}),e.jsx(a,{type:"historical",title:"Birth of Neural Machine Translation",content:"The seq2seq architecture was independently proposed by Sutskever et al. (2014) at Google and Cho et al. (2014). Sutskever's key trick was reversing the source sentence order, which shortened the distance between corresponding words and improved BLEU scores. Within two years, Google deployed a production NMT system (Wu et al., 2016) based on these ideas.",id:"note-nmt-history"}),e.jsx(a,{type:"tip",title:"Tricks for Better Seq2Seq",content:"Common improvements include: (1) using bidirectional encoder, (2) multi-layer LSTMs with residual connections, (3) reversing the source sequence, (4) beam search decoding instead of greedy, (5) input feeding (concatenating attention output to decoder input). Each provides incremental gains, but attention was the transformative addition.",id:"note-seq2seq-tricks"})]})}const $=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Teacher Forcing"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Teacher forcing is a training strategy for autoregressive models where the ground-truth previous token is fed as input to the decoder at each step, rather than the model's own prediction. It dramatically speeds up training but introduces a train-test mismatch known as exposure bias."}),e.jsx(i,{title:"Teacher Forcing",definition:"During training, at each decoder time step $t$, use the ground-truth token $y_{t-1}^*$ as input instead of the model's predicted token $\\hat{y}_{t-1}$. The training objective becomes $\\mathcal{L} = -\\sum_{t=1}^{T} \\log P(y_t^* | y_1^*, \\ldots, y_{t-1}^*, c)$.",notation:"With teacher forcing: input is $y_{t-1}^*$. Without: input is $\\hat{y}_{t-1} = \\arg\\max P(y | h_{t-1})$.",id:"def-teacher-forcing"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Why Teacher Forcing Works"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Without teacher forcing, early in training the decoder makes many mistakes. These errors compound: a wrong token at step 3 pushes the decoder into states it never saw during training, leading to garbage outputs for the rest of the sequence. Teacher forcing prevents this error cascade by always providing correct context."}),e.jsx(t.BlockMath,{math:"\\text{Teacher forcing: } h_t^{\\text{dec}} = f(y_{t-1}^*, h_{t-1}^{\\text{dec}}, c)"}),e.jsx(t.BlockMath,{math:"\\text{Free running: } h_t^{\\text{dec}} = f(\\hat{y}_{t-1}, h_{t-1}^{\\text{dec}}, c)"}),e.jsx(i,{title:"Exposure Bias",definition:"Exposure bias is the discrepancy between training (where the decoder sees ground-truth tokens) and inference (where it sees its own predictions). The model is never 'exposed' to its own errors during training, so it cannot learn to recover from mistakes.",id:"def-exposure-bias"}),e.jsx(o,{title:"Exposure Bias in Translation",problem:"Show how a single error cascades during free-running decoding.",steps:[{formula:'\\text{Target: "The cat sat on the mat"}',explanation:"Ground truth sequence the model should produce."},{formula:'\\hat{y}_1 = \\text{"The"} \\checkmark',explanation:"First token is correct."},{formula:'\\hat{y}_2 = \\text{"dog"} \\text{ (error!)}',explanation:'Model makes an error at step 2, predicting "dog" instead of "cat".'},{formula:'\\hat{y}_3 = \\text{"ran"} \\text{ (cascading)}',explanation:'Conditioned on "The dog", the model generates a plausible continuation but completely diverges from the target.'},{formula:'\\hat{y}_4 = \\text{"away"} \\to \\text{total divergence}',explanation:"Each step compounds the error. The decoder has never seen this trajectory during training."}],id:"example-exposure-bias"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Scheduled Sampling"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Scheduled sampling (Bengio et al., 2015) bridges the gap between teacher forcing and free-running by gradually decreasing the probability of using ground-truth tokens during training. The curriculum starts with mostly teacher forcing and transitions to mostly free running."}),e.jsx(t.BlockMath,{math:"\\epsilon_i = \\max(\\epsilon_{\\min}, k^i) \\quad \\text{(exponential decay)}"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["where ",e.jsx(t.InlineMath,{math:"\\epsilon_i"})," is the probability of using ground truth at epoch ",e.jsx(t.InlineMath,{math:"i"}),", and ",e.jsx(t.InlineMath,{math:"k < 1"})," controls the decay rate."]}),e.jsx(n,{title:"teacher_forcing_training.py",code:`import torch
import torch.nn as nn
import random

def train_step(model, src, tgt, optimizer, criterion,
               teacher_forcing_ratio=0.5):
    """Single training step with teacher forcing."""
    model.train()
    optimizer.zero_grad()

    batch_size, tgt_len = tgt.shape
    vocab_size = model.decoder.fc_out.out_features
    outputs = torch.zeros(batch_size, tgt_len, vocab_size,
                          device=src.device)

    hidden, cell = model.encoder(src)
    input_tok = tgt[:, 0:1]  # <SOS>

    for t in range(1, tgt_len):
        pred, hidden, cell = model.decoder(input_tok, hidden, cell)
        outputs[:, t] = pred

        # Decide: teacher forcing or autoregressive
        use_teacher = random.random() < teacher_forcing_ratio
        if use_teacher:
            input_tok = tgt[:, t:t+1]       # ground truth
        else:
            input_tok = pred.argmax(-1, keepdim=True)  # model prediction

    # Reshape for cross-entropy: (batch * tgt_len, vocab)
    loss = criterion(
        outputs[:, 1:].reshape(-1, vocab_size),
        tgt[:, 1:].reshape(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()`,id:"code-teacher-forcing"}),e.jsx(n,{title:"scheduled_sampling.py",code:`import math

def get_teacher_forcing_ratio(epoch, strategy='exponential',
                              k=0.95, min_ratio=0.1):
    """Compute scheduled sampling ratio."""
    if strategy == 'exponential':
        # Exponential decay: starts at 1.0, decays by k each epoch
        return max(min_ratio, k ** epoch)
    elif strategy == 'linear':
        # Linear decay over 50 epochs
        return max(min_ratio, 1.0 - epoch / 50)
    elif strategy == 'inverse_sigmoid':
        # Inverse sigmoid decay (smoother transition)
        return max(min_ratio, k / (k + math.exp(epoch / k)))
    else:
        return 0.5  # constant

# Training loop with scheduled sampling
num_epochs = 100
for epoch in range(num_epochs):
    tf_ratio = get_teacher_forcing_ratio(epoch, strategy='exponential')
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: teacher_forcing_ratio = {tf_ratio:.3f}")
    # loss = train_step(model, src, tgt, optimizer, criterion, tf_ratio)

# Output:
# Epoch 0: teacher_forcing_ratio = 1.000
# Epoch 10: teacher_forcing_ratio = 0.599
# Epoch 20: teacher_forcing_ratio = 0.358
# Epoch 30: teacher_forcing_ratio = 0.215
# Epoch 40: teacher_forcing_ratio = 0.129
# Epoch 50: teacher_forcing_ratio = 0.100 (clamped)`,id:"code-scheduled-sampling"}),e.jsx(s,{title:"Scheduled Sampling Has Limitations",content:"While scheduled sampling reduces exposure bias, it is not a principled solution. Mixing ground-truth and predicted tokens creates an inconsistent training signal -- the decoder sees a trajectory that is neither fully correct nor fully self-generated. More principled approaches include sequence-level training with REINFORCE or minimum risk training.",id:"warning-scheduled-limits"}),e.jsx(a,{type:"tip",title:"Modern Approaches to Exposure Bias",content:"Transformers largely sidestep exposure bias during training because teacher forcing is exact parallel computation (no sequential dependence). However, autoregressive decoding at inference time still suffers from error accumulation. Techniques like nucleus sampling, beam search with length penalties, and speculative decoding help manage this.",id:"note-modern-approaches"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Bahdanau Attention"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Bahdanau attention (2015) was the breakthrough that eliminated the information bottleneck in seq2seq models. Instead of compressing the entire input into a single vector, attention allows the decoder to look back at all encoder hidden states and dynamically focus on the most relevant parts for each output token. This is the direct ancestor of the attention mechanism in transformers."}),e.jsx(i,{title:"Bahdanau (Additive) Attention",definition:"At each decoder step $t$, compute alignment scores $e_{t,j} = v^T \\tanh(W_1 h_j^{\\text{enc}} + W_2 s_{t-1}^{\\text{dec}})$ for all encoder positions $j$. Normalize via softmax to get attention weights $\\alpha_{t,j}$, then compute the context vector as a weighted sum $c_t = \\sum_j \\alpha_{t,j} h_j^{\\text{enc}}$.",notation:"$e_{t,j} \\in \\mathbb{R}$ is the alignment score, $\\alpha_{t,j} \\in [0,1]$ are attention weights, $c_t \\in \\mathbb{R}^d$ is the context vector.",id:"def-bahdanau"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Attention Equations"}),e.jsx(t.BlockMath,{math:"e_{t,j} = v^T \\tanh(W_1 h_j^{\\text{enc}} + W_2 s_{t-1}^{\\text{dec}})"}),e.jsx(t.BlockMath,{math:"\\alpha_{t,j} = \\frac{\\exp(e_{t,j})}{\\sum_{k=1}^{S} \\exp(e_{t,k})}"}),e.jsx(t.BlockMath,{math:"c_t = \\sum_{j=1}^{S} \\alpha_{t,j} \\, h_j^{\\text{enc}}"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The context vector ",e.jsx(t.InlineMath,{math:"c_t"})," is concatenated with the decoder input and fed into the decoder RNN:"]}),e.jsx(t.BlockMath,{math:"s_t^{\\text{dec}} = f(s_{t-1}^{\\text{dec}}, [y_{t-1}; c_t])"}),e.jsx(o,{title:"Attention Alignment",problem:"For translating 'the black cat' to 'le chat noir', what should the attention weights look like?",steps:[{formula:"\\alpha_{1,:} \\approx [0.9, 0.05, 0.05]",explanation:'When generating "le", attention focuses on "the" (position 1).'},{formula:"\\alpha_{2,:} \\approx [0.05, 0.1, 0.85]",explanation:'When generating "chat", attention focuses on "cat" (position 3), not "black".'},{formula:"\\alpha_{3,:} \\approx [0.05, 0.85, 0.1]",explanation:'When generating "noir", attention focuses on "black" (position 2). Note the word-order reversal handled naturally by attention.'}],id:"example-alignment"}),e.jsx(n,{title:"bahdanau_attention.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch, src_len, enc_dim)
        # decoder_hidden: (batch, dec_dim)

        # Expand decoder hidden to match src_len dimension
        dec_expanded = decoder_hidden.unsqueeze(1)  # (batch, 1, dec_dim)

        # Compute alignment scores
        energy = self.v(
            torch.tanh(self.W1(encoder_outputs) + self.W2(dec_expanded))
        )  # (batch, src_len, 1)

        attention_weights = F.softmax(energy.squeeze(-1), dim=-1)
        # (batch, src_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)  # (batch, enc_dim)

        return context, attention_weights

# Test
attn = BahdanauAttention(enc_dim=512, dec_dim=512, attn_dim=256)
enc_out = torch.randn(4, 20, 512)    # 4 sentences, 20 tokens
dec_h = torch.randn(4, 512)          # decoder state
ctx, weights = attn(enc_out, dec_h)
print(f"Context: {ctx.shape}")        # (4, 512)
print(f"Weights: {weights.shape}")    # (4, 20)
print(f"Weights sum: {weights.sum(-1)}")  # [1, 1, 1, 1]`,id:"code-bahdanau"}),e.jsx(n,{title:"attention_decoder.py",code:`import torch
import torch.nn as nn

class AttentionDecoder(nn.Module):
    """Decoder with Bahdanau attention."""
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(enc_dim, dec_dim, attn_dim)
        # Input: embedding + context vector
        self.rnn = nn.GRU(embed_dim + enc_dim, dec_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_dim + enc_dim + embed_dim, vocab_size)

    def forward(self, tgt_token, decoder_hidden, encoder_outputs):
        # tgt_token: (batch,) single token ids
        embedded = self.embedding(tgt_token)  # (batch, embed_dim)

        context, attn_weights = self.attention(
            encoder_outputs, decoder_hidden
        )

        # Concatenate embedding and context as RNN input
        rnn_input = torch.cat([embedded, context], dim=-1)
        rnn_input = rnn_input.unsqueeze(1)  # (batch, 1, embed+enc)

        output, hidden = self.rnn(
            rnn_input, decoder_hidden.unsqueeze(0)
        )
        hidden = hidden.squeeze(0)  # (batch, dec_dim)

        # Prediction from hidden + context + embedding
        prediction = self.fc_out(
            torch.cat([hidden, context, embedded], dim=-1)
        )
        return prediction, hidden, attn_weights`,id:"code-attn-decoder"}),e.jsx(a,{type:"intuition",title:"Attention as Soft Addressing",content:"Think of encoder hidden states as memory slots and the decoder state as a query. Attention computes a similarity between the query and each memory slot, then retrieves a weighted combination. This is essentially a differentiable dictionary lookup -- the foundation of the Query-Key-Value framework in transformers.",id:"note-soft-addressing"}),e.jsx(s,{title:"Attention Complexity",content:"Bahdanau attention has O(S * T) time complexity where S is the source length and T is the target length. For each of the T decoder steps, we compute scores against all S encoder positions. This is acceptable for short sequences but becomes costly for very long inputs (thousands of tokens), motivating efficient attention variants.",id:"warning-attention-cost"}),e.jsx(a,{type:"historical",title:"The Paper That Changed NLP",content:"'Neural Machine Translation by Jointly Learning to Align and Translate' (Bahdanau, Cho, Bengio, 2015) introduced attention to NLP. The alignment visualization -- showing which source words the model attends to for each target word -- was groundbreaking because it made the model interpretable. This paper has over 30,000 citations and directly inspired the transformer's attention mechanism.",id:"note-bahdanau-history"})]})}const q=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Luong Attention Variants"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Luong et al. (2015) proposed simplified attention mechanisms that compute alignment scores using the current decoder hidden state (rather than the previous one as in Bahdanau). They introduced three scoring functions -- dot, general, and concat -- and compared global vs. local attention. The dot-product scoring became the basis for transformer attention."}),e.jsx(i,{title:"Luong Attention Scoring Functions",definition:"Given encoder hidden state $h_j$ and decoder hidden state $s_t$, Luong defines three scoring functions: (1) Dot: $\\text{score}(s_t, h_j) = s_t^T h_j$, (2) General: $\\text{score}(s_t, h_j) = s_t^T W_a h_j$, (3) Concat: $\\text{score}(s_t, h_j) = v_a^T \\tanh(W_a [s_t; h_j])$.",notation:"In all cases, $\\alpha_{t,j} = \\text{softmax}_j(\\text{score}(s_t, h_j))$ and $c_t = \\sum_j \\alpha_{t,j} h_j$.",id:"def-luong-attention"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Scoring Functions Compared"}),e.jsx(t.BlockMath,{math:"\\text{Dot: } e_{t,j} = s_t^T h_j"}),e.jsx(t.BlockMath,{math:"\\text{General: } e_{t,j} = s_t^T W_a h_j"}),e.jsx(t.BlockMath,{math:"\\text{Concat: } e_{t,j} = v_a^T \\tanh(W_a [s_t; h_j])"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The dot product is the simplest and fastest, requiring no learnable parameters in the scoring function itself. However, it requires encoder and decoder to have the same hidden dimension. The general scoring adds a learnable matrix",e.jsx(t.InlineMath,{math:"W_a \\in \\mathbb{R}^{d \\times d}"})," that can handle different dimensions and learn a task-specific similarity metric."]}),e.jsx(o,{title:"Computational Comparison",problem:"Compare FLOPs for the three scoring functions with hidden_dim=512 and src_len=30.",steps:[{formula:"\\text{Dot: } 30 \\times 512 = 15{,}360 \\text{ multiplies}",explanation:"One dot product per source position. No parameters to learn."},{formula:"\\text{General: } 512^2 + 30 \\times 512 = 277{,}504",explanation:"Matrix-vector product W_a * h_j (can be precomputed) plus dot products."},{formula:"\\text{Concat: } 30 \\times (1024 \\times d_a + d_a)",explanation:"Most expensive: concatenation, linear projection, tanh, and dot with v for each position."},{formula:"\\text{Dot} \\ll \\text{General} < \\text{Concat}",explanation:"Dot product is orders of magnitude faster, which is why transformers use scaled dot-product attention."}],id:"example-scoring-flops"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Global vs. Local Attention"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Global attention attends to all source positions (like Bahdanau). Local attention predicts an alignment position ",e.jsx(t.InlineMath,{math:"p_t"})," and only attends to a window ",e.jsx(t.InlineMath,{math:"[p_t - D, p_t + D]"})," around it. This reduces computation from ",e.jsx(t.InlineMath,{math:"O(S)"})," to ",e.jsx(t.InlineMath,{math:"O(D)"})," per decoder step."]}),e.jsx(t.BlockMath,{math:"p_t = S \\cdot \\sigma(v_p^T \\tanh(W_p s_t))"}),e.jsx(t.BlockMath,{math:"\\alpha_{t,j} = \\text{align}(t, j) \\cdot \\exp\\left(-\\frac{(j - p_t)^2}{2\\sigma^2}\\right)"}),e.jsx(n,{title:"luong_attention_variants.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    """Luong attention with dot, general, and concat scoring."""
    def __init__(self, enc_dim, dec_dim, method='dot'):
        super().__init__()
        self.method = method
        if method == 'general':
            self.W = nn.Linear(enc_dim, dec_dim, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(enc_dim + dec_dim, dec_dim, bias=False)
            self.v = nn.Linear(dec_dim, 1, bias=False)

    def score(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, dec_dim)
        # encoder_outputs: (batch, src_len, enc_dim)
        if self.method == 'dot':
            # (batch, src_len)
            return torch.bmm(
                encoder_outputs,
                decoder_hidden.unsqueeze(2)
            ).squeeze(2)

        elif self.method == 'general':
            # W transforms encoder outputs, then dot with decoder
            energy = self.W(encoder_outputs)  # (batch, src_len, dec_dim)
            return torch.bmm(
                energy, decoder_hidden.unsqueeze(2)
            ).squeeze(2)

        elif self.method == 'concat':
            src_len = encoder_outputs.size(1)
            dec_expanded = decoder_hidden.unsqueeze(1).expand(
                -1, src_len, -1
            )
            concat = torch.cat([dec_expanded, encoder_outputs], dim=2)
            energy = torch.tanh(self.W(concat))
            return self.v(energy).squeeze(2)

    def forward(self, decoder_hidden, encoder_outputs):
        scores = self.score(decoder_hidden, encoder_outputs)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), weights

# Compare all three variants
for method in ['dot', 'general', 'concat']:
    attn = LuongAttention(512, 512, method=method)
    enc_out = torch.randn(4, 30, 512)
    dec_h = torch.randn(4, 512)
    ctx, w = attn(dec_h, enc_out)
    n_params = sum(p.numel() for p in attn.parameters())
    print(f"{method:>8}: context={ctx.shape}, params={n_params:,}")
# Output:
#      dot: context=torch.Size([4, 512]), params=0
#  general: context=torch.Size([4, 512]), params=262,144
#   concat: context=torch.Size([4, 512]), params=525,312`,id:"code-luong-variants"}),e.jsx(n,{title:"luong_decoder_integration.py",code:`import torch
import torch.nn as nn

class LuongDecoder(nn.Module):
    """Decoder using Luong attention (compute attention AFTER RNN step)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, method='general'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.attention = LuongAttention(hidden_dim, hidden_dim, method)
        # Attentional hidden state: concat hidden + context
        self.W_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, decoder_hidden, encoder_outputs):
        embedded = self.embedding(tgt_token).unsqueeze(1)
        # Step 1: RNN step (key difference from Bahdanau)
        rnn_output, hidden = self.rnn(embedded, decoder_hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)

        # Step 2: Compute attention using CURRENT hidden state
        context, attn_weights = self.attention(hidden, encoder_outputs)

        # Step 3: Attentional vector
        attn_hidden = torch.tanh(
            self.W_c(torch.cat([hidden, context], dim=-1))
        )
        prediction = self.fc_out(attn_hidden)
        return prediction, hidden, attn_weights

decoder = LuongDecoder(8000, 256, 512, method='general')
enc_out = torch.randn(4, 25, 512)
h = torch.randn(4, 512)
tok = torch.randint(0, 8000, (4,))
pred, h_new, w = decoder(tok, h, enc_out)
print(f"Prediction: {pred.shape}")  # (4, 8000)`,id:"code-luong-decoder"}),e.jsx(a,{type:"intuition",title:"From Luong Dot-Product to Transformer Attention",content:"The dot-product scoring function in Luong attention is essentially the same as transformer attention without the scaling factor. Transformers generalize this by (1) adding the 1/sqrt(d) scaling, (2) using separate Q, K, V projections, and (3) applying multiple attention heads in parallel. Understanding Luong attention makes the transformer mechanism feel like a natural evolution.",id:"note-to-transformers"}),e.jsx(s,{title:"Bahdanau vs. Luong: Timing Matters",content:"A subtle but important difference: Bahdanau computes attention BEFORE the RNN step (using s_{t-1}), while Luong computes it AFTER (using s_t). Luong's approach is simpler and empirically performs slightly better. In practice, the Luong-style 'attend after decode' pattern became standard.",id:"warning-timing"}),e.jsx(a,{type:"historical",title:"Luong's Contribution",content:"'Effective Approaches to Attention-based Neural Machine Translation' (Luong, Pham, Manning, 2015) systematically compared attention variants and established that simple dot-product attention is highly effective. This paper, combined with Bahdanau's, formed the foundation that Vaswani et al. built upon when creating the transformer in 2017.",id:"note-luong-history"})]})}const R=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"1D Convolutions for Text"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Convolutional neural networks, originally designed for images, can be adapted for text by treating a sequence of word embeddings as a 1D signal. A 1D convolutional filter slides across the sequence to detect local n-gram patterns. Unlike RNNs, CNNs process all positions in parallel, making them significantly faster to train."}),e.jsx(i,{title:"1D Convolution for Text",definition:"Given an input sequence of embeddings $X \\in \\mathbb{R}^{L \\times d}$ and a filter $w \\in \\mathbb{R}^{k \\times d}$ of width $k$, the 1D convolution produces feature map $c_i = f(w \\cdot X_{i:i+k-1} + b)$ where $X_{i:i+k-1}$ is a window of $k$ consecutive embeddings.",notation:"$L$ = sequence length, $d$ = embedding dimension, $k$ = filter width (kernel size), $f$ = nonlinear activation.",id:"def-1d-conv"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"How 1D Conv Detects N-grams"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["A filter of width ",e.jsx(t.InlineMath,{math:"k"})," looks at ",e.jsx(t.InlineMath,{math:"k"})," consecutive tokens at a time, functioning as a learnable n-gram detector. A filter of width 2 detects bigrams, width 3 detects trigrams, and so on. Multiple filters with different widths capture patterns at different scales."]}),e.jsx(t.BlockMath,{math:"c_i = \\text{ReLU}\\left(\\sum_{j=0}^{k-1} w_j \\cdot x_{i+j} + b\\right)"}),e.jsx(t.BlockMath,{math:"\\text{Output length} = L - k + 1 \\quad \\text{(no padding)}"}),e.jsx(t.BlockMath,{math:"\\text{Output length} = L \\quad \\text{(with padding } \\lfloor k/2 \\rfloor \\text{)}"}),e.jsx(o,{title:"Trigram Detection",problem:"A filter of width 3 slides over the sentence 'I love this movie very much'. What n-grams does it capture?",steps:[{formula:"\\text{Position 0: } [\\text{I, love, this}]",explanation:"The filter sees the first trigram."},{formula:"\\text{Position 1: } [\\text{love, this, movie}]",explanation:"Slides one position right."},{formula:"\\text{Position 2: } [\\text{this, movie, very}]",explanation:"Continues sliding..."},{formula:"\\text{Position 3: } [\\text{movie, very, much}]",explanation:"Last valid window. With 6 tokens and filter width 3, we get 4 output positions."}],id:"example-trigram"}),e.jsx(n,{title:"text_1d_conv.py",code:`import torch
import torch.nn as nn

# Demonstrate 1D convolution on text embeddings
vocab_size = 10000
embed_dim = 128
seq_len = 50
batch_size = 8

# Simulate embedded text
embedding = nn.Embedding(vocab_size, embed_dim)
tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
x = embedding(tokens)  # (8, 50, 128)

# For Conv1d, input shape is (batch, channels, length)
# Channels = embed_dim, length = seq_len
x_conv = x.transpose(1, 2)  # (8, 128, 50)

# Single filter of width 3 (trigram detector)
conv = nn.Conv1d(
    in_channels=embed_dim,   # input channels = embedding dim
    out_channels=64,         # number of filters
    kernel_size=3,           # filter width (trigram)
    padding=1,               # same padding to preserve length
)
output = conv(x_conv)  # (8, 64, 50)
print(f"Input: {x_conv.shape}")   # (8, 128, 50)
print(f"Output: {output.shape}")  # (8, 64, 50)

# Multiple filter sizes for multi-scale detection
for k in [2, 3, 4, 5]:
    conv_k = nn.Conv1d(embed_dim, 32, kernel_size=k)
    out_k = conv_k(x_conv)
    print(f"  kernel={k}: output length = {out_k.shape[2]}")
# kernel=2: output length = 49
# kernel=3: output length = 48
# kernel=4: output length = 47
# kernel=5: output length = 46`,id:"code-1d-conv"}),e.jsx(n,{title:"conv_pooling_classifier.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvClassifier(nn.Module):
    """Basic 1D CNN text classifier with max-over-time pooling."""
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_size,
                 num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) token IDs
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed, seq)
        conv_out = F.relu(self.conv(emb))         # (batch, filters, L')
        # Max-over-time pooling: capture strongest activation
        pooled = conv_out.max(dim=2).values       # (batch, filters)
        return self.fc(self.dropout(pooled))

model = SimpleConvClassifier(
    vocab_size=10000, embed_dim=128,
    num_filters=100, kernel_size=3, num_classes=5
)
x = torch.randint(1, 10000, (16, 60))
print(f"Output: {model(x).shape}")  # (16, 5)`,id:"code-conv-classifier"}),e.jsx(a,{type:"intuition",title:"Max-Over-Time Pooling",content:"After convolution, we apply max pooling across the entire sequence to get one value per filter. This captures the single strongest activation of each filter, regardless of where it occurred. If a filter detects 'not good', max pooling will fire whenever that pattern appears anywhere in the review, giving translation-invariant feature detection.",id:"note-max-pooling"}),e.jsx(s,{title:"CNNs Have a Fixed Receptive Field",content:"A single convolutional layer with kernel size k can only see k consecutive tokens. To capture longer-range dependencies, you need either (1) deeper networks where multiple layers expand the receptive field, (2) larger kernels (expensive), or (3) dilated convolutions. This is a fundamental limitation compared to RNNs that can theoretically propagate information across the entire sequence.",id:"warning-receptive-field"}),e.jsx(a,{type:"tip",title:"Padding Strategy Matters",content:"Use 'same' padding (padding = kernel_size // 2) when you need the output to have the same length as input (for sequence labeling). Use no padding when you want to reduce length and will apply pooling afterward (for classification).",id:"note-padding"})]})}const W=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"TextCNN (Kim 2014)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'TextCNN, proposed by Yoon Kim in "Convolutional Neural Networks for Sentence Classification" (2014), demonstrated that a simple CNN architecture with multiple filter sizes and pre-trained word embeddings achieves excellent results on text classification. Despite its simplicity, TextCNN remains a strong baseline and is widely used in production due to its speed and effectiveness.'}),e.jsx(i,{title:"TextCNN Architecture",definition:"TextCNN applies multiple convolutional filters of different widths (e.g., 3, 4, 5) to an embedding matrix, applies ReLU and max-over-time pooling to each filter's output, concatenates the pooled features, and classifies through a fully connected layer with dropout.",id:"def-textcnn"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture Overview"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The model has four stages:"}),e.jsx(t.BlockMath,{math:"\\text{Embed} \\to \\text{Conv}_{k_1, k_2, k_3} \\to \\text{ReLU + MaxPool} \\to \\text{Concat + FC}"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["For each kernel size ",e.jsx(t.InlineMath,{math:"k_i"})," with ",e.jsx(t.InlineMath,{math:"n_f"})," filters, the convolution produces an output of shape ",e.jsx(t.InlineMath,{math:"(n_f, L - k_i + 1)"}),". Max-over-time pooling reduces each filter to a single scalar, giving ",e.jsx(t.InlineMath,{math:"n_f"})," features per kernel size. With 3 kernel sizes and 100 filters each, the final representation is 300-dimensional."]}),e.jsx(o,{title:"TextCNN Feature Dimensions",problem:"Calculate the feature dimensions for TextCNN with kernel_sizes=[3,4,5], 100 filters each, on a 60-token input.",steps:[{formula:"\\text{Conv}_{k=3}: (100, 58) \\xrightarrow{\\text{maxpool}} (100,)",explanation:"Kernel 3: 60-3+1=58 positions, max-pooled to 100 features."},{formula:"\\text{Conv}_{k=4}: (100, 57) \\xrightarrow{\\text{maxpool}} (100,)",explanation:"Kernel 4: 60-4+1=57 positions, max-pooled to 100 features."},{formula:"\\text{Conv}_{k=5}: (100, 56) \\xrightarrow{\\text{maxpool}} (100,)",explanation:"Kernel 5: 60-5+1=56 positions, max-pooled to 100 features."},{formula:"\\text{Concatenate: } 100 + 100 + 100 = 300",explanation:"Final feature vector is 300-dimensional, fed to a dropout + linear layer."}],id:"example-textcnn-dims"}),e.jsx(n,{title:"textcnn.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """Kim (2014) CNN for sentence classification."""
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_filters=100, kernel_sizes=(3, 4, 5),
                 dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # One Conv1d per kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed, seq)

        # Apply each conv + ReLU + max-pool
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))            # (batch, num_filters, L')
            p = c.max(dim=2).values          # (batch, num_filters)
            pooled.append(p)

        # Concatenate all filter outputs
        cat = torch.cat(pooled, dim=1)       # (batch, num_filters * 3)
        return self.fc(self.dropout(cat))

# Instantiate and test
model = TextCNN(
    vocab_size=25000, embed_dim=300, num_classes=5,
    num_filters=100, kernel_sizes=(3, 4, 5)
)
x = torch.randint(1, 25000, (32, 60))
logits = model(x)
print(f"Output: {logits.shape}")  # (32, 5)

# Count parameters
total = sum(p.numel() for p in model.parameters())
embed = model.embedding.weight.numel()
print(f"Total params: {total:,}")
print(f"Embedding params: {embed:,}")
print(f"CNN params: {total - embed:,}")`,id:"code-textcnn"}),e.jsx(n,{title:"textcnn_multichannel.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNMultichannel(nn.Module):
    """Kim (2014) multichannel variant: static + non-static embeddings."""
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_filters=100, kernel_sizes=(3, 4, 5)):
        super().__init__()
        # Channel 1: static (frozen pretrained)
        self.embed_static = nn.Embedding(vocab_size, embed_dim)
        self.embed_static.weight.requires_grad = False

        # Channel 2: non-static (fine-tuned pretrained)
        self.embed_nonstatic = nn.Embedding(vocab_size, embed_dim)

        # Convolutions operate on 2-channel input
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim * 2, num_filters, k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        e1 = self.embed_static(x).transpose(1, 2)
        e2 = self.embed_nonstatic(x).transpose(1, 2)
        emb = torch.cat([e1, e2], dim=1)  # (batch, 2*embed, seq)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))
            pooled.append(c.max(dim=2).values)
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))

model = TextCNNMultichannel(25000, 300, 5)
x = torch.randint(1, 25000, (16, 50))
print(f"Multichannel output: {model(x).shape}")  # (16, 5)`,id:"code-multichannel"}),e.jsx(a,{type:"note",title:"Kim's Key Findings",content:"Kim (2014) showed that: (1) Pre-trained word2vec embeddings are crucial -- random initialization loses ~2% accuracy. (2) The multichannel model (one static, one fine-tuned embedding) helps on small datasets. (3) A single-layer CNN with dropout 0.5 is sufficient -- deeper is not always better for this architecture. (4) TextCNN matches or beats more complex models on 4 out of 7 benchmarks.",id:"note-kim-findings"}),e.jsx(s,{title:"TextCNN Limitations",content:"TextCNN treats max-pooled features as a bag of detected patterns with no positional information. It cannot model word order beyond the kernel size. For tasks requiring long-range reasoning (e.g., document-level inference, coreference resolution), RNNs or transformers are needed.",id:"warning-textcnn-limits"}),e.jsx(a,{type:"historical",title:"TextCNN's Lasting Impact",content:"Despite being published in 2014, TextCNN remains surprisingly competitive. It is still used as a baseline in NLP papers and deployed in production for tasks where speed matters (content moderation, spam detection). The key lesson: for classification tasks with short texts, local n-gram features captured by CNNs are often sufficient.",id:"note-textcnn-impact"})]})}const B=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Dilated Convolutions"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["Dilated (atrous) convolutions expand the receptive field exponentially without increasing the number of parameters or losing resolution. By inserting gaps between filter elements, a dilated convolution with dilation rate ",e.jsx(t.InlineMath,{math:"d"})," and kernel size ",e.jsx(t.InlineMath,{math:"k"})," has an effective receptive field of ",e.jsx(t.InlineMath,{math:"k + (k-1)(d-1)"})," tokens. Stacking layers with exponentially growing dilation rates allows a network to model very long-range dependencies efficiently."]}),e.jsx(i,{title:"Dilated Convolution",definition:"A dilated convolution with dilation rate $d$ applies the filter at every $d$-th position: $(x *_d w)_t = \\sum_{j=0}^{k-1} w_j \\cdot x_{t - d \\cdot j}$. Standard convolution is the special case $d=1$.",notation:"$d$ = dilation rate, $k$ = kernel size. Effective receptive field = $k + (k-1)(d-1)$.",id:"def-dilated-conv"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Exponential Receptive Field Growth"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["By stacking layers with dilation rates ",e.jsx(t.InlineMath,{math:"1, 2, 4, 8, \\ldots"}),", the receptive field grows exponentially while the number of parameters grows only linearly:"]}),e.jsx(t.BlockMath,{math:"\\text{Receptive field after } L \\text{ layers} = 1 + k \\cdot \\sum_{l=0}^{L-1}(d_l - 1) + (k-1) \\cdot L"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"With kernel size 3 and dilation rates [1, 2, 4, 8], four layers cover a receptive field of 31 tokens, compared to just 9 for standard convolutions."}),e.jsx(o,{title:"Receptive Field Comparison",problem:"Compare receptive fields for 4 stacked conv layers (kernel=3) with vs. without dilation.",steps:[{formula:"\\text{Standard (d=1): } 3 + 2 + 2 + 2 = 9 \\text{ tokens}",explanation:"Each standard conv layer adds (k-1)=2 to the receptive field."},{formula:"\\text{Dilated (d=1,2,4,8): } 3 + 4 + 8 + 16 = 31 \\text{ tokens}",explanation:"Dilation doubles the receptive field contribution of each layer."},{formula:"\\text{Parameters: identical } (4 \\times 3 \\times d_{\\text{model}}^2)",explanation:"Both use the same kernel size and number of layers -- same parameter count!"},{formula:"\\text{Ratio: } 31 / 9 \\approx 3.4\\times",explanation:"Dilated convolutions cover 3.4x more context with zero extra parameters."}],id:"example-receptive-field"}),e.jsx(n,{title:"dilated_conv_demo.py",code:`import torch
import torch.nn as nn

# Compare standard vs. dilated convolution
embed_dim = 128
x = torch.randn(4, embed_dim, 100)  # (batch, channels, seq_len)

# Standard convolution stack
conv_standard = nn.Sequential(
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
)

# Dilated convolution stack (note: padding = dilation for 'same' output)
conv_dilated = nn.Sequential(
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=4, dilation=4),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=8, dilation=8),
)

out_std = conv_standard(x)
out_dil = conv_dilated(x)
print(f"Standard output: {out_std.shape}")  # (4, 128, 100)
print(f"Dilated output:  {out_dil.shape}")  # (4, 128, 100)

# Same parameters, but dilated has much larger receptive field
params_std = sum(p.numel() for p in conv_standard.parameters())
params_dil = sum(p.numel() for p in conv_dilated.parameters())
print(f"Standard params: {params_std:,}")  # 262,656
print(f"Dilated params:  {params_dil:,}")  # 262,656 -- identical!`,id:"code-dilated-demo"}),e.jsx(n,{title:"dilated_causal_conv.py",code:`import torch
import torch.nn as nn

class CausalDilatedConv(nn.Module):
    """Causal dilated convolution block (WaveNet-style)."""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        # Causal padding: pad only on the left side
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        # Left-pad for causal convolution
        x_padded = nn.functional.pad(x, (self.padding, 0))
        out = self.conv(x_padded)
        # Residual connection
        out = out + x
        # Layer norm (transpose for channel-last)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return torch.relu(out)

class DilatedConvStack(nn.Module):
    """Stack of causal dilated convolutions with residual connections."""
    def __init__(self, vocab_size, embed_dim, num_layers=6,
                 num_classes=5, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            CausalDilatedConv(embed_dim, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed, seq)
        h = emb
        for layer in self.layers:
            h = layer(h)
        # Global average pooling
        pooled = h.mean(dim=2)  # (batch, embed)
        return self.fc(pooled)

model = DilatedConvStack(10000, 128, num_layers=6, num_classes=5)
x = torch.randint(1, 10000, (8, 200))
print(f"Output: {model(x).shape}")  # (8, 5)
# Receptive field: 1 + 2*(1+2+4+8+16+32) = 127 tokens with just 6 layers!
print(f"Receptive field: {1 + 2*(1+2+4+8+16+32)} tokens")`,id:"code-causal-dilated"}),e.jsx(a,{type:"intuition",title:"Why Dilation Works",content:"Imagine reading a document by looking at every word (standard), versus sampling every 2nd, 4th, and 8th word in successive passes. Each pass captures patterns at a different scale. Dilation lets each layer specialize: lower layers detect local patterns (bigrams, trigrams), while higher layers with large dilation rates detect document-level themes.",id:"note-dilation-intuition"}),e.jsx(s,{title:"Gridding Artifacts",content:"When dilation rates are too large or poorly chosen, dilated convolutions can develop 'gridding artifacts' where some input positions are never covered by any filter. A common mitigation is to repeat the dilation pattern (e.g., [1,2,4,1,2,4]) or use hybrid architectures that combine dilated and standard convolutions.",id:"warning-gridding"}),e.jsx(a,{type:"historical",title:"From WaveNet to Text",content:"Dilated causal convolutions were popularized by WaveNet (van den Oord et al., 2016) for audio generation. The ByteNet architecture (Kalchbrenner et al., 2017) adapted this for NLP, achieving competitive machine translation results. These architectures inspired the idea that non-recurrent models could handle sequences effectively, paving the way for transformers.",id:"note-wavenet-history"})]})}const A=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Comparing CNNs vs RNNs for NLP"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"CNNs and RNNs represent two fundamentally different approaches to processing text. RNNs model sequences through recurrence and maintain an evolving hidden state, while CNNs detect local patterns through convolution and build global understanding through hierarchical stacking. Understanding their tradeoffs explains why both were eventually superseded by transformers, which combine the strengths of each."}),e.jsx(i,{title:"Computational Complexity Comparison",definition:"For a sequence of length $L$, hidden dimension $d$, and kernel size $k$: RNN has $O(L \\cdot d^2)$ sequential operations with path length $O(L)$. CNN has $O(k \\cdot L \\cdot d^2)$ parallelizable operations with path length $O(\\log_k L)$ for dilated convolutions.",id:"def-complexity"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Key Tradeoffs"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The fundamental difference is ",e.jsx("strong",{children:"sequential vs. parallel"})," computation. RNNs must process tokens one at a time (O(L) sequential steps), while CNNs process all positions simultaneously (O(1) sequential steps per layer, O(L/k) layers for full coverage)."]}),e.jsx(o,{title:"Training Speed Comparison",problem:"Compare wall-clock time to process a batch of 32 sequences of length 200 with hidden_dim=256.",steps:[{formula:"\\text{LSTM: } 200 \\text{ sequential steps} \\times O(d^2)",explanation:"Each step depends on the previous -- cannot parallelize across time steps."},{formula:"\\text{CNN (k=3): } O(1) \\text{ steps} \\times O(k \\cdot d^2)",explanation:"All positions computed simultaneously within each layer."},{formula:"\\text{Empirical: CNN } \\approx 3\\text{-}10\\times \\text{ faster}",explanation:"On GPU, CNNs exploit parallelism far better. LSTM achieves ~20K tokens/sec vs CNN ~100K tokens/sec."},{formula:"\\text{But: CNN needs more layers for global context}",explanation:"A single CNN layer only sees k tokens. Need O(log L) dilated layers for full coverage."}],id:"example-speed"}),e.jsx(n,{title:"benchmark_cnn_vs_rnn.py",code:`import torch
import torch.nn as nn
import time

def benchmark(model, x, name, n_runs=100):
    """Benchmark forward pass speed."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    print(f"{name}: {elapsed:.2f} ms/batch")

# Define comparable models
class LSTMModel(nn.Module):
    def __init__(self, vocab, embed, hidden, classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, classes)
    def forward(self, x):
        out, (h, _) = self.lstm(self.emb(x))
        return self.fc(h[-1])

class CNNModel(nn.Module):
    def __init__(self, vocab, embed, hidden, classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed, hidden // 3, k) for k in [3, 4, 5]
        ])
        self.fc = nn.Linear((hidden // 3) * 3, classes)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        pooled = [torch.relu(c(e)).max(2).values for c in self.convs]
        return self.fc(torch.cat(pooled, 1))

V, E, H, C = 10000, 128, 256, 5
lstm = LSTMModel(V, E, H, C)
cnn = CNNModel(V, E, H, C)
x = torch.randint(0, V, (32, 200))

benchmark(lstm, x, "LSTM")
benchmark(cnn, x, "CNN")
# Typical CPU results:
#   LSTM: ~45 ms/batch
#   CNN:  ~8 ms/batch (5-6x faster)`,id:"code-benchmark"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"When to Use Which"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The choice between CNN and RNN depends on the task characteristics:"}),e.jsx(n,{title:"task_comparison.py",code:`# Decision guide for CNN vs RNN (pre-transformer era)

task_recommendations = {
    "Sentiment analysis (short text)": {
        "winner": "CNN",
        "reason": "Local patterns (n-grams) are sufficient; speed matters"
    },
    "Named entity recognition": {
        "winner": "BiLSTM",
        "reason": "Needs full sequence context for each token prediction"
    },
    "Machine translation": {
        "winner": "LSTM + Attention",
        "reason": "Variable-length output, long-range dependencies"
    },
    "Document classification": {
        "winner": "CNN or hierarchical",
        "reason": "Key phrases can appear anywhere; max-pooling captures this"
    },
    "Language modeling": {
        "winner": "LSTM",
        "reason": "Autoregressive generation requires sequential processing"
    },
    "Spam detection": {
        "winner": "CNN",
        "reason": "Pattern matching (keywords, phrases); speed is critical"
    },
}

for task, info in task_recommendations.items():
    print(f"{task}")
    print(f"  Recommended: {info['winner']}")
    print(f"  Reason: {info['reason']}\\n")

# Note: In the transformer era (post-2018), transformers dominate
# nearly all of these tasks. But understanding CNN/RNN tradeoffs
# illuminates WHY transformers work: they combine parallel computation
# (like CNNs) with global attention (like RNNs).`,id:"code-comparison"}),e.jsx(a,{type:"intuition",title:"Why Transformers Won",content:"RNNs have O(L) sequential path length (slow training) but constant-time long-range connections. CNNs have O(1) parallel computation but O(log L) path length for long-range connections. Transformers achieve BOTH: O(1) parallel computation across all positions AND O(1) path length between any two tokens via direct attention. They combine the parallelism of CNNs with the global context of RNNs.",id:"note-why-transformers"}),e.jsx(s,{title:"Benchmarks Can Be Misleading",content:"Raw speed comparisons between CNNs and RNNs depend heavily on sequence length, batch size, hidden dimensions, and hardware (CPU vs GPU). CNNs are particularly advantaged on GPUs due to parallelism. On very short sequences (< 20 tokens), the overhead of multiple CNN layers can make LSTMs competitive.",id:"warning-benchmarks"}),e.jsx(a,{type:"note",title:"The Hybrid Approach",content:"Some architectures combine CNNs and RNNs: use CNN layers to extract local features and reduce sequence length, then feed the output to an RNN for global reasoning. The RCNN (Lai et al., 2015) and CNN-LSTM models were popular hybrids. Modern architectures like Mamba and RWKV revisit the RNN idea with better parallelism.",id:"note-hybrid"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Loss Functions for Language Tasks"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The choice of loss function determines what the model optimizes for. In NLP, cross-entropy loss dominates classification and language modeling tasks, while specialized losses like CTC handle sequence alignment problems. Understanding these losses is critical for training effective models and diagnosing training issues."}),e.jsx(i,{title:"Cross-Entropy Loss",definition:"For a classification problem with $C$ classes, the cross-entropy loss between the predicted distribution $\\hat{y}$ and true label $y$ is $\\mathcal{L} = -\\sum_{c=1}^{C} y_c \\log(\\hat{y}_c)$. For hard labels (one-hot), this simplifies to $\\mathcal{L} = -\\log(\\hat{y}_{y^*})$ where $y^*$ is the true class.",notation:"$\\hat{y}_c = \\text{softmax}(z)_c = \\frac{e^{z_c}}{\\sum_j e^{z_j}}$, where $z$ are the logits.",id:"def-cross-entropy"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Cross-Entropy for Language Modeling"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["In language modeling, cross-entropy is computed at each position in the sequence. The model predicts a distribution over the vocabulary",e.jsx(t.InlineMath,{math:"V"})," at each step, and the loss is the negative log probability of the correct next token:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{LM}} = -\\frac{1}{T}\\sum_{t=1}^{T} \\log P(w_t | w_1, \\ldots, w_{t-1})"}),e.jsx(t.BlockMath,{math:"\\text{Perplexity} = \\exp(\\mathcal{L}_{\\text{LM}}) = \\exp\\left(-\\frac{1}{T}\\sum_{t=1}^{T} \\log P(w_t | w_{<t})\\right)"}),e.jsx(o,{title:"Cross-Entropy Calculation",problem:"A model predicts P('cat')=0.7, P('dog')=0.2, P('fish')=0.1 and the true word is 'cat'. What is the loss?",steps:[{formula:"\\mathcal{L} = -\\log(0.7) = 0.357",explanation:"Loss is the negative log probability of the correct class."},{formula:'\\text{If true word were "fish": } \\mathcal{L} = -\\log(0.1) = 2.303',explanation:"Lower probability assignments yield higher loss."},{formula:"\\text{Perfect prediction: } \\mathcal{L} = -\\log(1.0) = 0",explanation:"Zero loss when the model assigns probability 1 to the correct class."},{formula:"\\text{Perplexity} = e^{0.357} = 1.43",explanation:"Perplexity of 1.43 means the model is as confused as choosing uniformly among 1.43 options."}],id:"example-ce-calc"}),e.jsx(n,{title:"cross_entropy_pytorch.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch cross-entropy combines log_softmax + NLLLoss
criterion = nn.CrossEntropyLoss()

# Classification example
logits = torch.tensor([[2.0, 1.0, 0.1]])  # raw model outputs
target = torch.tensor([0])                  # true class = 0
loss = criterion(logits, target)
print(f"Classification loss: {loss.item():.4f}")  # ~0.4170

# Manual verification
probs = F.softmax(logits, dim=-1)
manual_loss = -torch.log(probs[0, target[0]])
print(f"Manual loss: {manual_loss.item():.4f}")   # same

# Language modeling: loss over entire sequence
vocab_size = 10000
seq_len = 50
batch_size = 8

# Model outputs: (batch, seq_len, vocab_size)
logits_lm = torch.randn(batch_size, seq_len, vocab_size)
targets_lm = torch.randint(0, vocab_size, (batch_size, seq_len))

# Reshape for CrossEntropyLoss: (N, C) and (N,)
loss_lm = criterion(
    logits_lm.view(-1, vocab_size),
    targets_lm.view(-1)
)
perplexity = torch.exp(loss_lm)
print(f"LM loss: {loss_lm.item():.2f}")
print(f"Perplexity: {perplexity.item():.1f}")

# With padding: ignore padding tokens (index 0)
criterion_pad = nn.CrossEntropyLoss(ignore_index=0)
targets_padded = targets_lm.clone()
targets_padded[:, 40:] = 0  # simulate padding
loss_padded = criterion_pad(
    logits_lm.view(-1, vocab_size),
    targets_padded.view(-1)
)
print(f"Loss ignoring padding: {loss_padded.item():.2f}")`,id:"code-cross-entropy"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"CTC Loss"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Connectionist Temporal Classification (CTC) loss handles sequence-to-sequence alignment when the input and output lengths differ and the alignment is unknown. It marginalizes over all possible alignments, making it ideal for speech recognition and OCR."}),e.jsx(t.BlockMath,{math:"P(y | x) = \\sum_{\\pi \\in \\mathcal{B}^{-1}(y)} P(\\pi | x)"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["where ",e.jsx(t.InlineMath,{math:"\\mathcal{B}^{-1}(y)"})," is the set of all valid alignments that collapse to the target ",e.jsx(t.InlineMath,{math:"y"})," after removing blanks and repeated characters."]}),e.jsx(n,{title:"ctc_and_label_smoothing.py",code:`import torch
import torch.nn as nn

# CTC Loss for sequence alignment (e.g., speech recognition)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

T = 50        # input sequence length
C = 28        # number of classes (alphabet + blank)
batch_size = 4
target_lengths = torch.tensor([5, 7, 4, 6])
input_lengths = torch.full((batch_size,), T, dtype=torch.long)

# Log probabilities: (T, batch, C) -- note: time-first!
log_probs = torch.randn(T, batch_size, C).log_softmax(dim=2)
targets = torch.randint(1, C, (sum(target_lengths),))  # skip blank=0

loss_ctc = ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(f"CTC Loss: {loss_ctc.item():.2f}")

# Label Smoothing: prevents overconfident predictions
# Distributes (1 - smoothing) to true label, smoothing / (C-1) to others
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)
        # True class gets (1 - smoothing)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1))
        nll = nll.squeeze(-1)
        # Uniform gets smoothing
        smooth = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()

criterion_ls = LabelSmoothingLoss(classes=1000, smoothing=0.1)
logits = torch.randn(32, 1000)
targets = torch.randint(0, 1000, (32,))
print(f"Label smoothing loss: {criterion_ls(logits, targets).item():.2f}")`,id:"code-ctc-smoothing"}),e.jsx(a,{type:"tip",title:"Label Smoothing in Practice",content:"Label smoothing with epsilon=0.1 is standard for transformer training. Instead of training toward a one-hot target, it distributes 10% of the probability mass uniformly. This prevents the model from becoming overconfident, improves calibration, and acts as a regularizer. Vaswani et al. (2017) used it in the original transformer paper.",id:"note-label-smoothing"}),e.jsx(s,{title:"Numerical Stability",content:"Never compute cross-entropy as -log(softmax(x)). Instead, use log_softmax or PyTorch's CrossEntropyLoss, which applies the log-sum-exp trick for numerical stability. Direct computation of softmax followed by log can produce NaN or Inf for large logits due to floating-point overflow.",id:"warning-numerical"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Optimizers: Adam and AdamW"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Optimizers determine how model parameters are updated based on computed gradients. Adam (Adaptive Moment Estimation) became the default optimizer for deep learning due to its adaptive learning rates and momentum. AdamW fixes a subtle but important issue with weight decay in Adam and is now the standard for training transformers."}),e.jsx(i,{title:"Adam Optimizer",definition:"Adam maintains per-parameter exponential moving averages of the gradient (first moment $m_t$) and squared gradient (second moment $v_t$), with bias correction. The update rule is $\\theta_{t+1} = \\theta_t - \\eta \\cdot \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)$.",notation:"$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t$, $v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2$, $\\hat{m}_t = m_t/(1-\\beta_1^t)$, $\\hat{v}_t = v_t/(1-\\beta_2^t)$.",id:"def-adam"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Adam Update Equations"}),e.jsx(t.BlockMath,{math:"m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\quad \\text{(first moment)}"}),e.jsx(t.BlockMath,{math:"v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\quad \\text{(second moment)}"}),e.jsx(t.BlockMath,{math:"\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\quad \\text{(bias correction)}"}),e.jsx(t.BlockMath,{math:"\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t"}),e.jsx(o,{title:"Why Bias Correction Matters",problem:"Show why m_t is biased toward zero at the start of training (beta_1=0.9).",steps:[{formula:"m_1 = 0.9 \\times 0 + 0.1 \\times g_1 = 0.1 g_1",explanation:"After step 1, the moving average is only 10% of the true gradient due to initialization at 0."},{formula:"\\hat{m}_1 = m_1 / (1 - 0.9^1) = 0.1g_1 / 0.1 = g_1",explanation:"Bias correction divides by (1 - beta^t), restoring the true scale."},{formula:"m_{10} \\approx 0.65 g_{\\text{avg}}",explanation:"After 10 steps, the bias is smaller but still present."},{formula:"\\hat{m}_{10} = m_{10} / (1 - 0.9^{10}) = m_{10} / 0.65 \\approx g_{\\text{avg}}",explanation:"Correction factor (1 - 0.9^10) = 0.65 exactly compensates the bias."}],id:"example-bias-correction"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"AdamW: Decoupled Weight Decay"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Loshchilov and Hutter (2019) showed that L2 regularization and weight decay are",e.jsx("em",{children:" not equivalent"})," in Adam. Standard Adam with L2 regularization adds the penalty to the gradient, which gets scaled by the adaptive learning rate. AdamW instead applies weight decay directly to the parameters:"]}),e.jsx(t.BlockMath,{math:"\\text{Adam + L2: } g_t = \\nabla \\mathcal{L}(\\theta_t) + \\lambda \\theta_t \\quad \\text{(coupled)}"}),e.jsx(t.BlockMath,{math:"\\text{AdamW: } \\theta_{t+1} = \\theta_t - \\eta\\left(\\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_t\\right) \\quad \\text{(decoupled)}"}),e.jsx(n,{title:"adam_vs_adamw.py",code:`import torch
import torch.nn as nn

# Standard Adam with L2 regularization (NOT recommended)
model_l2 = nn.Linear(100, 10)
optimizer_l2 = torch.optim.Adam(
    model_l2.parameters(),
    lr=1e-3,
    weight_decay=0.01  # This is L2 in Adam, NOT true weight decay!
)

# AdamW with decoupled weight decay (RECOMMENDED)
model_wd = nn.Linear(100, 10)
optimizer_wd = torch.optim.AdamW(
    model_wd.parameters(),
    lr=1e-3,
    weight_decay=0.01  # True decoupled weight decay
)

# Typical transformer training configuration
model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),   # default momentum parameters
    eps=1e-8,              # numerical stability
    weight_decay=0.01      # standard for transformers
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Optimizer states per param: 2 (m_t and v_t)")
print(f"Optimizer memory: ~{sum(p.numel() for p in model.parameters()) * 2 * 4 / 1e6:.1f} MB")`,id:"code-adam-adamw"}),e.jsx(n,{title:"optimizer_param_groups.py",code:`import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 512)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        self.ln = nn.LayerNorm(512)
        self.head = nn.Linear(512, 30000)

    def forward(self, x):
        return self.head(self.ln(self.layers(self.embedding(x))))

model = TransformerLM()

# Separate parameter groups: no weight decay for biases and LayerNorm
no_decay = ['bias', 'ln', 'LayerNorm']
param_groups = [
    {
        'params': [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,  # no decay for biases and norms
    },
]

optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.98))

# Count parameters in each group
for i, group in enumerate(param_groups):
    n_params = sum(p.numel() for p in group['params'])
    print(f"Group {i}: {n_params:,} params, wd={group['weight_decay']}")`,id:"code-param-groups"}),e.jsx(a,{type:"intuition",title:"Why Adaptive Learning Rates Help NLP",content:"Word embeddings for rare words receive gradients very infrequently. With SGD, these embeddings are barely updated. Adam's per-parameter adaptive rate means rare-word embeddings get larger effective learning rates (due to small v_t), while frequent-word embeddings get smaller rates. This is why Adam converges much faster than SGD on NLP tasks.",id:"note-adaptive-nlp"}),e.jsx(s,{title:"Adam's Memory Overhead",content:"Adam stores two additional tensors (m_t and v_t) per parameter, tripling memory compared to SGD. For a 7B parameter model in fp32, parameters take 28GB, and Adam states add 56GB more, totaling 84GB. This is why mixed-precision training and optimizer offloading are essential for large models.",id:"warning-adam-memory"}),e.jsx(a,{type:"tip",title:"Hyperparameter Defaults",content:"For most NLP tasks, these defaults work well: lr=1e-4 to 3e-4, beta1=0.9, beta2=0.999 (or 0.98 for transformers), eps=1e-8, weight_decay=0.01 to 0.1. The learning rate is the most important hyperparameter to tune. Always use a learning rate schedule (warmup + decay) rather than a constant rate.",id:"note-defaults"})]})}const P=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Learning Rate Schedules"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Learning rate schedules adjust the learning rate during training to improve convergence and final performance. A warmup phase prevents early instability when gradients are noisy, while decay phases allow the model to settle into sharper minima. The right schedule can mean the difference between a good model and a great one."}),e.jsx(i,{title:"Learning Rate Warmup",definition:"Warmup linearly increases the learning rate from 0 (or a small value) to the peak rate over the first $W$ training steps: $\\eta_t = \\eta_{\\max} \\cdot \\min(1, t / W)$. This prevents large, unstable updates early in training when the model weights are random and gradients are unreliable.",notation:"$W$ = warmup steps, $\\eta_{\\max}$ = peak learning rate, $t$ = current step.",id:"def-warmup"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Common Schedules"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The transformer paper (Vaswani et al., 2017) introduced a specific schedule combining warmup with inverse square root decay:"}),e.jsx(t.BlockMath,{math:"\\eta_t = d_{\\text{model}}^{-0.5} \\cdot \\min(t^{-0.5}, t \\cdot W^{-1.5})"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Modern practice favors cosine decay after warmup:"}),e.jsx(t.BlockMath,{math:"\\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min})\\left(1 + \\cos\\left(\\frac{t - W}{T - W}\\pi\\right)\\right)"}),e.jsx(o,{title:"Warmup + Cosine Schedule",problem:"For a 100K step training run with 2K warmup steps, lr_max=3e-4, lr_min=1e-5, trace the learning rate.",steps:[{formula:"t=0: \\eta = 0",explanation:"Training starts with zero learning rate."},{formula:"t=1000: \\eta = 1.5 \\times 10^{-4}",explanation:"Halfway through warmup, LR is half of peak."},{formula:"t=2000: \\eta = 3 \\times 10^{-4}",explanation:"End of warmup, LR reaches peak value."},{formula:"t=51000: \\eta \\approx 1.55 \\times 10^{-4}",explanation:"Halfway through cosine decay, LR is approximately halfway between max and min."},{formula:"t=100000: \\eta = 1 \\times 10^{-5}",explanation:"End of training, LR reaches minimum value."}],id:"example-cosine"}),e.jsx(n,{title:"lr_schedules.py",code:`import torch
import torch.optim as optim
import math

# 1. Linear warmup + cosine decay (most common for transformers)
def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 2. Transformer (Vaswani) schedule
def get_transformer_schedule(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5),
                                        step * warmup_steps ** (-1.5))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 3. Linear warmup + linear decay
def get_linear_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Demo: print LR at key points
model = torch.nn.Linear(512, 512)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = get_cosine_schedule(optimizer, warmup_steps=2000,
                                 total_steps=100000)

steps_to_check = [0, 500, 1000, 2000, 10000, 50000, 90000, 100000]
for step in steps_to_check:
    # Reset and step to target
    for pg in optimizer.param_groups:
        pg['lr'] = 3e-4
    scheduler = get_cosine_schedule(optimizer, 2000, 100000)
    for _ in range(step):
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"Step {step:>6d}: lr = {lr:.6f}")`,id:"code-lr-schedules"}),e.jsx(n,{title:"lr_schedule_training_loop.py",code:`import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Full training loop with LR scheduling
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=4),
    num_layers=4
)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# OneCycleLR: warm up then anneal in one cycle (popular for fine-tuning)
total_steps = 10000
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=total_steps,
    pct_start=0.1,     # 10% warmup
    anneal_strategy='cos',
    div_factor=25,      # initial_lr = max_lr / 25
    final_div_factor=1000,  # final_lr = max_lr / 25 / 1000
)

# Training loop skeleton
criterion = nn.CrossEntropyLoss()
for step in range(total_steps):
    optimizer.zero_grad()
    # Simulate forward pass
    x = torch.randn(8, 32, 256)
    out = model(x)
    loss = out.sum()  # dummy loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()  # update LR AFTER optimizer step

    if step % 2000 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Step {step}: lr={lr:.6f}, loss={loss.item():.4f}")`,id:"code-training-loop"}),e.jsx(a,{type:"intuition",title:"Why Warmup Prevents Instability",content:"At initialization, the model's weights are random and produce near-uniform attention distributions. Gradients in this regime are noisy and can have very large magnitude. A large learning rate would cause wild parameter updates that the model never recovers from. Warmup lets the model first find a reasonable region of parameter space before taking large steps.",id:"note-warmup-intuition"}),e.jsx(s,{title:"Warmup Steps Must Scale with Model Size",content:"Larger models need more warmup steps. GPT-3 used 375M tokens of warmup (~375 steps at batch size 1M tokens). BERT used 10K warmup steps. Too few warmup steps cause training divergence, especially with large learning rates. A common heuristic: warmup for 1-5% of total training steps.",id:"warning-warmup-scaling"}),e.jsx(a,{type:"tip",title:"Cosine vs. Linear Decay",content:"Cosine decay keeps the learning rate higher for longer before a smooth decline at the end, while linear decay drops steadily. Empirically, cosine decay tends to produce slightly better results for transformer training. For fine-tuning on small datasets, linear decay is often sufficient.",id:"note-cosine-vs-linear"})]})}const D=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Gradient Clipping and Regularization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Gradient clipping prevents training instability by rescaling gradients when they exceed a threshold. Combined with regularization techniques like dropout and weight decay, these methods form the foundation of stable neural network training. Every modern LLM training run relies on gradient clipping to prevent occasional large gradients from derailing optimization."}),e.jsx(i,{title:"Gradient Clipping by Norm",definition:"Clip-by-norm rescales the entire gradient vector when its L2 norm exceeds a threshold $\\tau$: $g \\leftarrow g \\cdot \\frac{\\tau}{\\|g\\|}$ if $\\|g\\| > \\tau$. This preserves the gradient direction while bounding its magnitude.",notation:"$g = \\nabla_\\theta \\mathcal{L}$, $\\tau$ = max norm threshold, $\\|g\\| = \\sqrt{\\sum_i g_i^2}$.",id:"def-clip-norm"}),e.jsx(i,{title:"Gradient Clipping by Value",definition:"Clip-by-value independently clamps each gradient component to $[-\\tau, \\tau]$: $g_i \\leftarrow \\max(-\\tau, \\min(\\tau, g_i))$. This changes the gradient direction but is simpler to implement.",id:"def-clip-value"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Clip by Norm vs. Clip by Value"}),e.jsx(t.BlockMath,{math:"\\text{By norm: } g \\leftarrow \\begin{cases} g & \\text{if } \\|g\\|_2 \\leq \\tau \\\\ \\tau \\cdot \\frac{g}{\\|g\\|_2} & \\text{if } \\|g\\|_2 > \\tau \\end{cases}"}),e.jsx(t.BlockMath,{math:"\\text{By value: } g_i \\leftarrow \\text{clip}(g_i, -\\tau, \\tau) \\quad \\forall i"}),e.jsx(o,{title:"Clipping in Practice",problem:"A gradient vector g = [3.0, 4.0] has norm 5.0. Apply clip-by-norm with tau=2.0 and clip-by-value with tau=2.0.",steps:[{formula:"\\|g\\| = \\sqrt{9 + 16} = 5.0 > 2.0",explanation:"Norm exceeds threshold, clipping will be applied."},{formula:"\\text{By norm: } g \\leftarrow [3, 4] \\times 2/5 = [1.2, 1.6]",explanation:"Both components scaled equally. Direction preserved: still points the same way."},{formula:"\\text{By value: } g \\leftarrow [\\min(3, 2), \\min(4, 2)] = [2.0, 2.0]",explanation:"Components clamped independently. Direction changes: original angle was ~53deg, now 45deg."},{formula:"\\text{By norm } \\|g\\| = 2.0, \\text{ by value } \\|g\\| = 2.83",explanation:"Clip-by-norm gives exact norm control; clip-by-value does not."}],id:"example-clipping"}),e.jsx(n,{title:"gradient_clipping.py",code:`import torch
import torch.nn as nn

model = nn.LSTM(128, 256, num_layers=3, batch_first=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Simulate a training step
x = torch.randn(8, 50, 128)
output, _ = model(x)
target = torch.randint(0, 256, (8, 50))
loss = criterion(output.reshape(-1, 256), target.reshape(-1))
loss.backward()

# Method 1: Clip by global norm (RECOMMENDED)
# Returns the total norm before clipping
total_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0      # typical values: 0.5, 1.0, 5.0
)
print(f"Gradient norm before clip: {total_norm:.4f}")

optimizer.step()
optimizer.zero_grad()

# Method 2: Clip by value
loss.backward()  # recompute gradients
torch.nn.utils.clip_grad_value_(
    model.parameters(),
    clip_value=0.5    # clamp each element to [-0.5, 0.5]
)

# Method 3: Manual clipping (for monitoring)
loss.backward()
grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        grad_norm += p.grad.data.norm(2).item() ** 2
grad_norm = grad_norm ** 0.5
print(f"Manual grad norm: {grad_norm:.4f}")

# Clip
max_norm = 1.0
if grad_norm > max_norm:
    scale = max_norm / grad_norm
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(scale)`,id:"code-clipping"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Regularization Techniques"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Beyond gradient clipping, regularization prevents overfitting and improves generalization. The key techniques for NLP are dropout, weight decay, and layer normalization."}),e.jsx(n,{title:"regularization_techniques.py",code:`import torch
import torch.nn as nn

class RegularizedLSTM(nn.Module):
    """LSTM with multiple regularization strategies."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Embedding dropout: randomly zero entire word embeddings
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,    # dropout between LSTM layers
        )
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Output dropout
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embed_dropout(self.embedding(x))
        output, (h_n, _) = self.lstm(emb)
        h = self.layer_norm(h_n[-1])
        return self.fc(self.output_dropout(h))

# Full training setup with all regularization
model = RegularizedLSTM(10000, 128, 256, 5, dropout=0.3)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # L2 regularization (decoupled)
)

# Training loop with gradient monitoring
for epoch in range(5):
    model.train()
    x = torch.randint(1, 10000, (32, 100))
    y = torch.randint(0, 5, (32,))

    optimizer.zero_grad()
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    # Monitor and clip
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )
    optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}, "
          f"grad_norm={grad_norm:.4f}")`,id:"code-regularization"}),e.jsx(a,{type:"tip",title:"Standard Clipping Values",content:"For RNNs/LSTMs, clip-by-norm with max_norm=5.0 is common (Pascanu et al., 2013). For transformers, max_norm=1.0 is standard (used in GPT-2, GPT-3, BERT). For fine-tuning pre-trained models, max_norm=1.0 with a lower learning rate (2e-5 to 5e-5) works well. Always log gradient norms during training -- sudden spikes indicate instability.",id:"note-clipping-values"}),e.jsx(s,{title:"Clipping Hides Problems",content:"If gradient norms are consistently being clipped (e.g., > 50% of steps), the learning rate is likely too high, the model architecture has issues, or the data contains anomalous samples. Clipping is a safety net, not a solution. Investigate persistent clipping by examining loss spikes and adjusting hyperparameters.",id:"warning-hiding-problems"}),e.jsx(a,{type:"note",title:"Gradient Norm as a Training Diagnostic",content:"Logging gradient norms over training reveals important patterns: (1) Norms should generally decrease as training converges. (2) Sudden spikes may indicate bad data batches or learning rate issues. (3) Very small norms suggest vanishing gradients or a loss plateau. Tools like Weights & Biases and TensorBoard make this monitoring trivial.",id:"note-monitoring"})]})}const G=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));export{L as a,M as b,C as c,$ as d,S as e,q as f,R as g,W as h,B as i,A as j,E as k,O as l,P as m,D as n,G as o,z as s};
