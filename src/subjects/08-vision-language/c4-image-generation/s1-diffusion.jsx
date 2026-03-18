import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function DiffusionBasics() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Diffusion Process Basics</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Diffusion models generate data by learning to reverse a gradual noising process.
        Starting from pure Gaussian noise, the model iteratively denoises to produce realistic
        images. The mathematical framework connects forward noise addition (a Markov chain)
        with a learned reverse process, yielding state-of-the-art generative quality.
      </p>

      <DefinitionBlock
        title="Forward Diffusion Process"
        definition="The forward process gradually adds Gaussian noise to data $\mathbf{x}_0$ over $T$ timesteps according to a variance schedule $\{\beta_t\}_{t=1}^T$. At each step: $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1}, \beta_t \mathbf{I})$."
        notation="Let \( \alpha_t = 1 - \beta_t \) and \( \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s \). Then \( q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \)"
        id="def-forward-diffusion"
      />

      <h2 className="text-2xl font-semibold">The Reparameterization Trick</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A key insight is that we can sample <InlineMath math="\mathbf{x}_t" /> directly from{' '}
        <InlineMath math="\mathbf{x}_0" /> without iterating through all intermediate steps:
      </p>
      <BlockMath math="\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})" />

      <TheoremBlock
        title="DDPM Training Objective"
        statement="The simplified DDPM loss trains a network $\epsilon_\theta$ to predict the noise added at timestep $t$:"
        proof="\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]"
        corollaries={[
          'This is equivalent to learning the score function (gradient of log probability) of the noised data distribution.',
          'At inference, the model iteratively predicts and removes noise starting from pure Gaussian noise.'
        ]}
        id="thm-ddpm-loss"
      />

      <ExampleBlock
        title="Forward Process Step-by-Step"
        problem="Given x_0 (a clean image), compute x_t at t=500 with alpha_bar_500 = 0.05."
        steps={[
          { formula: '\\bar{\\alpha}_{500} = 0.05 \\implies \\sqrt{\\bar{\\alpha}_{500}} \\approx 0.224', explanation: 'The signal scaling factor is very small at t=500.' },
          { formula: '\\sqrt{1 - \\bar{\\alpha}_{500}} = \\sqrt{0.95} \\approx 0.975', explanation: 'The noise component dominates.' },
          { formula: '\\mathbf{x}_{500} = 0.224 \\cdot \\mathbf{x}_0 + 0.975 \\cdot \\boldsymbol{\\epsilon}', explanation: 'At t=500 of 1000, the image is mostly noise with faint signal.' },
        ]}
        id="example-forward-process"
      />

      <PythonCode
        title="ddpm_forward_reverse.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDDPM:
    """Simplified DDPM scheduler for forward and reverse processes."""
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = num_timesteps
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, x_0, t):
        """Add noise to x_0 at timestep t: q(x_t | x_0)."""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def training_loss(self, model, x_0):
        """Compute simplified DDPM loss."""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,))
        x_t, noise = self.forward_process(x_0, t)
        noise_pred = model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def reverse_step(self, model, x_t, t):
        """Single reverse diffusion step: p(x_{t-1} | x_t)."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        noise_pred = model(x_t, torch.tensor([t]))
        # Predicted mean
        mu = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )
        # Add noise (except at t=0)
        if t > 0:
            sigma = torch.sqrt(beta_t)
            mu = mu + sigma * torch.randn_like(x_t)
        return mu

# Visualize noise schedule
ddpm = SimpleDDPM()
timesteps = [0, 100, 250, 500, 750, 999]
for t in timesteps:
    signal = ddpm.alpha_bar[t].sqrt().item()
    noise = (1 - ddpm.alpha_bar[t]).sqrt().item()
    print(f"t={t:4d}: signal={signal:.3f}, noise={noise:.3f}, SNR={signal/noise:.3f}")`}
        id="code-ddpm"
      />

      <NoteBlock
        type="historical"
        title="Diffusion Model History"
        content="Sohl-Dickstein et al. (2015) first proposed diffusion for generative modeling. Ho et al. (2020) made it practical with DDPM. Song et al. (2021) unified the framework with score-based SDEs. Dhariwal & Nichol (2021) showed diffusion beats GANs on ImageNet, marking a paradigm shift in generative AI."
        id="note-diffusion-history"
      />

      <WarningBlock
        title="Slow Sampling"
        content="DDPM requires iterating through all T=1000 steps during generation, making it much slower than GANs or VAEs. DDIM (Song et al., 2020) reduces this to ~50 steps, and modern schedulers (DPM-Solver, Euler) can achieve good quality in 20-30 steps."
        id="warning-slow-sampling"
      />
    </div>
  )
}
