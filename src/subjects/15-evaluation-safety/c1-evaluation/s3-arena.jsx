import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function Arena() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Chatbot Arena and ELO Rating</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Chatbot Arena is a crowdsourced evaluation platform where users chat with two anonymous
        models side-by-side and vote for the better response. Votes are aggregated using the
        ELO rating system to produce a continuously updated leaderboard.
      </p>

      <DefinitionBlock
        title="Chatbot Arena"
        definition="An open evaluation platform (LMSYS) where users interact with pairs of anonymous LLMs in blind A/B tests. Users submit any prompt and vote which response they prefer, generating organic preference data at scale."
        id="def-arena"
      />

      <DefinitionBlock
        title="ELO Rating System"
        definition="A method for calculating relative skill levels in competitor-vs-competitor games. Each player has a rating $R$; after a match the ratings are updated based on the expected vs. actual outcome. Originally developed for chess by Arpad Elo."
        notation="R_A, R_B \in \mathbb{R}; \; K \text{ is the update factor}"
        id="def-elo"
      />

      <h2 className="text-2xl font-semibold">ELO Rating Mathematics</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The expected score of model A against model B is:
      </p>
      <BlockMath math="E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}" />
      <p className="text-gray-700 dark:text-gray-300">
        After a match, ratings are updated:
      </p>
      <BlockMath math="R_A' = R_A + K(S_A - E_A)" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="S_A \in \{0, 0.5, 1\}" /> is the actual score (loss, tie, win)
        and <InlineMath math="K" /> controls update magnitude (typically 4-32).
      </p>

      <TheoremBlock
        title="ELO Convergence Property"
        statement="In a system with fixed player strengths, ELO ratings converge to stable values as the number of games increases. The expected score function ensures that a player's rating stabilizes when their win rate matches the prediction: $E_A = S_A$ on average."
        corollaries={[
          'A 400-point ELO difference corresponds to a roughly 10:1 win ratio.',
          'A 200-point difference corresponds to roughly a 75% win rate for the stronger model.',
        ]}
        id="thm-elo-convergence"
      />

      <ExampleBlock
        title="ELO Update Calculation"
        problem="Model A (R_A=1200) beats Model B (R_B=1400). K=32. Compute new ratings."
        steps={[
          { formula: 'E_A = \\frac{1}{1 + 10^{(1400-1200)/400}} = \\frac{1}{1 + 10^{0.5}} \\approx 0.240', explanation: 'Model A is expected to win only 24% of the time against the higher-rated Model B.' },
          { formula: 'R_A\' = 1200 + 32(1 - 0.240) = 1200 + 24.3 = 1224.3', explanation: 'Model A gains ~24 points for the upset victory.' },
          { formula: 'R_B\' = 1400 + 32(0 - 0.760) = 1400 - 24.3 = 1375.7', explanation: 'Model B loses ~24 points for the unexpected loss.' },
        ]}
        id="example-elo-update"
      />

      <PythonCode
        title="elo_rating_system.py"
        code={`import numpy as np

class ELOSystem:
    """ELO rating system for LLM evaluation."""

    def __init__(self, k=32, initial_rating=1000):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}

    def expected_score(self, ra, rb):
        """Expected score of player A vs player B."""
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, model_a, model_b, outcome):
        """Update ratings. outcome: 1=A wins, 0=B wins, 0.5=tie."""
        ra = self.ratings.get(model_a, self.initial_rating)
        rb = self.ratings.get(model_b, self.initial_rating)

        ea = self.expected_score(ra, rb)
        eb = 1 - ea

        self.ratings[model_a] = ra + self.k * (outcome - ea)
        self.ratings[model_b] = rb + self.k * ((1 - outcome) - eb)

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])

# Simulate arena battles
elo = ELOSystem(k=16)
models = ["GPT-4o", "Claude-3.5", "Llama-3.1-405B", "Gemini-1.5-Pro"]

# Simulate 1000 matches with different win probabilities
np.random.seed(42)
true_strength = {"GPT-4o": 1300, "Claude-3.5": 1280,
                 "Llama-3.1-405B": 1200, "Gemini-1.5-Pro": 1250}

for _ in range(1000):
    a, b = np.random.choice(models, 2, replace=False)
    # Simulate outcome based on true strength
    prob_a = 1 / (1 + 10 ** ((true_strength[b] - true_strength[a]) / 400))
    outcome = 1.0 if np.random.random() < prob_a else 0.0
    elo.update(a, b, outcome)

print("Leaderboard after 1000 matches:")
for model, rating in elo.leaderboard():
    print(f"  {model}: {rating:.0f}")`}
        id="code-elo"
      />

      <NoteBlock
        type="note"
        title="Bradley-Terry Model"
        content="Chatbot Arena uses the Bradley-Terry model for more statistically rigorous rankings. The probability that model i beats model j is P(i > j) = exp(beta_i) / (exp(beta_i) + exp(beta_j)), where beta values are estimated via maximum likelihood. This is equivalent to ELO but allows bootstrapped confidence intervals."
        id="note-bradley-terry"
      />

      <WarningBlock
        title="Arena Biases"
        content="Arena evaluations reflect the user population's preferences and prompt distribution, which skew toward English-speaking tech-savvy users. Users may also favor verbose, confident-sounding responses over concise, accurate ones. Category-specific leaderboards (coding, math, instruction following) provide more nuanced rankings."
        id="warning-arena-bias"
      />

      <NoteBlock
        type="tip"
        title="Using Arena Data"
        content="The Chatbot Arena leaderboard at lmarena.ai provides the most up-to-date human preference rankings. Use it alongside automated benchmarks for a complete picture. The full vote dataset is also publicly available for research."
        id="note-using-arena"
      />
    </div>
  )
}
