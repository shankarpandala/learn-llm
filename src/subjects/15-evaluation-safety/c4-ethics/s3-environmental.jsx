import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Environmental() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Compute Carbon Footprint</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Training and serving large language models requires enormous computational resources,
        with significant energy consumption and carbon emissions. Understanding and minimizing
        this environmental impact is an ethical responsibility of the AI community.
      </p>

      <DefinitionBlock
        title="Carbon Footprint of AI"
        definition="The total greenhouse gas emissions associated with training and deploying AI models, measured in metric tons of CO$_2$ equivalent (tCO$_2$eq). This includes operational emissions (electricity for compute) and embodied emissions (hardware manufacturing and data center construction)."
        id="def-carbon"
      />

      <h2 className="text-2xl font-semibold">Estimating Compute Emissions</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The carbon footprint of model training can be estimated from the compute used.
        Total energy consumption is:
      </p>
      <BlockMath math="E = \text{GPU hours} \times \text{TDP}_{\text{GPU}} \times \text{PUE}" />
      <p className="text-gray-700 dark:text-gray-300">
        where TDP is the thermal design power and PUE (Power Usage Effectiveness) accounts
        for cooling and infrastructure overhead (typically 1.1-1.4 for modern data centers).
        Carbon emissions are then:
      </p>
      <BlockMath math="C = E \times I_{\text{grid}}" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="I_{\text{grid}}" /> is the carbon intensity of the
        electricity grid (kg CO<sub>2</sub>/kWh), which varies by region from 0.02 (Norway)
        to 0.9 (coal-heavy grids).
      </p>

      <ExampleBlock
        title="Estimating GPT-3 Training Emissions"
        problem="Estimate the carbon footprint of training GPT-3 (175B parameters)."
        steps={[
          { formula: '\\text{Compute: } \\approx 3640 \\text{ petaflop-days} \\approx 10{,}000 \\text{ V100 GPU-hours}', explanation: 'Based on reported training compute of ~3.14e23 FLOPs.' },
          { formula: 'E = 10{,}000 \\times 300W \\times 1.2 \\times 24h = 86{,}400 \\text{ kWh}', explanation: 'V100 TDP ~300W, PUE ~1.2 for hyperscale data centers.' },
          { formula: 'C = 86{,}400 \\times 0.429 \\approx 552 \\text{ tCO}_2\\text{eq}', explanation: 'Using US average grid intensity. Actual: ~552 tCO2eq (Patterson et al., 2021).' },
        ]}
        id="example-gpt3-carbon"
      />

      <PythonCode
        title="carbon_footprint_calculator.py"
        code={`import numpy as np

class CarbonCalculator:
    """Estimate carbon footprint of LLM training and inference."""

    # GPU power consumption (TDP in watts)
    GPU_TDP = {
        "V100": 300, "A100_40": 400, "A100_80": 400,
        "H100": 700, "H200": 700, "B200": 1000,
    }

    # Grid carbon intensity (kg CO2 / kWh) by region
    GRID_INTENSITY = {
        "us_average": 0.429, "us_west": 0.258, "us_east": 0.386,
        "europe_average": 0.276, "nordics": 0.025, "france": 0.056,
        "uk": 0.233, "germany": 0.385, "china": 0.555, "india": 0.708,
    }

    def __init__(self, gpu_type: str = "H100", pue: float = 1.2,
                 region: str = "us_average"):
        self.gpu_tdp = self.GPU_TDP[gpu_type]
        self.pue = pue
        self.grid_intensity = self.GRID_INTENSITY[region]

    def training_emissions(self, num_gpus: int, hours: float) -> dict:
        """Estimate training carbon footprint."""
        energy_kwh = (num_gpus * self.gpu_tdp * self.pue * hours) / 1000
        co2_kg = energy_kwh * self.grid_intensity
        return {
            "energy_kwh": energy_kwh,
            "co2_kg": co2_kg,
            "co2_tonnes": co2_kg / 1000,
            "equivalent_flights_ny_sf": co2_kg / 900,  # ~900 kg per round trip
            "equivalent_car_miles": co2_kg / 0.404,  # avg US car
        }

    def inference_emissions(self, requests_per_day: int, tokens_per_request: int,
                           gpus_serving: int, days: int = 365) -> dict:
        """Estimate annual inference carbon footprint."""
        hours = days * 24  # GPUs run continuously
        energy_kwh = (gpus_serving * self.gpu_tdp * self.pue * hours) / 1000
        co2_kg = energy_kwh * self.grid_intensity
        total_requests = requests_per_day * days
        return {
            "annual_energy_kwh": energy_kwh,
            "annual_co2_kg": co2_kg,
            "annual_co2_tonnes": co2_kg / 1000,
            "co2_per_1k_requests": (co2_kg / total_requests) * 1000,
        }

    def compare_regions(self, num_gpus: int, hours: float) -> dict:
        """Compare emissions across data center locations."""
        results = {}
        for region, intensity in self.GRID_INTENSITY.items():
            energy = (num_gpus * self.gpu_tdp * self.pue * hours) / 1000
            results[region] = energy * intensity / 1000  # tonnes
        return dict(sorted(results.items(), key=lambda x: x[1]))

# Calculate emissions for a typical training run
calc = CarbonCalculator(gpu_type="H100", pue=1.1, region="us_average")

# Training: 1000 H100s for 30 days
training = calc.training_emissions(num_gpus=1000, hours=30*24)
print("=== Training Emissions (1000 H100s, 30 days) ===")
print(f"Energy: {training['energy_kwh']:,.0f} kWh")
print(f"CO2: {training['co2_tonnes']:.1f} tonnes")
print(f"Equivalent to {training['equivalent_flights_ny_sf']:.0f} NY-SF flights")

# Inference: serving 1M requests/day
inference = calc.inference_emissions(
    requests_per_day=1_000_000, tokens_per_request=500,
    gpus_serving=100, days=365,
)
print(f"\\n=== Annual Inference (100 GPUs serving) ===")
print(f"Annual CO2: {inference['annual_co2_tonnes']:.1f} tonnes")
print(f"CO2 per 1K requests: {inference['co2_per_1k_requests']:.3f} kg")

# Compare regions
print("\\n=== Regional Comparison (same workload) ===")
for region, tonnes in calc.compare_regions(1000, 720).items():
    bar = "#" * int(tonnes / 10)
    print(f"  {region:18s}: {tonnes:7.1f} t  {bar}")`}
        id="code-carbon"
      />

      <NoteBlock
        type="note"
        title="Inference Dominates Long-Term"
        content="While training gets the most attention, inference emissions often dwarf training over a model's lifetime. A model trained once but serving millions of daily requests for years generates far more cumulative emissions from inference. Optimizations like quantization, distillation, and efficient serving reduce both cost and carbon."
        id="note-inference-dominates"
      />

      <WarningBlock
        title="Hidden Environmental Costs"
        content="Carbon estimates often exclude: hardware manufacturing (embedded carbon in GPUs and servers), water consumption for cooling (data centers use millions of gallons annually), electronic waste from GPU refresh cycles, and the carbon cost of data storage and network infrastructure."
        id="warning-hidden-costs"
      />

      <NoteBlock
        type="tip"
        title="Reducing AI's Carbon Footprint"
        content="Practical strategies: train in regions with clean grids (Nordic countries, Quebec, Oregon), use efficient architectures (MoE, distillation), schedule training during low-carbon hours, use spot/preemptible instances to utilize idle capacity, quantize models for inference, and report emissions in model cards using tools like ML CO2 Impact or CodeCarbon."
        id="note-reducing"
      />
    </div>
  )
}
