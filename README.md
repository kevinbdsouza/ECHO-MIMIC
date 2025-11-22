# ECHO-MIMIC

## Overview

ECHO-MIMIC is a general computational framework for solving collective action problems—situations where individual incentives conflict with group welfare. It transforms these "Ill-Structured Problems" (ISPs) into "Well-Structured Problems" (WSPs) by discovering compact, executable heuristics and persuasive natural language rationales.

The framework is demonstrated on two distinct domains:
1.  **Agricultural Landscape Management**: Balancing farm-level profit with global ecological connectivity.
2.  **Carbon-Aware EV Charging**: Balancing individual charging preferences with grid-level load flattening and carbon reduction.

![ECHO-MIMIC Framework Overview](data/farm/intro_em.png)

## Key Features

-   **General-Purpose Framework**: Applicable to any domain with local-global incentive misalignment (e.g., decentralized resource management, policy design).
-   **Two-Phase Optimization**:
    -   **ECHO** (Evolutionary Crafting of Heuristics from Outcomes): Evolves executable Python policies (heuristics) that optimize global objectives.
    -   **MIMIC** (Mechanism Inference & Messaging for Individual-to-Collective Alignment): Evolves natural language messages (nudges) to persuade agents to adopt these policies.
-   **Four-Stage Pipeline**:
    1.  **Baseline**: Establish default, self-interested behavior.
    2.  **Imitation**: Learn compact heuristics that reproduce the baseline.
    3.  **Coordination**: Learn "global" heuristics that optimize the collective good.
    4.  **Nudging**: Generate messages to shift agents from baseline to coordination heuristics.
-   **LLM+EA Paradigm**: Combines the creativity of Large Language Models with the rigorous selection of Evolutionary Algorithms.
-   **Behavioral Modeling**: Supports heterogeneous agent personalities (e.g., Economic, Social, Resistant) to test policy robustness.

## Project Structure

```
.
├── echo_mimic/
│   ├── __init__.py              # Package entry point
│   ├── baselines/               # AutoGen + DSPy comparison methods
│   │   ├── autogen.py           # AutoGen-style planner/critic loop
│   │   └── dspy/                # DSPy baselines (farm/global/nudge)
│   ├── common/                  # LLM helpers, code execution, and fix utilities
│   ├── config.py                # Configuration and parameters
│   ├── domains/
│   │   ├── farm/
│   │   │   ├── __init__.py
│   │   │   ├── create_prompts.py        # Backwards-compatible prompt exports
│   │   │   ├── farm_evo_strat.py        # Stage 2 entry point
│   │   │   ├── graph_evo_strat.py       # Stage 3 entry point
│   │   │   ├── nudge_evo_strat.py       # Stage 4 entry point
│   │   │   └── run_experimental_suite.py # Personality-nudge runner
│   │   └── energy_ev/
│   │       ├── __init__.py
│   │       ├── energy_policy_evolution.py
│   │       ├── energy_local_evo_strat.py
│   │       ├── energy_global_evo_strat.py
│   │       └── energy_nudge_evo_strat.py
│   ├── prompts/                 # Prompt generation utilities
│   ├── tools.py                 # Plotting, metrics, and helper routines
│   ├── utils.py                 # Geometry helpers and plotting utilities
│   ├── rate_limiter.py          # API rate limiting utilities
│   └── dspy_rate_limiter.py     # DSPy LM wrapper with rate limiting
├── main.py                      # Unified CLI entry point
└── requirements.txt             # Python dependencies
```

All shared runtime logic and experiment drivers now live under the structured `echo_mimic`
package, keeping the repository root focused on orchestration entry points such as `main.py`.

Farm-specific datasets (plots, ground-truth labels, and prompt seeds) are located under `data/farm/`,
while the energy EV datasets remain in `data/energy_ev/`.

## Core Components

### 1. ECHO-MIMIC Framework Implementation

#### ECHO (Stages 2-3): Evolutionary Crafting of Heuristics from Outcomes
-   **`echo_mimic/domains/farm/farm_evo_strat.py`**: Farm Domain - Stage 2 (Imitation)
-   **`echo_mimic/domains/farm/graph_evo_strat.py`**: Farm Domain - Stage 3 (Coordination)
-   **`echo_mimic/domains/energy_ev/energy_local_evo_strat.py`**: EV Domain - Stage 2 (Imitation)
-   **`echo_mimic/domains/energy_ev/energy_global_evo_strat.py`**: EV Domain - Stage 3 (Coordination)

#### MIMIC (Stage 4): Mechanism Inference & Messaging
-   **`echo_mimic/domains/farm/nudge_evo_strat.py`**: Farm Domain - Stage 4 (Nudging)
-   **`echo_mimic/domains/energy_ev/energy_nudge_evo_strat.py`**: EV Domain - Stage 4 (Nudging)

### 2. DSPy Baselines (Comparison Methods)
-   **`echo_mimic/baselines/dspy/farm.py`**: Farm Domain baselines
-   **`echo_mimic/baselines/dspy/global_baseline.py`**: General global optimization baseline
-   **`echo_mimic/baselines/dspy/nudge.py`**: General nudging baseline

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- Google Gemini API key (for LLM functionality)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ECHO-MIMIC
   ```

2. **Create and activate a Python virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   Create a `.env` file in the project root and add your API keys:
   ```bash
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   
## Usage

### Unified Orchestrator

Run any combination of domain, mode, and method from a single entry point:

```bash
# Echo-MIMIC, farm domain, local heuristic optimization
python main.py --domain farm --mode local --method echo_mimic --agent-id alpha

# DSPy baseline for global farm heuristics
python main.py --domain farm --mode global --method dspy

# Energy domain with the lightweight AutoGen-style planner
python main.py --domain energy --mode nudge --method autogen --model gpt-4o-mini
```

The orchestrator also accepts `--population-size`, `--num-generations`,
`--inner-loop-size`, `--use-template`, `--no-hint`, `--halstead-metrics`, and
`--no-init` to fine-tune runs per scenario.

### Carbon-Aware EV Charging Scenario

The **EV Charging Domain** models a residential neighborhood where electric vehicle owners must schedule charging.

-   **Local Incentive**: Charge immediately upon arrival (convenience) or when electricity is cheapest.
-   **Global Objective**: Flatten the aggregate load curve and prioritize charging during low-carbon intensity windows.
-   **Conflict**: Uncoordinated charging leads to demand peaks (grid stress) and higher carbon emissions.

Run the EV domain pipeline:

```bash
# Energy domain with the lightweight AutoGen-style planner
python main.py --domain energy --mode nudge --method autogen --model gpt-4o-mini
```

Or use the dedicated pipeline script for granular control:

```bash
# Score a local imitation heuristic
python energy_pipeline.py evaluate-local data/energy_ev/scenario_1/local/heuristics_baseline.py

# Score a global coordination heuristic
python energy_pipeline.py evaluate-global data/energy_ev/scenario_1/global/heuristics_baseline.py

# Validate a nudging message
python energy_pipeline.py evaluate-nudge data/energy_ev/scenario_1/nudge/sample_nudge.json
```

## Key Concepts

### ECHO-MIMIC Pipeline: From ISPs to WSPs
The framework transforms Ill-Structured Problems (ISPs) into Well-Structured Problems (WSPs) through a systematic decomposition:

**The Collective Action Challenge**: Individual agents face unclear causal links between local actions and global outcomes, conflicting objectives, and no clear algorithm to bridge micro-level choices with macro-level welfare.

**ECHO-MIMIC Solution**: 
1. **Stage 1**: Establish baseline behavior - compute profit-maximizing actions for each agent
2. **Stage 2** (ECHO): Learn executable Python heuristics that reproduce baseline behavior  
3. **Stage 3** (ECHO): Learn heuristics that maximize global objectives (landscape connectivity)
4. **Stage 4** (MIMIC): Evolve persuasive natural language messages that motivate adoption of global heuristics

**Result**: Complex collective action becomes a simple set of agent-level instructions, making previously ill-structured problems solvable in practice.

### Supported Domains

#### 1. Agricultural Landscape Management
-   **Agents**: Farmers managing land plots.
-   **Actions**: Margin interventions (pollinator strips), habitat conversion.
-   **Local Goal**: Maximize Net Present Value (NPV) of crops.
-   **Global Goal**: Maximize Integral Index of Connectivity (IIC) for the ecosystem.
-   **Mechanism**: Nudges persuade farmers to sacrifice some yield for connectivity.

#### 2. Carbon-Aware EV Charging
-   **Agents**: EV owners in a neighborhood.
-   **Actions**: Choosing charging slots (hours of the day).
-   **Local Goal**: Maximize convenience (charge ASAP) and minimize personal cost.
-   **Global Goal**: Minimize peak load (grid stress) and carbon footprint.
-   **Mechanism**: Nudges persuade owners to shift charging to off-peak/low-carbon hours.

### Agent Heterogeneity
Both domains support diverse agent personalities to test the robustness of the evolved mechanisms:
-   **Resistant**: Skeptical, requires strong evidence/incentives.
-   **Economic**: Driven purely by financial/utility maximization.
-   **Socially Influenced**: Follows the crowd or community norms.

## Configuration

The `config.py` file contains all major configuration parameters:

- **Model Settings**: LLM model selection and API configuration
- **Rate Limiting**: API request throttling parameters
- **Economic Parameters**: Crop prices, costs, discount rates
- **Spatial Parameters**: Habitat types, intervention costs

## Data Format

The system expects farm data in GeoJSON format with:
- Agricultural plots (`type='ag_plot'`)
- Existing habitat plots (`type='hab_plots'`)
- Spatial geometry and connectivity information
- Economic parameters and crop types

## Output

The system generates:
- **Optimized Heuristics**: Python code for agricultural intervention strategies
- **Performance Metrics**: Fitness scores, connectivity indices, NPV calculations
