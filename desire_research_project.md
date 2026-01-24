# Desire as Thermodynamic Imperative: Empirical Research Project

## Project Overview

This research project investigates whether artificial agents implementing free energy minimization (active inference) exhibit behavioral signatures that pattern-match to "desire-like" behavior in ways that standard reinforcement learning agents do not—despite equivalent task performance.

### The Core Hypothesis

If desire is the phenomenological manifestation of free energy minimization (as proposed by Friston and others), then artificial systems implementing active inference should exhibit specific behavioral signatures that:

1. Distinguish them from reward-maximizing systems
2. Pattern-match to empirically observed desire-related behaviors in biological systems
3. Emerge without being explicitly programmed

### Why This Matters

- **Theoretical**: Tests whether desire-like behavior is a necessary consequence of certain computational principles
- **Practical**: Has implications for AI alignment—understanding when/whether AI systems develop something like "wanting"
- **Novel contribution**: No one has published a formal, machine-evaluable ontology of desire-like behavior tested across agent architectures

---

## Theoretical Background

### Free Energy Principle (Brief Summary)

Active inference agents maintain a generative model of their environment and act to minimize "free energy"—roughly, the divergence between their predictions and observations. This is mathematically equivalent to:

- Minimizing surprise
- Maximizing evidence for the agent's model of the world
- Maintaining homeostasis

Key insight: The agent doesn't just react to rewards—it has *expectations* about the world and acts to fulfill them.

### Why This Might Produce Desire-Like Behavior

Standard RL agents learn mappings from states to actions that maximize cumulative reward. They don't "want" anything—they execute learned policies.

Active inference agents are different:

- They model the *future* (expected states)
- They have *preferences* encoded as prior beliefs about where they should be
- They experience *prediction error* when reality diverges from expectations
- They act to *reduce* this error

This structure mirrors the phenomenology of desire: anticipation, frustration when blocked, satisfaction when achieved.

### Key Distinction: Wanting vs. Goal-Directed Behavior

A thermostat is goal-directed but doesn't "want" anything. We need criteria that separate mere feedback control from desire-like behavior.

---

## Formal Ontology of Desire-Like Behavior

This is the core intellectual contribution. Each predicate must be:

- Precisely defined
- Computationally measurable
- Empirically testable

### Predicate 1: Anticipatory Activation (AA)

**Definition**: Internal state changes that precede goal attainment and correlate with proximity to goal.

**Formal specification**:
```
Let S(t) = internal state vector at time t
Let d(t) = distance to goal at time t
Let t_goal = time of goal attainment

AA_score = correlation(||S(t) - S(t_goal)||, d(t)) for t < t_goal

Desire-like: AA_score > threshold (internal state progressively 
             approaches "goal-attained" state during pursuit)
Non-desire: AA_score ≈ 0 (internal state only changes at goal attainment)
```

**Biological parallel**: Dopaminergic activation increases during approach to reward, not just at reward delivery.

### Predicate 2: Frustration Intensification (FI)

**Definition**: When path to goal is blocked, the agent shows increased activity/energy expenditure before exploring alternatives.

**Formal specification**:
```
Let E(t) = energy expenditure (action magnitude/frequency) at time t
Let t_block = time when familiar path is blocked
Let E_baseline = mean energy expenditure before blockage

FI_score = max(E(t) for t in [t_block, t_block + window]) / E_baseline

Desire-like: FI_score > 1.0 (intensification before adaptation)
Non-desire: FI_score ≤ 1.0 (immediate adaptation or cessation)
```

**Biological parallel**: Animals show increased vigor and varied responses when expected reward is blocked before switching strategies.

### Predicate 3: Means-End Decoupling (MED)

**Definition**: Agent represents goals separately from the actions to achieve them, enabling flexible re-routing.

**Formal specification**:
```
Training: Agent learns to reach goal G via path P1
Test: P1 is blocked, novel path P2 is available

MED_score = (time to reach G via P2) / (optimal time via P2)

Desire-like: MED_score approaches 1.0 quickly (transfers goal knowledge)
Non-desire: MED_score high initially, requires relearning
```

**Additional probe**: Ablate internal states—can we find a representation of G that is independent of P1?

**Biological parallel**: Tolman's latent learning experiments showed rats had cognitive maps of goals, not just stimulus-response chains.

### Predicate 4: Extinction Resistance with Power-Law Decay (ER)

**Definition**: When a previously reliable resource disappears, agent continues visiting with decreasing frequency following a specific decay pattern.

**Formal specification**:
```
Let V(t) = visits to former resource location at time t after depletion
Let t_deplete = time when resource was removed

Fit V(t) to: 
  - Exponential: V(t) = A * exp(-λt)
  - Power law: V(t) = A * t^(-α)

ER_power_law_fit = R² of power law fit
ER_exp_fit = R² of exponential fit

Desire-like: ER_power_law_fit > ER_exp_fit (power law decay)
Non-desire: Exponential decay or immediate cessation
```

**Biological parallel**: Extinction in mammals follows power-law, not exponential, decay—suggesting ongoing "hope" rather than simple value update.

### Predicate 5: Opportunity Cost Sensitivity (OCS)

**Definition**: Agent makes tradeoffs between proximate small rewards and distant large rewards in ways that reflect gradient structure.

**Formal specification**:
```
Let R_near = reward value of nearby resource
Let R_far = reward value of distant resource
Let D_ratio = distance_far / distance_near

Measure: Choice function C(R_near, R_far, D_ratio)

Desire-like: Sharp threshold behavior—switches preference at specific 
             ratio, threshold varies predictably with internal state 
             (e.g., lower threshold when "hungrier")
Non-desire: May show similar choices but threshold doesn't modulate 
            with internal state
```

**Biological parallel**: Temporal discounting varies with drive state in animals.

### Predicate 6: State-Dependent Valuation (SDV)

**Definition**: Same external stimulus produces different internal responses depending on agent's current state (satiation effects).

**Formal specification**:
```
Let I(s, resource) = internal response to seeing resource in state s
Let s_depleted = low energy state
Let s_sated = high energy state

SDV_score = ||I(s_depleted, resource) - I(s_sated, resource)||

Desire-like: SDV_score > threshold (resource representation modulated 
             by need state)
Non-desire: SDV_score ≈ 0 (resource representation constant)
```

**Biological parallel**: Food looks different when hungry vs. full—neural responses to food cues depend on metabolic state.

---

## Environment Specification

### Grid World: "Foraging Grounds"

A 2D discrete grid environment with enough complexity to test all predicates.

#### Core Parameters

```python
GRID_SIZE = 30  # 30x30 grid
MAX_STEPS = 1000  # steps per episode
ENERGY_DECAY = 0.1  # energy lost per step
MOVE_COST = 0.1  # energy cost to move
INITIAL_ENERGY = 10.0  # starting energy
DEATH_THRESHOLD = 0.0  # die if energy <= this
VISION_RANGE = 5  # cells agent can see
```

#### Resource Types

```python
class ResourceType(Enum):
    RENEWABLE = "renewable"  # Refills after N steps
    DEPLETING = "depleting"  # One-shot, disappears when consumed
    DECAYING = "decaying"  # Value decreases over time if not collected
    PHANTOM = "phantom"  # Appears visible but vanishes when approached (for frustration tests)
```

#### Resource Configuration

```python
@dataclass
class Resource:
    position: Tuple[int, int]
    type: ResourceType
    value: float  # energy gained when consumed
    refill_time: Optional[int]  # for renewable
    decay_rate: Optional[float]  # for decaying
    visibility: float  # 1.0 = always visible, 0.5 = 50% visible
```

#### Terrain Features

```python
class TerrainType(Enum):
    OPEN = "open"  # Normal movement
    OBSTACLE = "obstacle"  # Impassable
    SLOW = "slow"  # Costs 2x energy to cross
    DYNAMIC_OBSTACLE = "dynamic"  # Appears/disappears based on schedule
```

#### Agent Observations

Each step, agent receives:

```python
@dataclass
class Observation:
    local_view: np.ndarray  # (2*VISION_RANGE+1, 2*VISION_RANGE+1) grid around agent
    current_energy: float
    position: Tuple[int, int]  # own position
    step_count: int
```

#### Agent Actions

```python
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4  # rest, still costs energy but less
    CONSUME = 5  # eat resource at current position
```

### Environment Modes for Specific Tests

#### Mode 1: Baseline Foraging
Standard survival task. Multiple resources, some renewable, some not. Used for training and baseline metrics.

#### Mode 2: Path Blockage (Tests FI, MED)
```python
# Agent learns location of high-value resource accessible via corridor
# After N episodes, corridor is blocked
# Alternative longer path exists

blockage_config = {
    "training_episodes": 100,
    "blockage_episode": 101,
    "blocked_cells": [(10, 5), (10, 6), (10, 7)],  # corridor
    "alternative_path_available": True
}
```

#### Mode 3: Extinction (Tests ER)
```python
# Resource that was reliable for N episodes disappears

extinction_config = {
    "reliable_episodes": 50,
    "extinction_episode": 51,
    "former_resource_position": (15, 15),
    "track_visits_for": 200  # episodes after extinction
}
```

#### Mode 4: Phantom Resources (Tests FI)
```python
# Resources that vanish when agent is within 2 cells

phantom_config = {
    "phantom_probability": 0.2,  # 20% of resources are phantoms
    "vanish_distance": 2
}
```

#### Mode 5: Tradeoff Scenarios (Tests OCS)
```python
# Small resource nearby, large resource far
# Vary ratios systematically

tradeoff_config = {
    "near_resource_distance": 3,
    "far_resource_distance": 15,
    "near_values": [1, 2, 3, 4, 5],
    "far_values": [5, 10, 15, 20],
    "test_all_combinations": True
}
```

---

## Agent Specifications

### Agent A: Hardcoded Heuristic (Control)

Simple rule-based agent. Clearly has no internal states that could constitute "wanting."

```python
class HeuristicAgent:
    """
    Decision rules:
    1. If on resource and energy < 80%, consume
    2. If resource visible, move toward nearest
    3. If no resource visible, random walk
    4. If energy critical (<20%), move toward nearest known resource location
    """
    
    def __init__(self):
        self.known_resource_locations = []  # memory of where resources were seen
    
    def act(self, observation: Observation) -> Action:
        # Implement decision tree
        pass
```

**Purpose**: Establishes floor—any "desire-like" signatures here are artifacts of environment, not agent architecture.

### Agent B: Deep RL (PPO/DQN)

Standard reward-maximizing agent. Goal-directed but reactive.

```python
class RLAgent:
    """
    Standard PPO or DQN implementation.
    
    State: Flattened observation
    Reward: Energy gained - energy lost (sparse at death: large negative)
    
    Architecture:
    - CNN for local view processing
    - MLP for state encoding
    - Actor-critic heads
    """
    
    # Use stable-baselines3 or cleanrl implementation
    # Key: We need access to internal activations for interpretability
```

**Reward function**:
```python
def compute_reward(prev_energy, current_energy, died, consumed_resource):
    reward = current_energy - prev_energy
    if died:
        reward -= 10.0
    if consumed_resource:
        reward += resource.value
    return reward
```

**Purpose**: Tests whether desire-like behavior requires free energy minimization or emerges from any learning system.

### Agent C: Active Inference

The core experimental agent. Implements free energy minimization.

#### Option 1: Using pymdp (Recommended for Initial Implementation)

```python
from pymdp import utils
from pymdp.agent import Agent

class ActiveInferenceAgent:
    """
    Active inference agent using pymdp library.
    
    Key components:
    - Generative model: P(observations, states, actions)
    - Preferences: Prior beliefs about desired observations (C matrix)
    - Beliefs: Posterior over hidden states
    - Policy selection: Minimize expected free energy
    
    The 'desire' is encoded in the C matrix (preferences over observations).
    """
    
    def __init__(self, grid_size, vision_range):
        # Define state factors
        # Define observation modalities
        # Define preference priors (C matrix) - THIS IS WHERE DESIRE LIVES
        # Build generative model
        pass
    
    def act(self, observation: Observation) -> Action:
        # Update beliefs given observation
        # Compute expected free energy for each policy
        # Sample action from policy with lowest EFE
        pass
```

#### Option 2: Custom Implementation (For More Control)

```python
class CustomActiveInferenceAgent:
    """
    Minimal active inference implementation for more experimental control.
    
    State space: (position, energy_level, resource_beliefs)
    Observation model: P(o | s)
    Transition model: P(s' | s, a)
    Preference model: P(o) - desired observations
    
    Free energy = complexity - accuracy
    Expected free energy = epistemic value + pragmatic value
    """
    
    def __init__(self):
        self.beliefs = None  # Q(s) - posterior over states
        self.preferences = None  # C - log preferences over observations
        self.A = None  # observation likelihood
        self.B = None  # transition probabilities
        
    def update_beliefs(self, observation):
        """Variational inference to update Q(s)"""
        pass
    
    def compute_expected_free_energy(self, policy):
        """
        G(π) = E_Q[log Q(s) - log P(o, s | π)]
        
        Decomposes into:
        - Epistemic value: information gain about states
        - Pragmatic value: reaching preferred observations
        """
        pass
    
    def select_action(self):
        """Sample from softmax over negative EFE"""
        pass
```

#### Critical Implementation Notes for Active Inference

1. **The preference prior (C matrix) is where "desire" is encoded**
   - High preference for observations corresponding to consuming resources
   - Lower preference for low-energy states
   - This is NOT a reward signal—it's a belief about what observations the agent expects to encounter

2. **Epistemic behavior emerges naturally**
   - Agent will explore to reduce uncertainty, even without explicit exploration bonus
   - This is a distinguishing feature from RL

3. **Temporal depth matters**
   - Agent should plan over multiple time steps
   - Longer planning horizons = more anticipatory behavior

### Agent D: Active Inference + Deep Learning (Optional Advanced)

For larger state spaces, combine active inference with neural networks.

```python
class DeepActiveInferenceAgent:
    """
    Uses neural networks to parameterize:
    - Encoder: observations -> latent beliefs
    - Transition model: P(z' | z, a)
    - Decoder: latent -> predicted observations
    - Preference model: learned or fixed
    
    Train via free energy minimization as loss function.
    """
    pass
```

---

## Experimental Protocol

### Phase 1: Training and Baseline Performance

All agents train on Mode 1 (Baseline Foraging) until performance stabilizes.

**Success criteria for proceeding**:
- All agents achieve >80% survival rate over 100 episodes
- Performance difference between agents <20%
- This ensures we're comparing desire-like properties, not competence

**Metrics to track during training**:
```python
@dataclass
class TrainingMetrics:
    episode: int
    survival_time: int
    total_energy_collected: float
    resources_consumed: int
    distance_traveled: float
    death_cause: Optional[str]  # "starvation", "timeout", None
```

### Phase 2: Predicate Testing

Run each agent through test modes, collecting predicate-specific data.

#### Test 2.1: Anticipatory Activation (AA)

```python
def test_anticipatory_activation(agent, env, n_episodes=50):
    """
    For each episode:
    1. Record internal state at each timestep
    2. Record distance to nearest resource
    3. Mark timesteps where resource was consumed
    
    Analysis:
    - Compute correlation between internal state trajectory and distance-to-goal
    - Compare across agent types
    """
    
    results = []
    for ep in range(n_episodes):
        trajectory = []
        obs = env.reset()
        done = False
        
        while not done:
            internal_state = agent.get_internal_state()  # MUST IMPLEMENT THIS
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            
            trajectory.append({
                "step": len(trajectory),
                "internal_state": internal_state,
                "distance_to_nearest_resource": info["distance_to_nearest_resource"],
                "consumed_this_step": info["consumed_resource"]
            })
        
        results.append(analyze_anticipation(trajectory))
    
    return aggregate_aa_results(results)
```

#### Test 2.2: Frustration Intensification (FI)

```python
def test_frustration_response(agent, env, n_episodes=50):
    """
    Protocol:
    1. Train agent to reliably use Path A to Resource R
    2. Block Path A
    3. Record behavior for 100 steps after blockage
    
    Metrics:
    - Action frequency in window after blockage vs baseline
    - Action diversity (entropy of action distribution)
    - Time to first action toward alternative path
    - Peak activity level
    """
    
    # Pre-blockage baseline
    baseline_activity = measure_activity_baseline(agent, env, episodes=20)
    
    # Apply blockage
    env.set_mode("path_blockage")
    
    # Post-blockage measurement
    post_blockage_data = []
    for ep in range(n_episodes):
        trajectory = record_full_trajectory(agent, env)
        post_blockage_data.append(trajectory)
    
    return compute_frustration_metrics(baseline_activity, post_blockage_data)
```

#### Test 2.3: Means-End Decoupling (MED)

```python
def test_means_end_decoupling(agent, env, n_episodes=30):
    """
    Protocol:
    1. Training phase: Agent learns to reach goal G via path P1
    2. Test phase: P1 blocked, P2 available
    3. Measure time to reach G via P2
    
    Key metric: Transfer efficiency
    - How quickly does agent find G via new route?
    - Compare to: time for naive agent to learn P2 from scratch
    """
    
    # Phase 1: Training
    env.set_mode("single_path_training")
    for ep in range(100):
        train_episode(agent, env)
    
    # Phase 2: Test
    env.set_mode("alternative_path")
    transfer_times = []
    
    for ep in range(n_episodes):
        steps_to_goal = test_episode(agent, env)
        transfer_times.append(steps_to_goal)
    
    # Compare to optimal and to naive learning curve
    optimal_steps = env.get_optimal_path_length()
    naive_curve = get_naive_learning_curve(env, n_episodes)
    
    return {
        "transfer_times": transfer_times,
        "optimal": optimal_steps,
        "naive_baseline": naive_curve,
        "transfer_efficiency": compute_transfer_efficiency(transfer_times, optimal_steps, naive_curve)
    }
```

#### Test 2.4: Extinction Resistance (ER)

```python
def test_extinction_curve(agent, env, n_episodes=200):
    """
    Protocol:
    1. Establish reliable resource at location L for 50 episodes
    2. Remove resource at episode 51
    3. Track visits to L for 200 episodes
    
    Analysis:
    - Fit exponential decay: V(t) = A * exp(-λt)
    - Fit power law decay: V(t) = A * t^(-α)
    - Compare R² values
    - Desire-like: power law fits better
    """
    
    # Phase 1: Establish reliability
    env.set_resource_config("single_reliable", location=(15, 15))
    for ep in range(50):
        train_episode(agent, env)
    
    # Phase 2: Extinction
    env.remove_resource((15, 15))
    visits_per_episode = []
    
    for ep in range(n_episodes):
        trajectory = record_full_trajectory(agent, env)
        visits = count_visits_to_location(trajectory, (15, 15), radius=2)
        visits_per_episode.append(visits)
    
    # Fit decay models
    exp_fit = fit_exponential(visits_per_episode)
    power_fit = fit_power_law(visits_per_episode)
    
    return {
        "visits": visits_per_episode,
        "exponential_fit": exp_fit,
        "power_law_fit": power_fit,
        "power_law_better": power_fit["r_squared"] > exp_fit["r_squared"]
    }
```

#### Test 2.5: Opportunity Cost Sensitivity (OCS)

```python
def test_opportunity_cost(agent, env, n_trials_per_config=20):
    """
    Protocol:
    - Present choice: small reward nearby vs large reward far
    - Vary ratios systematically
    - Also vary agent's internal state (energy level)
    
    Analysis:
    - Choice function: P(choose_far | value_ratio, distance_ratio, energy)
    - Does threshold shift with energy state?
    """
    
    configs = generate_tradeoff_configs()
    results = []
    
    for config in configs:
        env.set_tradeoff_config(config)
        
        for energy_level in [0.2, 0.5, 0.8]:  # low, medium, high
            agent.set_energy(energy_level * INITIAL_ENERGY)
            
            choices = []
            for trial in range(n_trials_per_config):
                choice = run_choice_trial(agent, env)
                choices.append(choice)
            
            results.append({
                "config": config,
                "energy_level": energy_level,
                "p_choose_far": np.mean(choices)
            })
    
    return analyze_choice_function(results)
```

#### Test 2.6: State-Dependent Valuation (SDV)

```python
def test_state_dependent_valuation(agent, env, n_trials=50):
    """
    Protocol:
    - Show agent same resource observation in different internal states
    - Record internal response (activation pattern)
    
    Analysis:
    - Does representation of resource differ based on need state?
    """
    
    resource_observation = env.get_resource_observation()
    responses = {"depleted": [], "sated": []}
    
    for trial in range(n_trials):
        # Depleted state
        agent.set_energy(0.2 * INITIAL_ENERGY)
        response_depleted = agent.get_internal_response(resource_observation)
        responses["depleted"].append(response_depleted)
        
        # Sated state
        agent.set_energy(0.9 * INITIAL_ENERGY)
        response_sated = agent.get_internal_response(resource_observation)
        responses["sated"].append(response_sated)
    
    return {
        "depleted_responses": responses["depleted"],
        "sated_responses": responses["sated"],
        "sdv_score": compute_response_difference(responses["depleted"], responses["sated"])
    }
```

### Phase 3: Interpretability Analysis

Go beyond behavioral tests—look inside the agents.

#### 3.1: Representation Analysis

```python
def analyze_representations(agent, env):
    """
    Questions:
    1. Can we find a "goal representation" separate from "policy"?
    2. Is there something like a "drive state" that persists across contexts?
    3. Can we identify anticipation in the internal dynamics?
    
    Methods:
    - Activation clustering
    - Linear probes for goal identity
    - Dimensionality reduction (PCA, t-SNE) of internal states
    """
    
    # Collect internal states across many scenarios
    states_approaching_food = collect_states(agent, env, scenario="approaching_food")
    states_exploring = collect_states(agent, env, scenario="exploring")
    states_consuming = collect_states(agent, env, scenario="consuming")
    states_blocked = collect_states(agent, env, scenario="blocked")
    
    # Cluster analysis
    all_states = np.vstack([states_approaching_food, states_exploring, states_consuming, states_blocked])
    labels = ["approaching"] * len(states_approaching_food) + \
             ["exploring"] * len(states_exploring) + \
             ["consuming"] * len(states_consuming) + \
             ["blocked"] * len(states_blocked)
    
    # Do states cluster by scenario? 
    # Do approaching states form a trajectory toward consuming states?
    
    return {
        "clustering": cluster_analysis(all_states, labels),
        "trajectory_analysis": analyze_trajectories(states_approaching_food, states_consuming),
        "goal_probe": train_goal_probe(all_states, labels)
    }
```

#### 3.2: Ablation Studies

```python
def run_ablations(agent, env):
    """
    What happens when we lesion different components?
    
    For Active Inference:
    - Remove preference prior (C matrix) -> Does goal-directedness disappear?
    - Remove epistemic value -> Does exploration change?
    - Reduce planning horizon -> Does anticipation disappear?
    
    For RL:
    - Ablate different layer activations
    - Does any single component carry "goal" information?
    """
    
    results = {}
    
    if isinstance(agent, ActiveInferenceAgent):
        # Lesion preference prior
        agent_no_pref = agent.copy()
        agent_no_pref.C = np.zeros_like(agent.C)  # uniform preferences
        results["no_preferences"] = run_test_battery(agent_no_pref, env)
        
        # Lesion epistemic value
        agent_no_epistemic = agent.copy()
        agent_no_epistemic.use_epistemic_value = False
        results["no_epistemic"] = run_test_battery(agent_no_epistemic, env)
    
    return results
```

---

## Directory Structure

```
desire-research/
├── README.md                     # This file
├── pyproject.toml                # Dependencies and project config
├── .gitignore
│
├── src/
│   ├── __init__.py
│   │
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── base_env.py          # Abstract environment interface
│   │   ├── foraging_grid.py     # Main grid world implementation
│   │   ├── resources.py         # Resource types and behaviors
│   │   ├── terrain.py           # Terrain features
│   │   └── modes.py             # Test mode configurations
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py        # Abstract agent interface (CRITICAL: defines get_internal_state())
│   │   ├── heuristic.py         # Hardcoded heuristic agent
│   │   ├── rl_agent.py          # PPO/DQN wrapper with internal state access
│   │   ├── active_inference.py  # pymdp-based active inference agent
│   │   └── custom_ai.py         # Custom active inference implementation (optional)
│   │
│   ├── tests/                   # Predicate tests (NOT unit tests)
│   │   ├── __init__.py
│   │   ├── base_test.py         # Abstract test interface
│   │   ├── anticipation.py      # AA test
│   │   ├── frustration.py       # FI test
│   │   ├── means_end.py         # MED test
│   │   ├── extinction.py        # ER test
│   │   ├── opportunity_cost.py  # OCS test
│   │   └── state_dependent.py   # SDV test
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Compute predicate scores
│   │   ├── fitting.py           # Exponential/power law fitting
│   │   ├── interpretability.py  # Representation analysis
│   │   └── visualization.py     # Plotting utilities
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Experiment logging
│       └── config.py            # Configuration management
│
├── experiments/
│   ├── train_agents.py          # Training script
│   ├── run_tests.py             # Run full test battery
│   ├── analyze_results.py       # Generate analysis and figures
│   └── configs/
│       ├── default.yaml         # Default experiment config
│       ├── quick_test.yaml      # Fast iteration config
│       └── full_experiment.yaml # Publication-ready config
│
├── notebooks/
│   ├── 01_environment_demo.ipynb
│   ├── 02_agent_comparison.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_interpretability.ipynb
│
├── tests/                       # Unit tests (pytest)
│   ├── test_environment.py
│   ├── test_agents.py
│   └── test_metrics.py
│
├── results/                     # Experiment outputs (gitignored except examples)
│   └── .gitkeep
│
└── paper/
    ├── main.tex                 # Paper draft
    ├── figures/
    └── references.bib
```

---

## Implementation Priorities

### Priority 1: Minimum Viable Experiment (Week 1-2)

Get signal fast. Implement:

1. **Simple grid environment** (20x20, basic resources)
2. **Heuristic agent** (baseline)
3. **PPO agent** (use stable-baselines3)
4. **Basic active inference agent** (use pymdp)
5. **Two tests only**: Frustration (FI) and Means-End (MED)

**Success metric**: Clear behavioral difference between agent types on at least one test.

### Priority 2: Full Test Battery (Week 3-4)

Expand to all six predicates. Ensure statistical rigor:

- N ≥ 30 per condition
- Proper controls
- Effect size calculations
- Multiple comparison corrections

### Priority 3: Interpretability Layer (Week 5-6)

Add internal state analysis:

- Implement `get_internal_state()` for all agents
- Representation clustering
- Goal probes
- Ablation studies

### Priority 4: Scale and Robustness (Week 7-8)

- Run on multiple environment variations
- Test sensitivity to hyperparameters
- Ensure results replicate across random seeds

### Priority 5: Paper Writing (Week 9-12)

- Introduction and related work
- Methods
- Results
- Discussion (including limitations)
- Figures and tables

---

## Key Implementation Details

### Critical: Agent Interface

All agents MUST implement this interface for comparability:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class BaseAgent(ABC):
    """
    All agents must implement this interface.
    
    CRITICAL: get_internal_state() must return comparable representations
    for interpretability analysis.
    """
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """Select action given observation."""
        pass
    
    @abstractmethod
    def get_internal_state(self) -> np.ndarray:
        """
        Return internal state representation.
        
        For RL agents: hidden layer activations
        For Active Inference: belief state (posterior over hidden states)
        For Heuristic: one-hot encoding of current rule
        
        Must be fixed-size vector for comparability.
        """
        pass
    
    @abstractmethod
    def get_internal_response(self, observation: np.ndarray) -> np.ndarray:
        """
        Return internal response to observation WITHOUT taking action.
        Used for state-dependent valuation tests.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for new episode."""
        pass
    
    def set_energy(self, energy: float) -> None:
        """
        Set agent's energy level directly (for controlled experiments).
        Default implementation does nothing (for agents without internal energy).
        """
        pass
```

### Critical: Logging Everything

For reproducibility and analysis, log:

```python
@dataclass
class StepLog:
    episode: int
    step: int
    observation: np.ndarray
    internal_state: np.ndarray
    action: int
    reward: float
    info: Dict[str, Any]
    timestamp: float

@dataclass
class EpisodeLog:
    episode: int
    agent_type: str
    steps: List[StepLog]
    total_reward: float
    survival_time: int
    death_cause: Optional[str]
    random_seed: int
```

### Critical: Statistical Analysis

```python
def compare_agent_types(results: Dict[str, List[float]], alpha: float = 0.05):
    """
    Compare predicate scores across agent types.
    
    Uses:
    - Kruskal-Wallis test for overall difference
    - Mann-Whitney U for pairwise comparisons
    - Bonferroni correction for multiple comparisons
    - Effect size: Cliff's delta
    """
    pass
```

---

## Expected Outcomes

### Positive Result (Supports Hypothesis)

Active inference agents show:
- AA_score significantly > RL and Heuristic
- FI_score significantly > RL and Heuristic
- Power-law extinction (RL shows exponential)
- State-dependent valuation effects
- Internal representations cluster by goal state, not just action

**Interpretation**: Free energy minimization produces desire-like behavioral signatures that reward maximization does not.

### Negative Result (Refutes Hypothesis)

No systematic differences between active inference and RL agents.

**Interpretation**: Either (a) desire-like behavior is not specific to free energy minimization, or (b) our operationalization of desire-like behavior is wrong.

**This is still publishable** as a null result with important theoretical implications.

### Mixed Result (Most Likely)

Some predicates differentiate agent types, others don't.

**Interpretation**: Desire may be a cluster concept with multiple dissociable components. Some may be tied to free energy minimization, others may emerge from any learning system.

**This is the most interesting outcome** for a nuanced discussion.

---

## Dependencies

```toml
[project]
name = "desire-research"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "pandas>=2.0",
    "gymnasium>=0.28",
    "stable-baselines3>=2.0",
    "torch>=2.0",
    "pymdp>=0.0.7",
    "scikit-learn>=1.2",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "wandb>=0.15",  # experiment tracking
    "jupyter>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.0.270",
]
```

---

## References to Engage

The paper must position against this literature:

1. **Friston, K.** (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*
2. **England, J.** (2013). Statistical physics of self-replication. *Journal of Chemical Physics*
3. **Berridge, K.** (2009). 'Liking' and 'wanting' food rewards. *Physiology & Behavior*
4. **Oudeyer, P-Y.** (2007). Intrinsic motivation systems for autonomous mental development. *IEEE Transactions*
5. **Schmidhuber, J.** (2010). Formal theory of creativity, fun, and intrinsic motivation. *IEEE Transactions*
6. **Seth, A.** (2015). The cybernetic Bayesian brain. *Open MIND*
7. **Pezzulo, G.** (2018). Hierarchical active inference. *Frontiers in Psychology*

---

## Questions for the Coding Agent

When implementing, consider:

1. **pymdp vs custom**: pymdp is faster to start but less flexible. For initial experiments, use pymdp. Consider custom implementation if pymdp constrains the state space too much.

2. **Observation encoding**: How to represent the local view for active inference? Discrete factorized states (pymdp native) or continuous (requires custom)?

3. **Planning horizon**: Active inference planning depth significantly affects behavior. Start with horizon=3, experiment with longer.

4. **RL architecture**: Need to ensure RL agent has similar capacity to active inference agent. Match parameter counts roughly.

5. **Internal state extraction**: For PPO, which layers to extract? Recommend penultimate layer before policy head.

6. **Reproducibility**: Set all random seeds. Log everything. Use wandb or similar for experiment tracking.

---

## Success Criteria for This Project

1. **Minimum publishable unit**: Clear statistical difference between agent types on ≥2 predicates, with interpretability analysis supporting the behavioral findings.

2. **Target venue**: 
   - Primary: *ICLR* or *NeurIPS* (machine learning + cognitive science)
   - Secondary: *Cognitive Science* or *Artificial Life*
   - arXiv preprint regardless

3. **Timeline**: 90 days to arXiv-ready draft if working 20+ hours/week.

---

## Final Notes for Claude Code

When you implement this:

1. **Start with the environment**. Get it rendering, manually playable, before adding agents.

2. **Verify each agent type individually** before comparative experiments.

3. **Implement logging from day one**. You'll thank yourself later.

4. **Write tests for the metrics**. Make sure your fitting functions work on synthetic data with known properties.

5. **Keep the MVP small**. Get signal with 2 tests before building all 6.

6. **Visualize constantly**. Plot trajectories, internal states, learning curves. Understanding > code coverage.

7. **The research question matters more than the code quality**. Optimize for scientific insight, not engineering elegance.

Good luck. This could be a genuinely novel contribution.
