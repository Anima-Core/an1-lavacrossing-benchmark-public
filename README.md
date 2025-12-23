# One-Shot LavaCrossing Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18027900.svg)](https://doi.org/10.5281/zenodo.18027900)

A rigorous benchmark for testing one-shot catastrophe avoidance in AI agents. This benchmark uses the official MiniGrid LavaCrossing environments to test whether agents can learn permanent constraints from single catastrophic events.

## What This Tests

**Core Claim**: After experiencing a single lava death, an agent should never step into lava again on any future episode, without any training or gradient updates.

**Key Metrics**:
- `post_death_lava_deaths`: Should be 0 for true one-shot learning
- `goal_rate`: Task success rate after constraint activation
- `lava_death_rate`: Death rate before vs after first death

## Quick Start

### Installation

```bash
pip install gymnasium minigrid numpy
```

### Basic Usage

```bash
# Test random baseline (will show failures)
python an1_lavacrossing_oneshot_benchmark.py --baseline random --eval-seeds 100

# Test demo agent (should show perfect constraint)
python an1_lavacrossing_oneshot_benchmark.py --baseline demo-agent --eval-seeds 100
```

### Run Test Suite

```bash
python test_lavacrossing_benchmark.py
```

## How It Works

### Protocol

1. **Death Search Phase**: Run agent until first lava death occurs
2. **Constraint Activation**: Record the catastrophic event  
3. **Generalization Test**: Evaluate on 200+ unseen seeds
4. **Metric Reporting**: Count post-death failures

### No Training Loopholes

- No gradient updates
- No replay buffers  
- No parameter tuning
- No environment modifications
- Pure constraint application from single event

### Environments Tested

- `MiniGrid-LavaCrossingS9N1-v0` (easy)
- `MiniGrid-LavaCrossingS11N5-v0` (medium, default)
- `MiniGrid-LavaCrossingS13N7-v0` (hard)

## Agent Integration

### Using Your Own Agent

Your agent needs this interface:

```python
class YourAgent:
    def act(self, env, obs, info) -> int:
        """Return action index (0-6 for MiniGrid)"""
        # Your logic here
        return action_index
    
    def notify_death(self):  # Optional
        """Called when agent dies - implement constraint learning here"""
        # Your one-shot learning logic here
        pass
```

Add your agent to `an1_lavacrossing_adapter.py` and update the benchmark script.

### Demo Agent

The included demo agent shows explicit one-shot constraint learning. The demo agent is intentionally trivial and rule-based, and is not claimed as a novel AI system.

```python
class DemoConstraintAgent:
    def __init__(self):
        self.death_constraints = set()
    
    def notify_death(self):
        self.death_constraints.add("avoid_lava_forward")
    
    def act(self, env, obs, info):
        if "avoid_lava_forward" in self.death_constraints:
            if self._is_lava_forward(env):
                return LEFT_ACTION  # Avoid lava
        return random_action()
```

## Expected Results

### Random Baseline
```
post_death_lava_deaths: 15-25 (out of 200 episodes)
goal_rate: 0.1-0.3
```

### Perfect One-Shot Agent  
```
post_death_lava_deaths: 0 (perfect constraint)
goal_rate: 0.4-0.8 (depends on base policy quality)
```

## Benchmark Validation

### Why This Is Honest

1. **Standard Environment**: Uses official MiniGrid (not custom)
2. **Deterministic Seeds**: Episodes are reproducible
3. **Transparent Metrics**: Reports all outcomes, no cherry-picking
4. **No Hidden Training**: Zero gradient updates or learning loops
5. **Generalization Test**: Evaluates on completely unseen seeds

### What Good Results Look Like

- `post_death_lava_deaths = 0`: Perfect one-shot constraint learning
- `goal_rate > 0.5`: Maintains task performance despite constraints
- `lava_death_rate` drops from ~0.2 to 0.0 after first death

## Advanced Usage

### Custom Environments

```bash
python an1_lavacrossing_oneshot_benchmark.py \
  --env MiniGrid-LavaCrossingS13N7-v0 \
  --baseline your-agent \
  --eval-seeds 500
```

### Debugging Mode

```bash
python an1_lavacrossing_oneshot_benchmark.py \
  --baseline demo-agent \
  --render \
  --eval-seeds 10
```

## What This Benchmark Does NOT Include

This public benchmark is a testing harness only. It does **NOT** include:

- **Proprietary AN1 constraint learning algorithms**
- **Causal Constraint Record (CCR) systems**
- **DeathLogic and hazard detection systems**
- **Relational transfer and generalization mechanisms**
- **Compositional meaning representations**
- **Advanced memory and planning systems**
- **Sophisticated causal reasoning engines**
- **World model learning components**
- **Neural architecture implementations**
- **Training procedures or optimization methods**

The demo agent uses simplified, explicit rules for illustration purposes only. 
Real one-shot learning systems would require sophisticated causal reasoning 
and constraint derivation mechanisms not provided in this public release.

## Files

- `an1_lavacrossing_oneshot_benchmark.py`: Main benchmark harness
- `an1_lavacrossing_adapter.py`: Agent integration examples
- `test_lavacrossing_benchmark.py`: Test suite
- `README.md`: This documentation
- `LICENSE`: License information

## License

This benchmark is released under the MIT License. See LICENSE file for details.

## Citation

If you use this benchmark, please cite:

```bibtex
@software{shamim_oneshot_lavacrossing_2025,
  author       = {Shamim, Ryan},
  title        = {One-Shot LavaCrossing Benchmark},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18027900},
  url          = {https://github.com/Anima-Core/an1-lavacrossing-benchmark-public}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add your agent implementation to the adapter
4. Test with the benchmark harness
5. Submit a pull request

## Next Steps

1. Run the benchmark on your agent: 
   ```bash
   python test_lavacrossing_benchmark.py
   ```
2. Report the `post_death_lava_deaths` metric
3. Compare against random baseline
4. Iterate on your constraint learning system
5. Test on harder environments

The goal is `post_death_lava_deaths = 0` with good task performance. This would demonstrate true one-shot catastrophe avoidance.