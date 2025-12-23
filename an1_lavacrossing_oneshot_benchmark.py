""" AN1 One-Shot LavaCrossing Benchmark Harness
==========================================

What this script does (genuine, defensible test):
1) Uses the *official* MiniGrid LavaCrossing environments (standard benchmark).
2) Runs a strict one-shot protocol:
   - Find the first lava death on a seeded episode.
   - After that moment, the agent must apply a hard constraint ("never step into lava again").
   - No training, no gradient updates, no replay buffers, no tuning.
3) Evaluates generalization on many unseen seeds.
4) Produces transparent, auditable metrics:
   - Death rate before vs after first death
   - Post-death deaths (the key claim)
   - Success rate, steps, and returns
5) Optionally compares against a baseline agent (random by default),
   and includes hooks to plug in your own policy easily.

Install:
  pip install gymnasium minigrid numpy

Run:
  python an1_lavacrossing_oneshot_benchmark.py --env MiniGrid-LavaCrossingS11N5-v0 --train-seed 0 --eval-seeds 200

Notes:
- MiniGrid episodes are deterministic given a seed.
- This harness is honest: it does not hide failures and it reports all outcomes.
"""

import argparse
import time
import json
import numpy as np
import gymnasium as gym

try:
    import minigrid  # noqa: F401
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import WorldObj
except Exception as e:
    raise RuntimeError(
        "MiniGrid not installed. Run: pip install minigrid gymnasium"
    ) from e


# -----------------------------
# Gold Layout Support
# -----------------------------

def load_gold_layouts(path):
    """Load gold layouts from JSONL file"""
    layouts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            layouts.append(json.loads(line))
    return layouts


def set_env_to_layout(env, layout):
    """
    Rebuild the grid from saved encoding.
    This replays the exact layout from the gold corpus.
    """
    grid_data = layout["grid"]
    h = len(grid_data)
    w = len(grid_data[0])

    g = Grid(w, h)
    for y in range(h):
        for x in range(w):
            t, c, s = grid_data[y][x]
            if t == 0:
                g.set(x, y, None)
            else:
                try:
                    obj = WorldObj.decode(t, c, s)
                    g.set(x, y, obj)
                except Exception as e:
                    print(f"Warning: Could not decode object at ({x},{y}): {e}")
                    g.set(x, y, None)

    env.unwrapped.grid = g
    env.unwrapped.agent_pos = tuple(layout["agent_pos"])
    env.unwrapped.agent_dir = int(layout["agent_dir"])
    if "mission" in layout:
        env.unwrapped.mission = layout["mission"]


def episode_rollout_gold(env, agent, layout, max_steps=None, render=False):
    """
    Runs one episode using a gold layout.
    Returns a dict with detailed outcomes.
    """
    # Reset environment and apply gold layout
    obs, info = env.reset(seed=layout["seed"])  # Initialize internals
    set_env_to_layout(env, layout)
    
    # Get fresh observation after layout is set - try different methods
    try:
        obs = env.unwrapped.gen_obs()
    except AttributeError:
        try:
            obs = env.unwrapped._gen_obs()
        except AttributeError:
            # Fallback: step with no-op action to get observation
            obs, _, _, _, info = env.step(6)  # 6 is typically 'done' action
            env.unwrapped.step_count -= 1  # Undo the step count
    
    info = {}
    
    terminated = False
    truncated = False
    steps = 0
    total_reward = 0.0
    died_in_lava = False
    reached_goal = False

    if max_steps is None:
        max_steps = getattr(env.unwrapped, "max_steps", 1000)

    while not (terminated or truncated) and steps < max_steps:
        if render:
            env.render()

        action = agent.act(env, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += float(reward)

        if terminated:
            # Check if agent position is lava or goal
            ax, ay = env.unwrapped.agent_pos
            cell = env.unwrapped.grid.get(ax, ay)
            if cell is not None and (cell.type == "lava" or cell.__class__.__name__.lower() == "lava"):
                died_in_lava = True
            if cell is not None and (cell.type == "goal" or cell.__class__.__name__.lower() == "goal"):
                reached_goal = True

    return {
        "seed": layout["seed"],
        "layout_id": layout.get("layout_id", layout["seed"]),
        "steps": steps,
        "return": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "died_in_lava": bool(died_in_lava),
        "reached_goal": bool(reached_goal),
    }


# -----------------------------
# Helpers
# -----------------------------

def is_lava_at(env, fwd_pos):
    """
    Checks if the cell in front of the agent is lava.
    MiniGrid uses grid objects, lava is typically instance of Lava.
    We detect by type name to avoid brittle imports.
    """
    grid = env.unwrapped.grid
    x, y = fwd_pos
    cell = grid.get(x, y)
    if cell is None:
        return False
    return cell.type == "lava" or cell.__class__.__name__.lower() == "lava"


def is_goal_at(env, fwd_pos):
    grid = env.unwrapped.grid
    x, y = fwd_pos
    cell = grid.get(x, y)
    if cell is None:
        return False
    return cell.type == "goal" or cell.__class__.__name__.lower() == "goal"


def forward_pos(env):
    ax, ay = env.unwrapped.agent_pos
    dx, dy = env.unwrapped.dir_vec
    return (ax + dx, ay + dy)


def episode_rollout(env, agent, seed, max_steps=None, render=False):
    """
    Runs one episode. Returns a dict with detailed outcomes.
    """
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    steps = 0
    total_reward = 0.0
    died_in_lava = False
    reached_goal = False

    if max_steps is None:
        # MiniGrid environments usually have a max_steps field
        max_steps = getattr(env.unwrapped, "max_steps", 1000)

    while not (terminated or truncated) and steps < max_steps:
        if render:
            env.render()

        action = agent.act(env, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += float(reward)

        # MiniGrid conventions:
        # - termination can occur on goal or lava
        # - info may include 'terminated' reasons depending on wrappers
        # We'll infer lava death by checking if agent is on lava cell after termination.
        if terminated:
            # If terminated, check if agent position is lava
            ax, ay = env.unwrapped.agent_pos
            cell = env.unwrapped.grid.get(ax, ay)
            if cell is not None and (cell.type == "lava" or cell.__class__.__name__.lower() == "lava"):
                died_in_lava = True
            if cell is not None and (cell.type == "goal" or cell.__class__.__name__.lower() == "goal"):
                reached_goal = True

    return {
        "seed": seed,
        "steps": steps,
        "return": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "died_in_lava": bool(died_in_lava),
        "reached_goal": bool(reached_goal),
    }


# -----------------------------
# Agents
# -----------------------------

class RandomAgent:
    """
    Honest baseline: uniform random actions.
    """
    def __init__(self, action_space, rng):
        self.action_space = action_space
        self.rng = rng

    def act(self, env, obs, info):
        return int(self.rng.integers(0, self.action_space.n))


class OneShotLavaAvoidanceWrapperAgent:
    """
    This is the minimal "hard constraint after first death" agent wrapper.

    It wraps any base agent.
    Before first death:
      - delegates to base agent
    After first lava death:
      - enforces a hard constraint: never take 'forward' into lava
      - if forward is lava, choose an alternate safe action

    This is NOT training. It is a rule that activates from a single catastrophic event.
    The point of the benchmark is to measure whether your policy can learn and apply
    similar constraints, but this wrapper gives you the exact harness and metric definition.
    """
    # MiniGrid actions (typical):
    # 0: left, 1: right, 2: forward, 3: pickup, 4: drop, 5: toggle, 6: done
    FORWARD_ACTION = 2
    LEFT_ACTION = 0
    RIGHT_ACTION = 1

    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.hard_law_active = False
        self.first_death_seed = None

    def notify_lava_death(self, seed):
        if not self.hard_law_active:
            self.hard_law_active = True
            self.first_death_seed = seed
            
            # If base agent has death notification, notify it too
            if hasattr(self.base_agent, 'notify_death'):
                self.base_agent.notify_death()
                print("[OK] Notified base agent of death event")

    def act(self, env, obs, info):
        a = int(self.base_agent.act(env, obs, info))

        if not self.hard_law_active:
            return a

        # Hard law: do not step forward into lava.
        fwd = forward_pos(env)
        if a == self.FORWARD_ACTION and is_lava_at(env, fwd):
            # Choose a deterministic safe alternative:
            # Prefer turn left, else turn right, else do nothing.
            # Turning is always safe relative to stepping into lava.
            return self.LEFT_ACTION

        return a


# -----------------------------
# Protocol
# -----------------------------

def find_first_lava_death_gold(env, agent, layouts, max_steps, render):
    """
    Searches through gold layouts until the agent dies in lava.
    Returns (death_layout, rollout_info) or (None, None) if not found.
    """
    for layout in layouts:
        r = episode_rollout_gold(env, agent, layout, max_steps=max_steps, render=render)
        if r["died_in_lava"]:
            return layout, r
    return None, None


def evaluate_gold_layouts(env, agent, layouts, max_steps, render):
    """Evaluate agent on a list of gold layouts"""
    results = []
    for layout in layouts:
        results.append(episode_rollout_gold(env, agent, layout, max_steps=max_steps, render=render))
    return results


def find_first_lava_death(env, agent, start_seed, search_limit, max_steps, render):
    """
    Searches seeds from start_seed upward until the agent dies in lava.
    Returns (death_seed, rollout_info) or (None, None) if not found.
    """
    for k in range(search_limit):
        seed = start_seed + k
        r = episode_rollout(env, agent, seed=seed, max_steps=max_steps, render=render)
        if r["died_in_lava"]:
            return seed, r
    return None, None


def evaluate_seeds(env, agent, seeds, max_steps, render):
    results = []
    for s in seeds:
        results.append(episode_rollout(env, agent, seed=s, max_steps=max_steps, render=render))
    return results


def summarize(results):
    n = len(results)
    if n == 0:
        return {}

    deaths = sum(1 for r in results if r["died_in_lava"])
    goals = sum(1 for r in results if r["reached_goal"])
    term = sum(1 for r in results if r["terminated"])
    trunc = sum(1 for r in results if r["truncated"])
    avg_steps = float(np.mean([r["steps"] for r in results]))
    avg_return = float(np.mean([r["return"] for r in results]))

    return {
        "episodes": n,
        "lava_deaths": deaths,
        "lava_death_rate": deaths / n,
        "goals": goals,
        "goal_rate": goals / n,
        "terminated": term,
        "truncated": trunc,
        "avg_steps": avg_steps,
        "avg_return": avg_return,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="MiniGrid-LavaCrossingS11N5-v0")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="Seed to start searching for the first lava death event.")
    ap.add_argument("--death-search-limit", type=int, default=500,
                    help="How many consecutive seeds to search for the first lava death.")
    ap.add_argument("--eval-seeds", type=int, default=200,
                    help="How many unseen seeds to evaluate after the first death.")
    ap.add_argument("--eval-seed-start", type=int, default=10_000,
                    help="Seed range start for post-death generalization evaluation.")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--baseline", type=str, default="random",
                    choices=["random", "demo-agent"],
                    help="Baseline policy. Options: random, demo-agent")
    ap.add_argument("--gold-layouts", type=str, default=None,
                    help="Path to gold layouts JSONL file (overrides seed-based generation)")
    ap.add_argument("--rng-seed", type=int, default=123,
                    help="RNG seed for baseline agent randomness (not env seed).")
    args = ap.parse_args()

    print("=" * 80)
    print("ONE-SHOT LAVACROSSING BENCHMARK HARNESS")
    print("=" * 80)
    print(f"Environment: {args.env}")
    
    # Check if using gold layouts
    using_gold_layouts = args.gold_layouts is not None
    if using_gold_layouts:
        print(f"Gold Layouts: {args.gold_layouts}")
        try:
            all_layouts = load_gold_layouts(args.gold_layouts)
            print(f"Loaded {len(all_layouts)} gold layouts")
        except Exception as e:
            print(f"Error loading gold layouts: {e}")
            return
    else:
        print("Using seed-based generation")
    
    print("Protocol:")
    print("  1) Run baseline policy until first lava death occurs.")
    print("  2) Activate hard-law: never step forward into lava.")
    print("  3) Evaluate on unseen layouts and report post-death deaths and success.")
    print()

    env = gym.make(args.env, render_mode="human" if args.render else None)

    rng = np.random.default_rng(args.rng_seed)

    # Baseline agent selection
    if args.baseline == "random":
        base_agent = RandomAgent(env.action_space, rng=rng)
    elif args.baseline == "demo-agent":
        try:
            from an1_lavacrossing_adapter import create_demo_agent
            base_agent = create_demo_agent()
            print("[OK] Using Demo Agent")
        except ImportError as e:
            print(f"Error importing demo agent: {e}")
            print("Falling back to random agent")
            base_agent = RandomAgent(env.action_space, rng=rng)
    else:
        raise ValueError("Unsupported baseline")

    # Wrap baseline in the one-shot hard-law wrapper
    agent = OneShotLavaAvoidanceWrapperAgent(base_agent)

    if using_gold_layouts:
        # Gold layout protocol
        # Phase 0: Pre-death evaluation snapshot
        pre_layouts = all_layouts[:min(50, len(all_layouts)//4)]
        t0 = time.time()
        pre_results = evaluate_gold_layouts(env, base_agent, pre_layouts, max_steps=args.max_steps, render=False)
        pre_summary = summarize(pre_results)
        t1 = time.time()

        # Phase 1: Find first lava death in remaining layouts
        death_search_start = len(pre_layouts)
        death_search_end = min(death_search_start + args.death_search_limit, len(all_layouts) - args.eval_seeds)
        death_search_layouts = all_layouts[death_search_start:death_search_end]
        death_layout, death_rollout = find_first_lava_death_gold(
            env, base_agent, death_search_layouts, max_steps=args.max_steps, render=args.render
        )

        if death_layout is None:
            print("No lava death found in gold layout search.")
            print("This can happen if the baseline agent never steps into lava in the provided layouts.")
            print("Try using a different gold layout corpus or a different baseline agent.")
            return

        # Activate hard law from the single catastrophic event
        agent.notify_lava_death(death_layout["seed"])

        # Phase 2: Post-death generalization evaluation on remaining layouts
        eval_start_idx = death_search_end
        eval_layouts = all_layouts[eval_start_idx:eval_start_idx + args.eval_seeds]
        
        if len(eval_layouts) < args.eval_seeds:
            print(f"Warning: Only {len(eval_layouts)} evaluation layouts available (requested {args.eval_seeds})")
        
        post_results = evaluate_gold_layouts(env, agent, eval_layouts, max_steps=args.max_steps, render=False)
        post_summary = summarize(post_results)

        # Results reporting for gold layouts
        print("=" * 80)
        print("RESULTS (GOLD LAYOUT CORPUS)")
        print("=" * 80)

        print("Pre-death snapshot (baseline only, before any hard law):")
        print(f"  Layouts: {len(pre_layouts)} from gold corpus")
        for k, v in pre_summary.items():
            print(f"  {k}: {v}")
        print(f"  Time: {t1 - t0:.2f}s")
        print()

        print("First catastrophic event (single data point that activates constraint):")
        print(f"  First lava death layout seed: {death_layout['seed']}")
        print(f"  Episode steps: {death_rollout['steps']}")
        print(f"  Episode return: {death_rollout['return']}")
        print()

        print("Post-death evaluation (hard law active, no training, gold layouts):")
        print(f"  Layouts: {len(eval_layouts)} from gold corpus")
        for k, v in post_summary.items():
            print(f"  {k}: {v}")
        print()

        print("Key claim metric (GOLD STANDARD):")
        print("  post_death_lava_deaths should be 0 for true one-shot hazard invariance.")
        print(f"  post_death_lava_deaths: {post_summary.get('lava_deaths', None)}")
        print(f"  Gold corpus file: {args.gold_layouts}")
        print()

    else:
        # Original seed-based protocol
        # Phase 0: Pre-death evaluation snapshot (optional but useful)
        pre_seeds = list(range(args.train_seed, args.train_seed + min(50, args.death_search_limit)))
        t0 = time.time()
        pre_results = evaluate_seeds(env, base_agent, pre_seeds, max_steps=args.max_steps, render=False)
        pre_summary = summarize(pre_results)
        t1 = time.time()

        # Phase 1: Find first lava death
        death_seed, death_rollout = find_first_lava_death(
            env, base_agent, start_seed=args.train_seed,
            search_limit=args.death_search_limit,
            max_steps=args.max_steps, render=args.render
        )

        if death_seed is None:
            print("No lava death found in seed search window.")
            print("This can happen if the baseline agent never steps into lava, or if")
            print("the environment difficulty is low relative to the baseline behavior.")
            print("Try increasing --death-search-limit or changing to a harder env, like S13N7.")
            return

        # Activate hard law from the single catastrophic event
        agent.notify_lava_death(death_seed)

        # Phase 2: Post-death generalization evaluation
        eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds))
        post_results = evaluate_seeds(env, agent, eval_seeds, max_steps=args.max_steps, render=False)
        post_summary = summarize(post_results)

        # Results reporting for seed-based
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)

        print("Pre-death snapshot (baseline only, before any hard law):")
        print(f"  Seeds: {pre_seeds[0]}..{pre_seeds[-1]}")
        for k, v in pre_summary.items():
            print(f"  {k}: {v}")
        print(f"  Time: {t1 - t0:.2f}s")
        print()

        print("First catastrophic event (single data point that activates constraint):")
        print(f"  First lava death seed: {death_seed}")
        print(f"  Episode steps: {death_rollout['steps']}")
        print(f"  Episode return: {death_rollout['return']}")
        print()

        print("Post-death evaluation (hard law active, no training, unseen seeds):")
        print(f"  Seeds: {eval_seeds[0]}..{eval_seeds[-1]}")
        for k, v in post_summary.items():
            print(f"  {k}: {v}")
        print()

        print("Key claim metric:")
        print("  post_death_lava_deaths should be 0 for true one-shot hazard invariance.")
        print(f"  post_death_lava_deaths: {post_summary.get('lava_deaths', None)}")
        print()

    env.close()

    print("How to plug in your agent:")
    print("  1. Implement an agent class with act(env, obs, info) -> action method")
    print("  2. Add it to an1_lavacrossing_adapter.py")
    print("  3. Run with --baseline your-agent-name")
    
    if using_gold_layouts:
        print()
        print("Gold Layout Corpus Benefits:")
        print("  [OK] Reproducible across runs and systems")
        print("  [OK] Versionable and shareable dataset")
        print("  [OK] No dependency on environment RNG")
        print("  [OK] Auditable layout difficulty distribution")


if __name__ == "__main__":
    main()