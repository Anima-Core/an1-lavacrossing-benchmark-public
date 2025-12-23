#!/usr/bin/env python3
"""
Test script for the LavaCrossing benchmark.
This verifies the benchmark works and demonstrates the key metrics.
"""

import subprocess
import sys
import os

def test_benchmark_installation():
    """Test if required packages are installed"""
    try:
        import gymnasium
        import minigrid
        import numpy as np
        print("[OK] All required packages installed")
        return True
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        print("Install with: pip install gymnasium minigrid numpy")
        return False

def run_benchmark_test(baseline="random", eval_seeds=20):
    """Run a quick benchmark test"""
    print(f"\n🧪 Testing {baseline} baseline with {eval_seeds} evaluation episodes...")
    
    cmd = [
        sys.executable, "an1_lavacrossing_oneshot_benchmark.py",
        "--baseline", baseline,
        "--eval-seeds", str(eval_seeds),
        "--death-search-limit", "100",
        "--train-seed", "0"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"[OK] {baseline} benchmark completed successfully")
            
            # Extract key metrics from output
            output = result.stdout
            if "post_death_lava_deaths:" in output:
                for line in output.split('\n'):
                    if "post_death_lava_deaths:" in line:
                        deaths = line.split(':')[-1].strip()
                        print(f"  📊 Post-death lava deaths: {deaths}")
                    elif "goal_rate:" in line and "post-death" in output[max(0, output.find(line)-200):output.find(line)]:
                        rate = line.split(':')[-1].strip()
                        print(f"  📊 Goal success rate: {rate}")
            
            return True
        else:
            print(f"[ERROR] {baseline} benchmark failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] {baseline} benchmark timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error running {baseline} benchmark: {e}")
        return False

def main():
    print("=" * 60)
    print("LAVACROSSING BENCHMARK TEST SUITE")
    print("=" * 60)
    
    # Test 1: Check installation
    if not test_benchmark_installation():
        return False
    
    # Test 2: Run random baseline (should work always)
    if not run_benchmark_test("random", eval_seeds=10):
        print("[ERROR] Random baseline test failed - check MiniGrid installation")
        return False
    
    # Test 3: Try demo agent
    print("\n" + "="*40)
    print("Testing Demo Agent Integration")
    print("="*40)
    
    if run_benchmark_test("demo-agent", eval_seeds=10):
        print("[OK] Demo agent integration successful!")
    else:
        print("⚠️  Demo agent test failed - check adapter implementation")
    
    print("\n" + "="*60)
    print("BENCHMARK TEST SUMMARY")
    print("="*60)
    print("The benchmark harness is working!")
    print("\nKey points:")
    print("• Random baseline should show >0 post-death lava deaths")
    print("• Demo agent should show 0 post-death lava deaths (perfect constraint)")
    print("• Goal success rate shows overall task performance")
    print("\nTo run full evaluation:")
    print("  python an1_lavacrossing_oneshot_benchmark.py --baseline demo-agent --eval-seeds 200")
    print("\nTo integrate your own agent:")
    print("  1. Add your agent to an1_lavacrossing_adapter.py")
    print("  2. Update the baseline choices in the benchmark script")
    print("  3. Run with --baseline your-agent-name")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)