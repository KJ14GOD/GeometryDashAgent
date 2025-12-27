"""
Test a trained PPO model on Geometry Dash
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from environment.geometry_dash_env import GeometryDashEnv


def test_model(model_path: str, num_episodes: int = 10):
    """
    Test a trained PPO model.
    
    Args:
        model_path: Path to saved model (without .zip extension)
        num_episodes: Number of episodes to run
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    print("Creating environment...")
    env = GeometryDashEnv()
    
    print("\nStarting test in 3 seconds - FOCUS THE GAME WINDOW!")
    time.sleep(3)
    
    episode_rewards = []
    episode_lengths = []
    
    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episode {ep + 1}: Reward={total_reward:.1f}, Steps={steps}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    
    finally:
        env.close()
    
    if episode_rewards:
        print(f"\n{'=' * 40}")
        print(f"Results over {len(episode_rewards)} episodes:")
        print(f"  Mean reward: {sum(episode_rewards) / len(episode_rewards):.1f}")
        print(f"  Mean steps:  {sum(episode_lengths) / len(episode_lengths):.1f}")
        print(f"  Max reward:  {max(episode_rewards):.1f}")
        print(f"  Max steps:   {max(episode_lengths)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PPO model on Geometry Dash")
    parser.add_argument("--model", type=str, default="rl/models/ppo_geometry_dash_final",
                        help="Path to model (without .zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    
    args = parser.parse_args()
    test_model(args.model, args.episodes)
