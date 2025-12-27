import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.geometry_dash_env import GeometryDashEnv
  

class ImitationFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that matches the imitation learning model architecture.
    
    Imitation model: 84 → 256 → 256 → 2
    We extract:      84 → 256 → 256 (features_dim=256)
    PPO adds:                       → action_net(2) + value_net(1)
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.net = nn.Sequential(
            nn.Linear(84, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeat action for N frames to speed up training.
    Returns max reward and last observation.
    """
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info


def load_imitation_weights_full(model: PPO, imitation_path: str) -> bool:
    """
    Load FULL imitation learning weights into PPO:
    - Feature extractor (84→256→256)
    - Action network (256→2)
    
    Returns True if successful, False otherwise.
    """
    try:
        imitation_state_dict = torch.load(imitation_path, map_location="cpu", weights_only=True)
        
        # === Load Feature Extractor ===
        features_extractor = model.policy.features_extractor
        fe_state_dict = {
            'net.0.weight': imitation_state_dict['neural_network.0.weight'],
            'net.0.bias': imitation_state_dict['neural_network.0.bias'],
            'net.2.weight': imitation_state_dict['neural_network.3.weight'],
            'net.2.bias': imitation_state_dict['neural_network.3.bias'],
        }
        features_extractor.load_state_dict(fe_state_dict)
        print(f"  ✓ Loaded feature extractor weights (84→256→256)")
        
        # === Load Action Network ===
        # IL model: neural_network.6 is Linear(256, 2)
        # PPO model: action_net is Linear(256, 2) when net_arch=dict(pi=[], vf=[])
        action_net = model.policy.action_net
        
        # Check shapes match
        il_weight = imitation_state_dict['neural_network.6.weight']
        il_bias = imitation_state_dict['neural_network.6.bias']
        
        if action_net.weight.shape == il_weight.shape:
            action_net.weight.data = il_weight.clone()
            action_net.bias.data = il_bias.clone()
            print(f"  ✓ Loaded action network weights (256→2)")
        else:
            print(f"  ⚠ Shape mismatch: IL={il_weight.shape}, PPO={action_net.weight.shape}")
            print(f"    Action network NOT loaded")
            return False
        
        print(f"✓ Successfully loaded FULL imitation model from {imitation_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load imitation weights: {e}")
        import traceback
        traceback.print_exc()
        return False


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        env = self.training_env.envs[0]
        if hasattr(env, "last_infer_ms") and self.step_count % 10 == 0:
            print(f"[train_ppo] YOLO inference {env.last_infer_ms:.1f}ms")
        return True


def make_env(frame_skip: int = 1):
    """Create and return the Geometry Dash environment."""
    env = GeometryDashEnv()
    if frame_skip > 1:
        env = FrameSkipWrapper(env, skip=frame_skip)
    return env


def train(
    total_timesteps: int = 50000,
    load_imitation: bool = True,
    imitation_path: str = "imitation/models/best_model.pth",
    save_freq: int = 10_000,
    log_dir: str = "logs/ppo_geometry_dash",
    model_dir: str = "rl/models",
    frame_skip: int = 2,
    resume_path: str = None,
):
    """
    Train PPO agent on Geometry Dash.
    
    Args:
        total_timesteps: Total training steps
        load_imitation: Whether to load imitation learning weights
        imitation_path: Path to imitation model weights
        save_freq: How often to save checkpoints (in timesteps)
        log_dir: Directory for TensorBoard logs
        model_dir: Directory to save model checkpoints
        frame_skip: Number of frames to repeat each action (1=no skip, 4=common)
        resume_path: Path to saved model to resume training from
    """
    print("=" * 60)
    print("PPO Training for Geometry Dash")
    print("=" * 60)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    print(f"\n[1/4] Creating environment (frame_skip={frame_skip})...")
    env = make_env(frame_skip=frame_skip)
    
    # Check if resuming from checkpoint
    if resume_path:
        print(f"\n[2/4] Loading saved model from {resume_path}...")
        # Add .zip extension if not present
        if not resume_path.endswith('.zip'):
            resume_path_full = resume_path + '.zip'
        else:
            resume_path_full = resume_path
            
        if os.path.exists(resume_path_full) or os.path.exists(resume_path):
            model = PPO.load(
                resume_path,
                env=env,
                tensorboard_log=log_dir,
                device="auto",
            )
            print(f"  ✓ Resumed from {resume_path}")
            print("\n[3/4] Skipping imitation loading (using checkpoint weights)")
        else:
            print(f"  ✗ Checkpoint not found: {resume_path}")
            print("  Starting fresh instead...")
            resume_path = None
    
    if not resume_path:
        # Define policy kwargs - NO extra layers so action_net is directly 256→2
        policy_kwargs = dict(
            features_extractor_class=ImitationFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[], vf=[]),  # No extra layers! Direct 256→2
        )
          
        # Create PPO model
        print("\n[2/4] Creating PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-5,  # Even lower LR for fine-tuning pretrained
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,        # Higher entropy = more exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device="auto",
        )
        
        # Load imitation weights if requested
        if load_imitation and os.path.exists(imitation_path):
            print("\n[3/4] Loading imitation learning weights...")
            success = load_imitation_weights_full(model, imitation_path)
            if not success:
                print("  ⚠ Continuing with partial/no pretrained weights")
        else:
            print("\n[3/4] Training from scratch (no imitation weights)")
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="ppo_geometry_dash",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    infer_callback = TensorboardCallback()
    
    # Training info
    print("\n[4/4] Starting training...")
    print(f"      Total timesteps: {total_timesteps:,}")
    print(f"      Frame skip: {frame_skip}x")
    print(f"      Checkpoints every: {save_freq:,} steps")
    print(f"      TensorBoard logs: {log_dir}")
    print(f"      Model saves: {model_dir}")
    print("\n" + "=" * 60)
    print("TRAINING STARTS IN 3 SECONDS - FOCUS THE GAME WINDOW!")
    print("=" * 60)
    time.sleep(3)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, infer_callback],
            progress_bar=True,
        )
        
        # Save final model
        final_path = os.path.join(model_dir, "ppo_geometry_dash_final")
        model.save(final_path)
        print(f"\n✓ Training complete! Final model saved to {final_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupt_path = os.path.join(model_dir, "ppo_geometry_dash_interrupted")
        model.save(interrupt_path)
        print(f"Model saved to {interrupt_path}")
        
    finally:
        env.close()
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO on Geometry Dash")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--no-imitation", action="store_true", help="Don't load imitation weights")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint frequency")
    parser.add_argument("--frame-skip", type=int, default=1, help="Frames to skip (1=none, 4=recommended)")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from (e.g., rl/models/ppo_geometry_dash_10000_steps)")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        load_imitation=not args.no_imitation,
        save_freq=args.save_freq,
        frame_skip=args.frame_skip,
        resume_path=args.resume,
    )
