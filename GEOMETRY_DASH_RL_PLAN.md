# Geometry Dash RL Agent - Implementation Plan

## Architecture Overview

```
Screen Capture → Detection → Feature Extraction → State Vector → Policy Network → Action
     ↓              ↓              ↓                  ↓              ↓            ↓
   mss          YOLO           Features          [84-dim         PPO         Keyboard
              (Roboflow)      (egocentric)       vector]        Agent       Input
```

**Key Design Decisions:**

- Screen capture + keyboard input (Mac compatible)
- YOLO-based object detection (custom-trained on Geometry Dash)
- Egocentric feature extraction (player-relative coordinates)
- PPO for RL (better sample efficiency than DQN)
- Imitation Learning → PPO fine-tuning pipeline

---

## Phase 1: Foundation & Perception Layer ✅ COMPLETED

### 1.1 Setup & Dependencies ✅

**Files created:**
- `requirements.txt` - All dependencies
- Project structure set up

**Dependencies:**
- `mss` - Screen capture (Mac compatible)
- `opencv-python` - Computer vision
- `numpy` - Numerical operations
- `gymnasium` - RL environment framework
- `stable-baselines3` - PPO implementation
- `torch` - Neural networks
- `pynput` - Keyboard control (Mac compatible)
- `ultralytics` - YOLO v8
- `roboflow` - Dataset management

### 1.2 Screen Capture Module ✅

**File:** `perception/screen_capture.py`

**Features:**
- Use `mss` to capture game window
- Configurable monitor region (1295x810 for Stereo Madness)
- Frame rate control (~60 FPS)
- Frame preprocessing (resize, normalize optional)

**Status:** Working, captures gameplay at 60 FPS

### 1.3 Detection Module ✅

**File:** `perception/detector.py`

**Features:**
- YOLO model trained on Roboflow dataset
- Detects: player, blocks, platforms, spikes, coins, portals, spaceship
- Returns bounding boxes and class IDs
- Model path: `models/roboflow_model/weights.pt`

**Status:** Working, detects game elements accurately

### 1.4 Feature Extraction Module ✅

**File:** `perception/feature_extractor.py`

**Features:**
- Converts YOLO detections → 84-dimensional state vector
- Egocentric coordinates (player-relative positions)
- Filters obstacles ahead of player
- Normalizes all features to [0, 1] range

**State Vector (84 dimensions):**

#### Player Features (4 dims):
- `player_y_norm` - Vertical position [0, 1]
- `player_velocity_y_norm` - Vertical velocity [-1, 1]
- `on_ground` - Ground contact flag {0, 1}
- `player_detected` - Detection confidence {0, 1}

#### Obstacle Features (6 obstacles × 13 dims = 78 dims):
For each of 6 nearest obstacles ahead:
- `dx_norm` - Horizontal distance (player-relative)
- `dy_norm` - Vertical distance (player-relative)
- `distance_norm` - Euclidean distance
- `class_type_norm` - Obstacle type [0, 1]
- `can_land_on_top` - Landable flag {0, 1}
- `collision_risk` - Danger score [0, 1]
- `class_one_hot[7]` - One-hot encoding (block, coin, platform, player, portal, spaceship, spike)

#### Environment Features (2 dims):
- `ground_distance_norm` - Distance to ground
- `next_spike_distance_norm` - Distance to nearest spike ahead

**Status:** Working, extracts meaningful features

---

## Phase 2: Environment & State Management 🚧 IN PROGRESS

### 2.1 Gym Environment 🚧

**File:** `environment/geometry_dash_env.py`

**Implements:** `gymnasium.Env` interface

**Methods:**
- `__init__()` - Initialize all modules (ScreenCapture, Detector, FeatureExtractor, ActionExecutor)
- `reset()` - Restart level, return initial state
- `step(action)` - Execute action, return (state, reward, terminated, truncated, info)
- `render()` - Visualization mode (placeholder)
- `close()` - Cleanup resources

**Spaces:**
- `observation_space`: `Box(low=0, high=1, shape=(84,), dtype=np.float32)`
- `action_space`: `Discrete(2)` - 0=release, 1=jump

**Current Issues:**
- ❌ Death detection inconsistent (menu screen detection)
- ❌ Starting position not always consistent
- ❌ `terminated` flag sometimes wrong (player detected on menu screen)

**TODO:**
- [ ] Implement robust menu screen detection (text pattern or color-based)
- [ ] Fix timing in `restart_level()` for consistent starts
- [ ] Add frame-skipping (execute action for N frames)
- [ ] Implement reward shaping

### 2.2 Reward Function ⏳ TODO

**File:** `environment/rewards.py` (to be created)

**Proposed Rewards:**
- Progress reward: distance traveled (scaled by 0.1)
- Survival reward: +0.1 per step alive
- Death penalty: -100.0 on collision
- Optional: completion bonus for level finish

**TODO:**
- [ ] Implement `RewardCalculator` class
- [ ] Track player progress (X position)
- [ ] Integrate into `geometry_dash_env.py`

### 2.3 Action Executor ✅

**File:** `control/action_executor.py`

**Features:**
- Sends keyboard inputs (spacebar for jump)
- `act(1)` = press space, `act(0)` = release space
- `restart_level()` = press space to dismiss menu

**Current Issues:**
- ❌ Timing inconsistent (waiting before vs after restart)
- ❌ No state tracking (can spam press/release)

**TODO:**
- [ ] Fix timing: wait AFTER restart, not before
- [ ] Add state tracking to prevent stuck keys
- [ ] Ensure consistent starting positions

---

## Phase 3: Imitation Learning ⏳ TODO

### 3.1 Data Collection

**File:** `imitation/collect_data.py` (to be created)

**Process:**
1. Record expert gameplay (you playing manually)
2. For each frame, save:
   - Screen capture
   - State vector (from FeatureExtractor)
   - Action taken (jump/no-op)
   - Timestamp
3. Save to HDF5 or pickle format
4. Collect 10-20 successful playthroughs

**TODO:**
- [ ] Create data collection script
- [ ] Record 10+ successful runs
- [ ] Save dataset in structured format

### 3.2 Data Preprocessing

**File:** `imitation/preprocess_data.py` (to be created)

**Steps:**
1. Load raw data
2. Balance dataset (weight jump actions higher)
3. Remove ambiguous frames (if needed)
4. Split: 80% train, 10% val, 10% test
5. Data augmentation (optional)

**TODO:**
- [ ] Implement preprocessing pipeline
- [ ] Balance action distribution
- [ ] Create train/val/test splits

### 3.3 Imitation Learning Model

**File:** `imitation/imitation_model.py` (to be created)

**Architecture:**
- Input: 84-dim state vector
- Hidden layers: Dense(256) → ReLU → Dense(256) → ReLU
- Output: Dense(2) → Softmax (jump/no-op probabilities)

**Training:**
- Loss: Binary cross-entropy (weighted for jump actions)
- Optimizer: Adam (lr=1e-3)
- Metrics: Accuracy, precision, recall

**TODO:**
- [ ] Define model architecture
- [ ] Implement training loop
- [ ] Save best model weights

### 3.4 Training Script

**File:** `imitation/train_imitation.py` (to be created)

**TODO:**
- [ ] Load preprocessed data
- [ ] Train imitation model
- [ ] Evaluate on validation set
- [ ] Save trained weights

---

## Phase 4: Reinforcement Learning (PPO) ⏳ TODO

### 4.1 PPO Agent Setup

**File:** `rl/ppo_agent.py` (to be created)

**Use Stable-Baselines3 PPO**

**Custom Policy Network:**
- Feature extractor: MLP(84 → 256 → 256)
- Policy head: MLP(256 → 2) → Softmax
- Value head: MLP(256 → 1)

**Transfer Learning:**
- Initialize policy network with imitation learning weights
- Fine-tune with PPO

**TODO:**
- [ ] Define custom policy network
- [ ] Load imitation weights
- [ ] Configure PPO hyperparameters

### 4.2 Training Script

**File:** `rl/train_ppo.py` (to be created)

**Training Loop:**
1. Create `GeometryDashEnv`
2. Load pre-trained imitation model
3. Initialize PPO with pre-trained weights
4. Collect rollouts (2048 steps)
5. Update policy (10 epochs)
6. Evaluate periodically
7. Save checkpoints
8. Log to TensorBoard

**TODO:**
- [ ] Implement training script
- [ ] Configure TensorBoard logging
- [ ] Set up checkpointing

### 4.3 Hyperparameters

**PPO Configuration:**
- Learning rate: 3e-4 (or lower for fine-tuning)
- Batch size: 64
- N steps: 2048
- N epochs: 10
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Entropy coefficient: 0.01

**TODO:**
- [ ] Tune hyperparameters
- [ ] Run ablation studies

---

## Phase 5: Testing & Evaluation ⏳ TODO

### 5.1 Test Script

**File:** `test/test_agent.py` (to be created)

**Features:**
- Load trained model
- Run episodes
- Visualize gameplay
- Record video

**Metrics:**
- Survival time (frames)
- Progress (max X position reached)
- Completion rate (%)
- Action distribution

**TODO:**
- [ ] Implement test script
- [ ] Add video recording
- [ ] Compute metrics

### 5.2 Evaluation Metrics

**File:** `test/evaluate.py` (to be created)

**Metrics:**
- Average episode length
- Average reward per episode
- Completion rate
- Action distribution (jump frequency)
- Comparison: imitation vs RL performance

**TODO:**
- [ ] Implement evaluation script
- [ ] Generate comparison plots

---

## Phase 6: Optimization & Monitoring ⏳ TODO

### 6.1 Performance Profiling

**File:** `utils/profiler.py` (to be created)

**Profile:**
- Screen capture latency
- Detection latency
- Feature extraction latency
- Action execution latency
- **Target:** <80ms end-to-end

**TODO:**
- [ ] Implement profiler
- [ ] Identify bottlenecks
- [ ] Optimize slow components

### 6.2 Telemetry

**File:** `utils/telemetry.py` (to be created)

**Log:**
- FPS tracking
- Detection accuracy
- Action distribution
- Reward tracking
- Episode statistics

**TODO:**
- [ ] Implement telemetry
- [ ] Integrate with TensorBoard

### 6.3 Visualization

**File:** `utils/visualizer.py` (to be created)

**Features:**
- Draw detection boxes on frames
- Show predicted actions
- Display state features
- Save video recordings

**TODO:**
- [ ] Implement visualization tools
- [ ] Add debug overlays

---

## File Structure

```
geometry-dash/
├── requirements.txt
├── GEOMETRY_DASH_RL_PLAN.md (this file)
│
├── perception/
│   ├── screen_capture.py ✅
│   ├── detector.py ✅
│   └── feature_extractor.py ✅
│
├── environment/
│   ├── geometry_dash_env.py 🚧
│   └── rewards.py ⏳
│
├── control/
│   └── action_executor.py ✅
│
├── imitation/ ⏳
│   ├── collect_data.py
│   ├── preprocess_data.py
│   ├── imitation_model.py
│   └── train_imitation.py
│
├── rl/ ⏳
│   ├── ppo_agent.py
│   └── train_ppo.py
│
├── test/ ⏳
│   ├── test_agent.py
│   └── evaluate.py
│
├── utils/ ⏳
│   ├── profiler.py
│   ├── telemetry.py
│   └── visualizer.py
│
├── data/
│   ├── expert_playthroughs/ (imitation learning data)
│   └── models/ (saved models)
│
├── models/
│   └── roboflow_model/ ✅
│       └── weights.pt
│
└── logs/
    └── tensorboard/ (training logs)
```

**Legend:**
- ✅ Completed
- 🚧 In Progress
- ⏳ To Do

---

## Current Status Summary

### ✅ Completed (Phase 1)
- Screen capture working
- YOLO detection working
- Feature extraction working (84-dim state vector)
- Basic Gym environment structure
- Basic action executor

### 🚧 Current Work (Phase 2)
- **Debugging Gym environment:**
  - Death detection (menu screen recognition)
  - Consistent starting positions
  - Reward function integration

### ⏳ Next Steps
1. **Fix death detection** (menu screen text/color pattern)
2. **Fix timing** in `restart_level()` for consistency
3. **Implement reward function**
4. **Test environment thoroughly** with random policy
5. **Move to Phase 3:** Collect imitation learning data

---

## Learning Resources

### Reinforcement Learning Fundamentals

1. **Sutton & Barto - Reinforcement Learning: An Introduction**
   - Free online: http://incompleteideas.net/book/
   - Chapters 1-6: MDPs, value functions, policy iteration
   - Chapters 6-9: Temporal difference learning, Q-learning

2. **David Silver's RL Course (UCL)**
   - YouTube: https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
   - Lecture slides: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

### Deep RL

3. **Spinning Up in Deep RL (OpenAI)**
   - https://spinningup.openai.com/
   - PPO explanation: https://spinningup.openai.com/en/latest/algorithms/ppo.html
   - Code implementations included

4. **CS234: Reinforcement Learning (Stanford)**
   - Lecture videos: https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u
   - Covers policy gradients, actor-critic, PPO

### PPO Specifically

5. **Proximal Policy Optimization (PPO) Paper**
   - https://arxiv.org/abs/1707.06347
   - Original paper with math

6. **PPO Explained (Lilian Weng)**
   - https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
   - Clear explanation with math

### Stable-Baselines3

7. **Stable-Baselines3 Documentation**
   - https://stable-baselines3.readthedocs.io/
   - PPO usage: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

8. **Custom Environments Tutorial**
   - https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

### Gymnasium

9. **Gymnasium Documentation**
   - https://gymnasium.farama.org/
   - Environment basics: https://gymnasium.farama.org/tutorials/gymnasium_basics/

---

## Key Concepts

### State Vector (84 dimensions)

Your agent sees the world as an 84-dimensional vector:

```python
[
    # Player (4 dims)
    player_y_norm,           # Where am I vertically?
    player_velocity_y_norm,  # Am I going up or down?
    on_ground,               # Am I on the ground?
    player_detected,         # Did the detector see me?
    
    # Obstacle 1 (13 dims) - nearest ahead
    dx_norm, dy_norm, distance_norm,  # Where is it relative to me?
    class_type_norm,                   # What type is it?
    can_land_on_top,                   # Can I jump on it?
    collision_risk,                    # How dangerous is it?
    [0,0,0,0,0,0,1],                  # One-hot: it's a spike!
    
    # Obstacle 2 (13 dims) - 2nd nearest
    ...
    
    # Obstacle 3-6 (13 dims each)
    ...
    
    # Environment (2 dims)
    ground_distance_norm,         # How far to ground?
    next_spike_distance_norm,     # How far to next spike?
]
```

### MDP Formulation

- **State** (s): 84-dim state vector
- **Action** (a): {0=release, 1=jump}
- **Reward** (r): +0.1 survival, +0.1*dx progress, -100 death
- **Transition** (P): Game physics (deterministic)
- **Policy** (π): Neural network (state → action probabilities)

### PPO Math (High-Level)

**Objective:** Maximize expected cumulative reward

```
J(θ) = E[Σ γ^t * r_t]
```

**Policy Gradient:**
```
∇J(θ) = E[∇log π_θ(a|s) * A(s,a)]
```

**PPO Clipped Objective:**
```
L^CLIP(θ) = E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]
```

where `r(θ) = π_θ(a|s) / π_θ_old(a|s)` (policy ratio)

**Intuition:** Update policy to increase probability of good actions, but don't change too much at once (clip prevents big jumps).

---

## Debugging Tips

### Environment Issues

**Player not detected:**
- Check YOLO model is loading correctly
- Verify monitor region captures game window
- Check lighting/graphics settings

**Inconsistent starting positions:**
- Fix timing in `restart_level()` (wait AFTER restart, not before)
- Increase wait time if game loads slowly
- Use retry logic in `reset()`

**Wrong death detection:**
- Implement menu screen detection (text/color patterns)
- Require multiple consecutive frames before declaring death
- Track player X position for teleport detection

### Training Issues

**Agent not learning:**
- Check reward function (is it informative?)
- Verify state normalization (all values in [0, 1])
- Tune hyperparameters (learning rate, entropy coefficient)
- Try different random seeds

**Agent too aggressive/conservative:**
- Adjust survival vs progress reward balance
- Tune entropy coefficient (higher = more exploration)
- Check action distribution in logs

---

## Success Metrics

### Phase 2 (Environment) Goals
- [ ] Environment passes `check_env()` validation
- [ ] Random policy survives >5 frames on average
- [ ] Death detection accuracy >95%
- [ ] Consistent starting positions (X position variance <10 pixels)

### Phase 3 (Imitation) Goals
- [ ] Collect 10+ successful playthroughs (0-100%)
- [ ] Imitation model accuracy >90% on validation set
- [ ] Imitation policy survives >50 frames on average

### Phase 4 (RL) Goals
- [ ] PPO agent survives >100 frames consistently
- [ ] PPO agent reaches >20% progress
- [ ] PPO agent completes Stereo Madness (stretch goal)

### Phase 5 (Optimization) Goals
- [ ] End-to-end latency <80ms
- [ ] Training time <8 hours for convergence
- [ ] Model size <50MB

---

## Timeline Estimate

- **Week 1:** Complete Phase 2 (fix environment issues)
- **Week 2:** Phase 3 (imitation learning data collection + training)
- **Week 3:** Phase 4 (PPO setup + initial training)
- **Week 4+:** Phase 4 continued (training, debugging, tuning)
- **Week 6+:** Phase 5 (testing, optimization, evaluation)

**Total:** 6-8 weeks to a trained agent

---

## Notes

- Start with Stereo Madness (easiest level)
- Use practice mode for consistent restarts (or implement robust menu detection)
- Collect at least 10 expert playthroughs for imitation learning
- Use transfer learning: initialize RL with imitation weights
- Monitor TensorBoard during training
- Iterate on reward function based on agent behavior
- Profile and optimize bottlenecks
- Save checkpoints frequently

---

## Contact / Questions

If stuck on a specific phase, refer to:
- Gymnasium docs for environment issues
- Stable-Baselines3 docs for PPO issues
- YOLO docs for detection issues
- This plan for overall structure

Good luck! 🚀

