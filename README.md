# Geometry Dash RL Agent

A reinforcement learning agent that learns to play Geometry Dash using computer vision and deep learning. The agent uses real-time screen capture, YOLO-based object detection, and Proximal Policy Optimization (PPO) to learn optimal jumping strategies.

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Excluded Files](#excluded-files)
- [Training Your Own Model](#training-your-own-model)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project creates an AI agent that learns to play and beat Geometry Dash by:

1. Capturing the game screen in real-time using Python MSS Library. 
2. Detecting game objects (player, spikes, blocks, platforms, etc.) using a custom-trained YOLO model 
3. Converting detections into a normalized state vector to feed into my neural network
4. Using imitation learning and reinforcement learning (PPO) to learn when to jump

The agent observes the game through an 84-dimensional state vector that encodes player position, velocity, and information about the 6 nearest obstacles ahead.

## Demo

This project uses imitation learning to bootstrap the policy with human actions, then reinforcement learning (PPO) to refine timing and recovery. The biggest data issue was skipped frames during preprocessing; decoupling keyboard capture from YOLO inference fixed that and brought capture up to ~42 FPS with consistent action labels.

### 1) YOLO perception

![YOLO detection](assets/yolo.png)

### 2) Gymnasium environment rewards

![Alive reward](assets/env_alive_reward.png)
![Death reward](assets/env_death_reward.png)

### 3) Imitation capture (actions + frames)

This capture runs without YOLO for speed, then YOLO runs offline to build the state vectors for imitation learning.

```
==================================================
EXPERT DATA SUMMARY
==================================================

Keys in file: ['actions', 'timestamps']

Actions shape: (9538,)
Timestamps shape: (9538,)

Total frames captured: 9538
Total jumps (action=1): 2937
Total no-jump (action=0): 6601
Jump percentage: 30.8%

==================================================
TIMING STATS
==================================================
Total duration: 224.26 seconds
Average FPS: 42.5
First timestamp: 1766114201.761
Last timestamp: 1766114426.017
Average frame interval: 23.5ms
Min frame interval: 19.8ms
Max frame interval: 140.1ms

==================================================
SAMPLE DATA
==================================================

First 10 actions: [0 0 0 0 0 0 0 0 0 0]
Last 10 actions: [0 0 0 0 0 0 0 0 0 0]

==================================================
ACTION SEQUENCES
==================================================
First 10 jump frame indices: [25 26 27 28 29 30 31 32 33 34]
Total jump sequences: 2937
Found consecutive jump frames (holding spacebar)

==================================================
DETAILED FRAME LOG (first 50 frames)
==================================================
Frame    0 | ---- | Time: 0.000s
Frame    1 | ---- | Time: 0.112s
Frame    2 | ---- | Time: 0.137s
Frame    3 | ---- | Time: 0.160s
Frame    4 | ---- | Time: 0.183s
Frame    5 | ---- | Time: 0.207s
Frame    6 | ---- | Time: 0.231s
Frame    7 | ---- | Time: 0.253s
Frame    8 | ---- | Time: 0.277s
Frame    9 | ---- | Time: 0.301s
Frame   10 | ---- | Time: 0.324s
Frame   11 | ---- | Time: 0.348s
Frame   12 | ---- | Time: 0.372s
Frame   13 | ---- | Time: 0.395s
Frame   14 | ---- | Time: 0.418s
Frame   15 | ---- | Time: 0.441s
Frame   16 | ---- | Time: 0.465s
Frame   17 | ---- | Time: 0.488s
Frame   18 | ---- | Time: 0.510s
Frame   19 | ---- | Time: 0.533s
Frame   20 | ---- | Time: 0.556s
Frame   21 | ---- | Time: 0.580s
Frame   22 | ---- | Time: 0.603s
Frame   23 | ---- | Time: 0.625s
Frame   24 | ---- | Time: 0.647s
Frame   25 | JUMP | Time: 0.670s
Frame   26 | JUMP | Time: 0.692s
Frame   27 | JUMP | Time: 0.714s
Frame   28 | JUMP | Time: 0.737s
Frame   29 | JUMP | Time: 0.759s
Frame   30 | JUMP | Time: 0.783s
Frame   31 | JUMP | Time: 0.806s
Frame   32 | JUMP | Time: 0.828s
Frame   33 | JUMP | Time: 0.851s
Frame   34 | JUMP | Time: 0.875s
Frame   35 | JUMP | Time: 0.898s
Frame   36 | JUMP | Time: 0.921s
Frame   37 | JUMP | Time: 0.944s
Frame   38 | JUMP | Time: 0.968s
Frame   39 | JUMP | Time: 0.991s
Frame   40 | JUMP | Time: 1.015s
Frame   41 | JUMP | Time: 1.038s
Frame   42 | JUMP | Time: 1.061s
Frame   43 | JUMP | Time: 1.084s
Frame   44 | JUMP | Time: 1.108s
Frame   45 | JUMP | Time: 1.132s
Frame   46 | JUMP | Time: 1.155s
Frame   47 | JUMP | Time: 1.179s
Frame   48 | JUMP | Time: 1.201s
Frame   49 | JUMP | Time: 1.223s

(Showing 50/9538 frames)

==================================================
FILES CREATED
==================================================
Frames saved to: data/expert_frames/ (frame_0.jpg to frame_9537.jpg)
Actions and timestamps saved to: data/imitation_data.npz
```

## How It Works

```
Screen Capture -> YOLO Detection -> Feature Extraction -> State Vector -> Policy Network -> Action
      |                 |                    |                  |               |             |
     mss          Custom YOLO            State Vector         [84-dim          PPO         Keyboard
                    (custom)             coordinates           vector]        Agent          Input
```

### Pipeline Components

1. **Screen Capture** (`perception/screen_capture.py`): Uses `mss` to capture the game window at 60 FPS 
2. **Object Detection** (`perception/detector.py`): Detecting game objects like spike, traps, etc 
3. **Custom YOLO Model** (`perception/custom_yolo.py`) Custom YOLO Model trained on 500 in game geometry dash labeled images
4. **Feature Extraction** (`perception/feature_extractor.py`): Converts detections to player-relative coordinates
5. **Environment** (`environment/geometry_dash_env.py`): Gymnasium-compatible RL environment using step and reset() functions
6. **Action Execution** (`control/action_executor.py`): Sends keyboard inputs to the game


### State Vector (84 dimensions)

The agent sees the game as an 84-dimensional normalized vector:

- **Player Features (4 dims)**: Y position, Y velocity, ground contact, detection confidence
- **Obstacle Features (6 obstacles x 13 dims = 78 dims)**: For each of the 6 nearest obstacles ahead:
  - Relative position (dx, dy, distance)
  - Object type (normalized + one-hot encoding)
  - Landability flag and collision risk score
- **Environment Features (2 dims)**: Ground distance, distance to next spike

### Detected Object Classes

The YOLO model recognizes 7 classes:
- `player` - The cube/icon you control
- `spike` - Triangular hazards (instant death)
- `block` - Solid blocks (can land on top)
- `platform` - Platforms (can land on top)
- `coin` - Collectible coins
- `portal` - Mode-changing portals
- `spaceship` - Ship mode elements

## Project Structure

```
geometry-dash/
|-- requirements.txt          # Python dependencies
|-- README.md                  # This file
|
|-- perception/                # Computer vision modules
|   |-- screen_capture.py     # Screen capture at 60 FPS
|   |-- detector.py           # YOLO-based object detection
|   |-- feature_extractor.py  # State vector extraction
|
|-- environment/               # RL environment
|   |-- geometry_dash_env.py  # Gymnasium environment wrapper
|
|-- control/                   # Game control
|   |-- action_executor.py    # Keyboard input execution
|
|-- image_collection/          # Data collection tools
|   |-- collect_training_images.py  # Captures frames for labeling
|
|-- data/                      # Training data for YOLO
|   |-- train/                # Training images and labels
|   |-- valid/                # Validation images and labels
|   |-- test/                 # Test images and labels
|   |-- images/               # Reference images (menu, restart screens)
|   |-- data.yaml             # YOLO dataset configuration
|   |-- expert_frames/        # Captured gameplay frames (local, ignored)
|   |-- imitation_data.npz    # Actions + timestamps (local, ignored)
|   |-- expert_data.npz       # State/action dataset (local, ignored)
|
|-- runs/                      # YOLO training outputs
|   |-- geometry_dash_detector_v3/
|       |-- args.yaml         # Training configuration
|       |-- results.csv       # Training metrics
|       |-- weights/          # Model weights (excluded from repo)
```

## Requirements

- Python 3.10+
- macOS (tested on macOS 14+) or Linux
- Geometry Dash running in a window
- Screen recording permissions for your terminal/IDE

## Local Data (Generated)

These files are created locally and ignored by git:

- `data/expert_frames/`: Raw frames captured during gameplay.
- `data/imitation_data.npz`: Per-frame action labels + timestamps.
- `data/expert_data.npz`: Final IL dataset (state vectors + actions).

### How to Generate

1. Capture actions + frames (no YOLO during play):
```bash
python imitation/action_data.py
```

2. Process frames offline into state vectors:
```bash
python imitation/state_data.py
```

Tip: Trim the initial black screen and end-of-level screen during offline processing to avoid skewing the dataset.

### System Dependencies

On macOS, you need to grant screen recording permissions:
1. Go to System Preferences > Privacy & Security > Screen Recording
2. Enable permissions for Terminal or your IDE (VS Code, Cursor, etc.)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/GeometryDashAgent.git
cd GeometryDashAgent
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download or train the YOLO model weights (see [Excluded Files](#excluded-files))

## Excluded Files

The following files are excluded from the repository due to size constraints. You need to obtain or create them yourself:

### Model Weights (Required)

These `.pt` files contain the trained neural network weights:

| File | Size | Description | How to Obtain |
|------|------|-------------|---------------|
| `models/roboflow_model/weights.pt` | ~109 MB | Main YOLO detection model | Train yourself (see below) or contact maintainer |
| `runs/geometry_dash_detector_v3/weights/best.pt` | ~54 MB | Best training checkpoint | Generated during YOLO training |
| `runs/geometry_dash_detector_v3/weights/last.pt` | ~54 MB | Final training checkpoint | Generated during YOLO training |
| `perception/yolo11n.pt` | ~6 MB | Base YOLO11 nano model | Download from [Ultralytics](https://docs.ultralytics.com/models/yolo11/) |
| `perception/yolo11s.pt` | ~19 MB | Base YOLO11 small model | Download from [Ultralytics](https://docs.ultralytics.com/models/yolo11/) |

### Video Files (Optional)

| File | Description |
|------|-------------|
| `*.mp4` | Recorded gameplay and detection visualizations |

### Local Data Captures (Generated)

| Path | Description |
|------|-------------|
| `data/expert_frames/` | Raw gameplay frames captured locally |
| `data/imitation_data.npz` | Actions + timestamps from keyboard capture |
| `data/expert_data.npz` | Processed state/action dataset for IL |

### To Download Base YOLO Weights

```bash
# The ultralytics package will auto-download these when first used
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

## Training Your Own Model

If you need to train your own YOLO detection model:

### 1. Collect Training Images

```bash
# Make sure Geometry Dash is running and visible
python image_collection/collect_training_images.py
```

This captures screenshots every 2 seconds. Play through the level to capture diverse scenarios.

### 2. Label Images

Use [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg) to annotate your images with bounding boxes for each class:
- player, spike, block, platform, coin, portal, spaceship

### 3. Organize Dataset

Structure your labeled data:

```
data/
|-- train/
|   |-- images/
|   |-- labels/
|-- valid/
|   |-- images/
|   |-- labels/
|-- test/
|   |-- images/
|   |-- labels/
|-- data.yaml
```

### 4. Train YOLO Model

```bash
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # Start from pretrained weights
model.train(
    data='data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='geometry_dash_detector'
)
```

### 5. Copy Trained Weights

```bash
mkdir -p models/roboflow_model
cp runs/detect/geometry_dash_detector/weights/best.pt models/roboflow_model/weights.pt
```

## Usage

### Test the Environment

Run a quick test with random actions:

```bash
python environment/geometry_dash_env.py
```

This will:
1. Register the Gymnasium environment
2. Validate it passes `check_env()`
3. Run 50 steps with random actions
4. Save a video to `env_random_policy.mp4`

Make sure Geometry Dash is:
- Running and visible on screen
- Positioned at the expected screen coordinates (see [Configuration](#configuration))
- On the Stereo Madness level (or adjust detection thresholds accordingly)

### Test Screen Capture and Detection

```bash
python perception/screen_capture.py
```

This runs real-time detection and saves a video showing bounding boxes around detected objects.

### Collect New Training Data

```bash
python image_collection/collect_training_images.py
```

Captures frames every 2 seconds while you play. Press Ctrl+C to stop.

## Configuration

### Screen Region

The default monitor region is configured for a specific window position:

```python
monitor = {"top": 70, "left": 85, "width": 1295, "height": 810}
```

To find your game window coordinates:
1. Position Geometry Dash on your screen
2. Take a screenshot and note the game window boundaries
3. Update the `monitor` dictionary in:
   - `perception/screen_capture.py`
   - `environment/geometry_dash_env.py`
   - `image_collection/collect_training_images.py`

### Death Detection

The environment uses template matching to detect the death/menu screen:

```python
# Templates located in data/images/
menu_template = cv.imread("data/images/menu.png", 0)
restart_template = cv.imread("data/images/restart.png", 0)
```

If death detection is unreliable, you may need to recapture these templates from your game.

### Reward Function

The current reward structure (in `geometry_dash_env.py`):

- **Survival**: +1.0 per step alive
- **Bonus**: +50 every 10 seconds survived (increasing up to +100)
- **Death**: -100.0 penalty

## Troubleshooting

### "Player not detected"

- Ensure the game window is at the correct screen position
- Check that the YOLO model weights are loaded correctly
- Verify screen recording permissions are granted
- Try adjusting the monitor region coordinates

### "No module named 'perception'"

Run scripts from the project root directory:

```bash
cd /path/to/geometry-dash
python environment/geometry_dash_env.py
```

Or add the project to your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/geometry-dash"
```

### Template matching fails (death detection)

Recapture the menu and restart button templates:

1. Die in-game to show the menu
2. Take a screenshot
3. Crop the "Menu" and "Restart" buttons
4. Save as `data/images/menu.png` and `data/images/restart.png`

### Low FPS / Slow detection

- Use the YOLO nano model (`yolo11n.pt`) instead of larger variants
- Reduce the capture resolution
- Close other GPU-intensive applications

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) for the RL environment framework
- [Roboflow](https://roboflow.com) for dataset management tools
