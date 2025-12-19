import numpy as np

data = np.load('data/imitation_data.npz')

print("=" * 50)
print("EXPERT DATA SUMMARY")
print("=" * 50)

print(f"\nKeys in file: {list(data.keys())}")

actions = data['actions']
timestamps = data['timestamps']

print(f"\nActions shape: {actions.shape}")
print(f"Timestamps shape: {timestamps.shape}")

print(f"\nTotal frames captured: {len(actions)}")
print(f"Total jumps (action=1): {np.sum(actions)}")
print(f"Total no-jump (action=0): {len(actions) - np.sum(actions)}")
print(f"Jump percentage: {100 * np.sum(actions) / len(actions):.1f}%")

print("\n" + "=" * 50)
print("TIMING STATS")
print("=" * 50)
total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
avg_fps = len(actions) / total_duration if total_duration > 0 else 0

print(f"Total duration: {total_duration:.2f} seconds")
print(f"Average FPS: {avg_fps:.1f}")
print(f"First timestamp: {timestamps[0]:.3f}")
print(f"Last timestamp: {timestamps[-1]:.3f}")

# Frame timing analysis
if len(timestamps) > 1:
    frame_intervals = np.diff(timestamps)
    print(f"Average frame interval: {np.mean(frame_intervals)*1000:.1f}ms")
    print(f"Min frame interval: {np.min(frame_intervals)*1000:.1f}ms") 
    print(f"Max frame interval: {np.max(frame_intervals)*1000:.1f}ms")

print("\n" + "=" * 50)
print("SAMPLE DATA")
print("=" * 50)
print(f"\nFirst 10 actions: {actions[:10]}")
print(f"Last 10 actions: {actions[-10:]}")

print("\n" + "=" * 50)
print("ACTION SEQUENCES")
print("=" * 50)
# Find jump sequences
jump_indices = np.where(actions == 1)[0]
if len(jump_indices) > 0:
    print(f"First 10 jump frame indices: {jump_indices[:10]}")
    print(f"Total jump sequences: {len(jump_indices)}")
    
    # Find consecutive jumps
    if len(jump_indices) > 1:
        consecutive_jumps = np.diff(jump_indices) == 1
        if np.any(consecutive_jumps):
            print("Found consecutive jump frames (holding spacebar)")
        else:
            print("All jumps are single-frame (quick taps)")
else:
    print("No jumps recorded!")

# Detailed frame-by-frame view
print("\n" + "=" * 50)
print("DETAILED FRAME LOG (first 50 frames)")
print("=" * 50)
show_frames = min(50, len(actions))
for i in range(show_frames):
    action_str = "JUMP" if actions[i] == 1 else "----"
    time_since_start = timestamps[i] - timestamps[0] if i < len(timestamps) else 0
    print(f"Frame {i:4d} | {action_str} | Time: {time_since_start:.3f}s")

print(f"\n(Showing {show_frames}/{len(actions)} frames)")

print("\n" + "=" * 50)
print("FILES CREATED")
print("=" * 50)
print(f"Frames saved to: data/expert_frames/ (frame_0.jpg to frame_{len(actions)-1}.jpg)")
print(f"Actions and timestamps saved to: data/imitation_data.npz")

