import numpy as np

data = np.load('data/imitation_state_and_action_data.npz')

print("=" * 50)
print("EXPERT DATA SUMMARY")
print("=" * 50)

print(f"\nKeys in file: {list(data.keys())}")

states = data['state_array']
actions = data['action_array']

print(f"\nStates shape: {states.shape}")
print(f"Actions shape: {actions.shape}")

print(f"\nTotal frames captured: {len(actions)}")
print(f"Total jumps (action=1): {np.sum(actions)}")
print(f"Total no-jump (action=0): {len(actions) - np.sum(actions)}")
print(f"Jump percentage: {100 * np.sum(actions) / len(actions):.1f}%")

print("\n" + "=" * 50)
print("STATE VECTOR STATS")
print("=" * 50)
print(f"Min value: {states.min():.4f}")
print(f"Max value: {states.max():.4f}")
print(f"Mean value: {states.mean():.4f}")

print("\n" + "=" * 50)
print("SAMPLE DATA")
print("=" * 50)
print(f"\nFirst 10 actions: {actions[:10]}")
print(f"Last 10 actions: {actions[-10:]}")
print(f"\nFirst state vector (first 10 dims):\n{states[0][:10]}")

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
show_frames = min(100, len(actions))
for i in range(show_frames):
    action_str = "JUMP" if actions[i] == 1 else "----"
    state_preview = states[i][:6]
    print(f"Frame {i:4d} | {action_str} | player_y={state_preview[0]:.2f} vel_y={state_preview[1]:.2f} on_ground={state_preview[2]:.0f} | spike_dist={state_preview[4]:.2f}")

print(f"\n(Showing {show_frames}/{len(actions)} frames)")

print("\n" + "=" * 50)
print("FILES CREATED")
print("=" * 50)
print("States and actions saved to: data/imitation_state_and_action_data.npz")
