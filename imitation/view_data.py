import numpy as np

data = np.load('data/expert_data.npz')

print("=" * 50)
print("EXPERT DATA SUMMARY")
print("=" * 50)

print(f"\nKeys in file: {list(data.keys())}")

states = data['state_array']
actions = data['action_array']

print(f"\nStates shape: {states.shape}")
print(f"Actions shape: {actions.shape}")

print(f"\nTotal frames: {len(actions)}")
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
print(f"\nFirst 5 actions: {actions[:5]}")
print(f"Last 5 actions: {actions[-5:]}")
print(f"\nFirst state vector (first 10 dims):\n{states[0][:10]}")

print("\n" + "=" * 50)
print("ACTION SEQUENCES")
print("=" * 50)
# Find jump sequences
jump_indices = np.where(actions == 1)[0]
if len(jump_indices) > 0:
    print(f"First 10 jump frame indices: {jump_indices[:10]}")
else:
    print("No jumps recorded!")

