import torch
import torch.nn as nn

from environment.geometry_dash_env import GeometryDashEnv


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_network = nn.Sequential(
            nn.Linear(84, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.neural_network(x)


def test_imitation():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "imitation/models/best_model.pth"

    print(f"Using device: {device}")
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = GeometryDashEnv()
    state, _ = env.reset()
    stats_window = 200
    state_buffer = []

    try:
        while True:
            state_buffer.append(state)
            if len(state_buffer) == stats_window:
                states_np = torch.tensor(state_buffer, dtype=torch.float32)
                print(
                    f"Live state stats (last {stats_window}): "
                    f"min={states_np.min().item():.4f} "
                    f"max={states_np.max().item():.4f} "
                    f"mean={states_np.mean().item():.4f}"
                )
                state_buffer.clear() 

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()

            print(f"Action: {action}")
            state, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                state, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    test_imitation()
