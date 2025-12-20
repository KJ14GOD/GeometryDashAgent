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

    try:
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()

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
