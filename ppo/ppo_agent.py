import torch
import torch.nn as nn
import torch.optim as optim


class PPOAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001, gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.policy_network = self._build_model()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Softmax(dim=-1),
        )

    def act(self, state: torch.Tensor) -> int:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_network(state)
        action = int(torch.multinomial(probs, 1).item())
        return action
