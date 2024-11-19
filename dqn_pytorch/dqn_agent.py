import random
from collections import deque

import torch
import torch.nn as nn


class DQNAgentPyTorch:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory: deque = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.device = "cpu"
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size),
        ).to(self.device)

    def remember(
        self, state: list, action: int, reward: float, next_state: list, done: bool
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: torch.Tensor) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return int(torch.argmax(q_values[0]).item())

    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for state, action, reward, next_state, done in minibatch:
            state = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            next_state = (
                torch.tensor(next_state, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).unsqueeze(0)

            target_f = self.model(state)
            predicted_q_value = target_f[0, action]

            optimizer.zero_grad()
            loss = criterion(predicted_q_value.unsqueeze(0), target)
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
