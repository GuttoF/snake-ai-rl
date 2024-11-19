import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch

from torch.utils.tensorboard import SummaryWriter  # type: ignore

from dqn_pytorch.dqn_agent import DQNAgentPyTorch
from snake_game import SnakeGame

# Argumentos de linha de comando
parser = argparse.ArgumentParser()
parser.add_argument(
    "--episodes", type=int, default=500, help="Número de episódios para o treinamento"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Tamanho do batch para replay"
)
parser.add_argument(
    "--render", action="store_true", help="Renderizar o jogo nos últimos episódios"
)
args = parser.parse_args()

if __name__ == "__main__":
    env = SnakeGame(grid_size=10)
    agent = DQNAgentPyTorch(state_size=100, action_size=4)
    episodes = args.episodes
    batch_size = args.batch_size

    writer = SummaryWriter(log_dir="runs/dqn_pytorch")

    for episode in range(episodes):
        state = env.reset().flatten()
        total_reward = 0

        while True:
            action = agent.act(torch.from_numpy(state).float())
            next_state, reward, done = env.step(action)
            next_state = next_state.flatten()
            agent.remember(state.tolist(), action, reward, next_state.tolist(), done)
            state = next_state
            total_reward += reward

            if args.render and episode >= episodes - 2:
                env.render()

            if done:
                print(
                    f"Episode {episode + 1}/{episodes}, Score: {env.score}, "
                    f"Total Reward: {total_reward}"
                )
                writer.add_scalar("Score", env.score, episode)
                writer.add_scalar("Total Reward", total_reward, episode)
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    writer.close()

    def save_model_to_json(model, filename):
        weights = {k: v.tolist() for k, v in model.state_dict().items()}
        with open(filename, "w") as f:
            json.dump(weights, f)

    os.makedirs("saved_models", exist_ok=True)
    save_model_to_json(agent.model, "saved_models/dqn_pytorch_weights.json")
    print(
        "Pesos do modelo DQN (PyTorch) salvos em saved_models/dqn_pytorch_weights.json"
    )
