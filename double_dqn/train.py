import argparse
import json
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter  # type: ignore

from double_dqn.double_dqn_agent import DoubleDQNAgent
from snake_game import SnakeGame

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


def save_model_to_json(model, filename):
    weights = {k: v.tolist() for k, v in model.state_dict().items()}
    with open(filename, "w") as f:
        json.dump(weights, f)


if __name__ == "__main__":
    env = SnakeGame(grid_size=10)
    agent = DoubleDQNAgent(state_size=100, action_size=4)
    episodes = args.episodes
    batch_size = args.batch_size

    writer = SummaryWriter(log_dir="runs/double_dqn")

    for episode in range(episodes):
        state = env.reset().flatten()
        total_reward = 0

        while True:
            action = agent.act(torch.tensor(state, dtype=torch.float32))
            next_state, reward, done = env.step(action)
            next_state = next_state.flatten()
            agent.remember(state.tolist(), action, reward, next_state.tolist(), done)
            state = next_state
            total_reward += reward

            if args.render and episode >= episodes - 2:
                env.render()

            if done:
                print(
                    f"Episode {episode + 1}/{episodes}, "
                    f"Score: {env.score}, Total Reward: {total_reward}"
                )
                writer.add_scalar("Score", env.score, episode)
                writer.add_scalar("Total Reward", total_reward, episode)
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    writer.close()

    os.makedirs("saved_models", exist_ok=True)
    save_model_to_json(agent.model, "saved_models/double_dqn_weights.json")
    print("Pesos do modelo Double DQN salvos em saved_models/double_dqn_weights.json")
