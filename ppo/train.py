import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter  # type: ignore

from ppo.ppo_agent import PPOAgent
from snake_game import SnakeGame

parser = argparse.ArgumentParser()
parser.add_argument(
    "--episodes", type=int, default=500, help="Número de episódios para o treinamento"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Tamanho do batch (não usado no PPO)"
)
parser.add_argument(
    "--render", action="store_true", help="Renderizar o jogo nos últimos episódios"
)
args = parser.parse_args()

if __name__ == "__main__":
    env = SnakeGame(grid_size=10)
    agent = PPOAgent(state_size=100, action_size=4)
    episodes = args.episodes

    writer = SummaryWriter(log_dir="runs/ppo")

    for episode in range(episodes):
        state = env.reset().flatten()
        total_reward = 0

        while True:
            action = agent.act(torch.from_numpy(state).float())
            next_state, reward, done = env.step(action)
            next_state = next_state.flatten()
            total_reward += reward
            state = next_state

            if args.render and episode >= episodes - 2:
                env.render()

            if done:
                print(
                    (
                        f"Episode {episode + 1}/{episodes}, Score: {env.score}, "
                        f"Total Reward: {total_reward}"
                    )
                )
                writer.add_scalar("Score", env.score, episode)
                writer.add_scalar("Total Reward", total_reward, episode)
                break

    writer.close()
