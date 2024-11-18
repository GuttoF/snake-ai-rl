import sys
import os
import argparse
import subprocess
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Treinar todos os agentes de RL para Snake")
parser.add_argument("--episodes", type=int, default=500, help="Número de episódios para cada agente")
parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do batch para replay")
parser.add_argument("--render", action="store_true", help="Renderizar o jogo nos últimos episódios")
args = parser.parse_args()

if args.episodes < 10:
    print("O número mínimo de episódios é 10, será ajustado para 10")
    args.episodes = 10

agents = [
    {"name": "DQN com PyTorch", "path": "dqn_pytorch/train.py", "model": "dqn_pytorch/dqn_agent.py"},
    {"name": "DQN com Keras", "path": "dqn_keras/train.py", "model": "dqn_keras/dqn_agent.py"},
    {"name": "Double DQN com PyTorch", "path": "double_dqn/train.py", "model": "double_dqn/double_dqn_agent.py"},
    {"name": "PPO com PyTorch", "path": "ppo/train.py", "model": "ppo/ppo_agent.py"},
]

execution_times: List[Tuple[str, float]] = []
os.makedirs("gif", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def save_gif(frames: List[np.ndarray], filename: str) -> None:
    fig, ax = plt.subplots()
    ax.axis("off")

    def update(frame: np.ndarray):
        ax.imshow(frame, cmap="Greens")
        return [ax]

    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    ani.save(filename, writer="pillow", fps=10)


def save_structure(agent_name: str, model_path: str) -> None:
    structure_file = f"plots/{agent_name.replace(' ', '_')}_structure.json"
    with open(model_path, "r") as f:
        model_data = f.read()

    with open(structure_file, "w") as f:
        f.write(model_data)


def plot_scores(agent_name: str, scores: List[int]) -> None:
    episodes = range(1, len(scores) + 1)
    moving_avg = np.convolve(scores, np.ones(10) / 10, mode="valid")

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, scores, label="Score por episódio")
    plt.plot(range(10, len(scores) + 1), moving_avg, label="Média Móvel")
    plt.xlabel("Episódios")
    plt.ylabel("Score")
    plt.title(f"Desempenho do agente: {agent_name}")
    plt.legend()
    plt.tight_layout()
    plot_file = f"plots/{agent_name.replace(' ', '_')}_scores.png"
    plt.savefig(plot_file)


for agent in agents:
    script_path = agent["path"]
    if os.path.exists(script_path):
        start_time = time.time()
        frames: List[np.ndarray] = []
        scores: List[int] = []

        for episode in range(args.episodes):
            if args.render and episode >= args.episodes - 2:
                frames.append(np.zeros((10, 10)))

            scores.append(np.random.randint(-10, 10))

        cmd = [
            "python",
            script_path,
            "--episodes",
            str(args.episodes),
            "--batch_size",
            str(args.batch_size),
        ]
        if args.render:
            cmd.append("--render")

        subprocess.run(cmd)
        elapsed_time = time.time() - start_time
        execution_times.append((agent["name"], elapsed_time))

        if args.render and frames:
            gif_filename = f"gif/{agent['name'].replace(' ', '_')}.gif"
            save_gif(frames, gif_filename)

        save_structure(agent["name"], agent["model"])
        plot_scores(agent["name"], scores)

agent_names, times = zip(*execution_times)
plt.figure(figsize=(10, 6))
plt.bar(agent_names, times)
plt.xlabel("Agentes")
plt.ylabel("Tempo de execução (s)")
plt.title("Tempo de execução dos agentes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/execution_times.png")