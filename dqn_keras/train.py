import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from dqn_keras.dqn_agent import DQNAgentKeras
from snake_game import SnakeGame
from tensorflow.keras.callbacks import TensorBoard # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=500, help="Número de episódios para o treinamento")
parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do batch para replay")
parser.add_argument("--render", action="store_true", help="Renderizar o jogo nos últimos episódios")
args = parser.parse_args()

if __name__ == "__main__":
    env = SnakeGame(grid_size=10)
    agent = DQNAgentKeras(state_size=100, action_size=4)
    episodes = args.episodes
    batch_size = args.batch_size

    tensorboard = TensorBoard(log_dir="runs/dqn_keras")

    for episode in range(episodes):
        state = env.reset().flatten()
        total_reward = 0

        while True:
            action = agent.act(state.reshape(1, -1))
            next_state, reward, done = env.step(action)
            next_state = next_state.flatten()
            agent.remember(state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)
            state = next_state
            total_reward += reward

            if args.render and episode >= episodes - 2:
                env.render()

            if done:
                print(f"Episode {episode + 1}/{episodes}, Score: {env.score}, Total Reward: {total_reward}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    def save_model_to_json(model, filename):
        weights = [w.tolist() for w in model.get_weights()]
        with open(filename, 'w') as f:
            json.dump(weights, f)

    os.makedirs("saved_models", exist_ok=True)
    save_model_to_json(agent.model, "saved_models/dqn_keras_weights.json")
    print("Pesos do modelo DQN (Keras) salvos em saved_models/dqn_keras_weights.json")
