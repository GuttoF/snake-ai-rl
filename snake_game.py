from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class SnakeGame:
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.max_steps = grid_size * grid_size * 2
        self.reset()

    def reset(self):
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.food = self._place_food()
        self.direction = (0, 1)
        self.done = False
        self.score = 0
        self.steps = 0
        self.last_distance = self._distance_to_food(self.snake[0])
        return self._get_state()

    def step(self, action: int):
        self._change_direction(action)
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        reward = -0.01  # Penalidade leve por cada movimento
        self.steps += 1

        if self._is_wall_collision(new_head):
            self.done = True
            return self._get_state(), -1, self.done

        if self._is_body_collision(new_head):
            self.done = True
            return self._get_state(), -0.5, self.done

        self.snake.appendleft(new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10  # Recompensa maior por comer a fruta
            self.food = self._place_food()
            self.steps = 0
            self.last_distance = self._distance_to_food(new_head)
        else:
            self.snake.pop()
            distance = self._distance_to_food(new_head)
            if distance < self.last_distance:
                reward += 0.1  # Recompensa por se aproximar da fruta
            else:
                reward -= 0.02  # Penalidade leve por se afastar
            self.last_distance = distance

        if self.steps > self.max_steps:
            self.done = True

        reward += 0.05  # Recompensa adicional por sobrevivência
        return self._get_state(), reward, self.done

    def _distance_to_food(self, position: tuple) -> float:
        return np.sqrt(
            (position[0] - self.food[0]) ** 2 + (position[1] - self.food[1]) ** 2
        )

    def _is_wall_collision(self, position: tuple) -> bool:
        x, y = position
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size

    def _is_body_collision(self, position: tuple) -> bool:
        return position in self.snake

    def _place_food(self) -> tuple:
        while True:
            food = (
                np.random.randint(self.grid_size),
                np.random.randint(self.grid_size),
            )
            if food not in self.snake:
                return food

    def _change_direction(self, action: int) -> None:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        proposed_direction = directions[action]
        if len(self.snake) > 1 and (
            self.snake[1][0] - self.snake[0][0],
            self.snake[1][1] - self.snake[0][1],
        ) == (-proposed_direction[0], -proposed_direction[1]):
            return
        self.direction = proposed_direction

    def _get_state(self) -> np.ndarray:
        state = np.zeros((self.grid_size, self.grid_size))
        for x, y in self.snake:
            state[x, y] = 1
        fx, fy = self.food
        state[fx, fy] = 2
        return state

    def render(self):
        grid = self._get_state()
        plt.imshow(grid, cmap="Greens", interpolation="nearest")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Score: {self.score}")
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
