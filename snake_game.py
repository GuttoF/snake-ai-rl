from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class SnakeGame:
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.food = self._place_food()
        self.direction = (0, 1)
        self.done = False
        self.score = 0
        return self._get_state()

    def step(self, action: int):
        self._change_direction(action)
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        if self._is_wall_collision(new_head):
            # Tenta evitar colisão redirecionando
            self.direction = self._safe_direction()
            new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

            # Se mesmo assim colidir, termina o jogo
            if self._is_wall_collision(new_head):
                self.done = True
                return self._get_state(), -2, self.done

        if self._is_body_collision(new_head):
            self.done = True
            return self._get_state(), -1, self.done

        self.snake.appendleft(new_head)
        reward = -0.1
        if new_head == self.food:
            self.score += 1
            reward = 1
            self.food = self._place_food()
        else:
            self.snake.pop()

        return self._get_state(), reward, self.done

    def _is_wall_collision(self, position):
        x, y = position
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size

    def _is_body_collision(self, position):
        return position in self.snake

    def _place_food(self):
        while True:
            food = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if food not in self.snake:
                return food

    def _safe_direction(self):
        """Retorna uma direção segura para evitar colisões"""
        head = self.snake[0]
        possible_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for direction in possible_directions:
            new_head = (head[0] + direction[0], head[1] + direction[1])
            if not self._is_wall_collision(new_head) and not self._is_body_collision(new_head):
                return direction
        return self.direction  # Mantém a direção atual se não encontrar alternativas

    def _change_direction(self, action):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        proposed_direction = directions[action]

        # Evita mudar para a direção oposta imediatamente
        if (
            len(self.snake) > 1
            and (self.snake[1][0] - self.snake[0][0], self.snake[1][1] - self.snake[0][1])
            == (-proposed_direction[0], -proposed_direction[1])
        ):
            return

        self.direction = proposed_direction

    def _get_state(self):
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
