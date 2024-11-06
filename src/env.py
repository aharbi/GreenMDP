import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GreenMDP(gym.Env):
    def __init__(
        self,
        grid_size=(5, 5),
        initial_temperature=20,
        deployment_cost=0.5,
        maintenance_cost=0.1,
        horizon=10,
    ):
        super(GreenMDP, self).__init__()

        self.grid_size = grid_size
        self.initial_temperature = initial_temperature
        self.deployment_cost = deployment_cost
        self.maintenance_cost = maintenance_cost
        self.num_cells = grid_size[0] * grid_size[1]

        self.horizon = horizon
        self.t = 0

        self.action_space = spaces.MultiBinary(self.num_cells)

        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.num_cells * 2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.temperature_grid = np.full(
            self.grid_size, self.initial_temperature, dtype=np.float32
        )
        self.green_rooftop_grid = np.zeros(self.grid_size, dtype=np.int32)

        self.t = 0

        return self._get_observation(), {}

    def step(self, action):
        action_grid = np.array(action).reshape(self.grid_size)

        deployment_cost = self._calculate_deployment_cost(action_grid)

        self.green_rooftop_grid = action_grid

        self._update_temperatures()

        reward = -np.mean(self.temperature_grid) - deployment_cost

        self.t += 1
        done = self.t >= self.horizon

        observation = self._get_observation()

        return observation, reward, done, False, {}

    def _get_observation(self):
        observation = np.concatenate(
            [self.temperature_grid.flatten(), self.green_rooftop_grid.flatten()]
        )
        return observation.astype(np.float32)

    def _update_temperatures(self):
        self.temperature_grid += 1

        new_temperature_grid = self.temperature_grid.copy()

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.green_rooftop_grid[i, j] == 1:
                    new_temperature_grid[i, j] -= 1

                if self.green_rooftop_grid[i, j] == 1:
                    neighbors = self._get_neighbors(i, j)
                    if np.random.rand() < 0.5:
                        for ni, nj in neighbors:
                            new_temperature_grid[ni, nj] -= 0.5

        self.temperature_grid = new_temperature_grid

    def _calculate_deployment_cost(self, action_grid):
        cost = 0
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if action_grid[i, j] == 1 and self.green_rooftop_grid[i, j] == 0:
                    cost += self.deployment_cost
                elif action_grid[i, j] == 1 and self.green_rooftop_grid[i, j] == 1:
                    cost += self.maintenance_cost
        return cost

    def _get_neighbors(self, i, j):

        neighbors = []
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                neighbors.append((ni, nj))

        return neighbors

    def render(self):
        print("Temperature Grid:")
        print(self.temperature_grid)
        print("Green Rooftop Grid:")
        print(self.green_rooftop_grid)


if __name__ == "__main__":
    env = GreenMDP(
        grid_size=(8, 8), initial_temperature=20, deployment_cost=2, horizon=10
    )