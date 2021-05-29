import numpy as np


class NoOpAgent:
    def act(self, obs):
        return np.array([0, 0])