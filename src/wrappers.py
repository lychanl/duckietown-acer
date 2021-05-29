import gym
from gym.spaces import Box
from copy import copy


class RescaleObsToFloatWrapper(gym.ObservationWrapper):
    def __init__(self, env, x0=0, scale=1):
        super().__init__(env)
        self.x0 = x0
        self.scale = scale
        self.observation_space = Box(
            (self.env.observation_space.low - x0) * scale,
            (self.env.observation_space.high - x0) * scale,
            self.env.observation_space.shape
        )

    def observation(self, observation):
        return (observation - self.x0) * self.scale
