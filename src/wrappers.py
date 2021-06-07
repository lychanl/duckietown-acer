import gym
from gym.spaces import Box
from gym_duckietown.envs.multimap_env import MultiMapEnv
import numpy as np

import tools


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


class DirectionChangePenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, penalty: float) -> None:
        super().__init__(env)
        if isinstance(env.unwrapped, MultiMapEnv):
            for env in env.unwrapped.env_list:
                env.full_transparency = True
                self.multimap = True
        else:
            env.unwrapped.full_transparency = True
            self.multimap = False

        self.penalty = penalty
        self.previous_heading = None

    def reset(self):
        self.previous_heading = None
        return super().reset()

    def step(self, action) -> tuple:
        try:
            obs, reward, done, info = super().step(action)
            
            closest_heading = tools.get_closest_heading(info, self.env)

            if self.previous_heading is not None:
                if np.dot(closest_heading, self.previous_heading) < 0:
                    reward -= self.penalty

            return obs, reward, done, info
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
