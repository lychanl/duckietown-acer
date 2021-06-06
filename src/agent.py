import numpy as np
import gym
import json
import os
import tools
from pyglet.window import key


class NoOpAgent:
    def __init__(self, **kwargs) -> None:
        pass

    def act(self, obs: np.array) -> np.array:
        return np.array([0, 0])

    def wrap(self, env: gym.Env, eval: bool) -> gym.Env:
        return env

    def prepare(self, env: gym.Env) -> None:
        pass

    def wait(self, env: gym.Env):
        return 0


class ActorAgent:
    def __init__(self, path: str = None, actor: str = None, wrappers_params: dict = None, **kwargs) -> None:
        assert actor or path

        if path:
            self.actor = None
            self.actor_args, self.wrappers_params = self._load_config(path)
        elif actor:
            self.actor = actor
            self.wrappers_params = wrappers_params

    def _load_config(self, path: str):
        with open(os.path.join(path, 'params'), 'r') as config_file:
            config = json.load(config_file)

        return (config, os.path.join(path, 'weights')), config['wrappers_params']

    def prepare(self, env: gym.Env) -> None:
        if not self.actor:
            self.actor = self._build_actor(env, *self.actor_args)

    def _build_actor(self, env: gym.Env, config: dict, weights: str):
        from algos.base import GaussianActor
        from tensorflow import Variable, int64
        actor = GaussianActor(env.observation_space, env.action_space, *config['args'], Variable(0, dtype=int64))
        actor.load_weights(weights)

        return actor

    def act(self, obs: np.array) -> np.array:
        return self.actor.act_deterministic(np.expand_dims(obs, 0)).numpy()[0]

    def wrap(self, env: gym.Env, eval: bool) -> gym.Env:
        for wrapper, params in tools.wrappers(**self.wrappers_params, eval=eval):
            env = wrapper(env, **params)
        return env

    def wait(self, env: gym.Env):
        return 0


class ManualAgent:
    def __init__(self, **kwargs) -> None:
        self.key_handler = key.KeyStateHandler()

    def wrap(self, env: gym.Env, eval: bool) -> gym.Env:
        return env

    def act(self, obs: np.array) -> np.array:
        action = np.array([0., 0.])
        if self.key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if self.key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if self.key_handler[key.LEFT]:
            action += np.array([0, +1])
        if self.key_handler[key.RIGHT]:
            action += np.array([0, -1])
        if self.key_handler[key.LSHIFT]:
            action *= 1.5

        return action

    def prepare(self, env: gym.Env) -> None:
        env.unwrapped.window.push_handlers(self.key_handler)

    def wait(self, env: gym.Env):
        return 1 / env.unwrapped.frame_rate


AGENTS = {
    'noop': NoOpAgent,
    'actor': ActorAgent,
    'manual': ManualAgent
}
