import gym
from gym_duckietown.envs.multimap_env import MultiMapEnv
import numpy as np
from wrappers import RescaleObsToFloatWrapper, DirectionChangePenaltyWrapper

def override_cnn(filters: list, kernels: list, strides: list):
    import models.cnn as cnn

    build = cnn.build_cnn_network
    def build_cnn_network():
        return build(
            filters=filters,
            kernels=kernels,
            strides=strides
        )
    cnn.build_cnn_network = build_cnn_network


def wrappers(
        no_grayscale: bool, obs_scale: int, center: bool, time_limit: int = None,
        reward_scale: int = None, eval: bool = False, dir_change_penalty: float = None
):
    return [
        (gym.wrappers.ResizeObservation, {'shape': (480 // obs_scale, 640 // obs_scale)})
    ] + ([] if no_grayscale else [(gym.wrappers.GrayScaleObservation, {})]) + [
        (RescaleObsToFloatWrapper, {'scale': 1/128, 'x0': 128} if center else {'scale': 1/256})
    ] + ([] if eval else ([
        (gym.wrappers.TimeLimit, {'max_episode_steps': time_limit}),
        (gym.wrappers.TransformReward, {'f': lambda x: x / reward_scale})
    ] + [] if dir_change_penalty else [
        (DirectionChangePenaltyWrapper, {'penalty': dir_change_penalty})
    ]))


def get_dir_vec(angle):
    x = np.cos(angle)
    z = -np.sin(angle)
    return np.array([x, 0, z])


def get_closest_heading(info, env):
    env = env.unwrapped.env_list[env.unwrapped.cur_env_idx] if isinstance(MultiMapEnv, env) else env.unwrapped

    i, j = env.get_grid_coords(info['Simulator']['cur_pos'])
    tile = env._get_tile(i, j)
    curves = tile['curves']
    curve_headings = curves[:, -1, :] - curves[:, 0, :]

    curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
    dir_vec = get_dir_vec(info['Simulator']['cur_angle'])

    dot_prods = np.dot(curve_headings, dir_vec)

    # Closest curve = one with largest dotprod
    closest = np.argmax(dot_prods)
    return curve_headings[closest]