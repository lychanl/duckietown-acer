import gym
from gym_duckietown.envs.multimap_env import MultiMapEnv
import numpy as np


def wrappers(
        no_grayscale: bool, obs_scale: int, center: bool, time_limit: int = None,
        reward_scale: int = None, reward_clip: float = None, eval: bool = False, dir_change_penalty: float = None,
        data_model_path: str = None, data_info: bool = False,
):
    from wrappers import RescaleObsToFloatWrapper, DirectionChangePenaltyWrapper
    from dataset_regression import load_or_build_data_model, DataModelWrapper, DataInfoWrapper

    wraps = [
        (gym.wrappers.ResizeObservation, {'shape': (480 // obs_scale, 640 // obs_scale)})
    ] 
    if not no_grayscale:
        wraps.append((gym.wrappers.GrayScaleObservation, {}))

    wraps.append((RescaleObsToFloatWrapper, {'scale': 1/128, 'x0': 128} if center else {'scale': 1/256}))

    if not eval:
        wraps.append((gym.wrappers.TimeLimit, {'max_episode_steps': time_limit}))
        wraps.append((
            gym.wrappers.TransformReward,
            {'f': lambda x: np.clip(x / reward_scale, -reward_clip, reward_clip)} if reward_clip else {'f': lambda x: x / reward_scale}))
        if dir_change_penalty:
            wraps.append((DirectionChangePenaltyWrapper, {'penalty': dir_change_penalty}))
        if data_info:
            wraps.append((DataInfoWrapper, {}))
    elif data_model_path:
        data_model = load_or_build_data_model(
            data_model_path, input_shape=(480 // obs_scale, 640 // obs_scale, 3 if no_grayscale else 1)
        )
        wraps.append((DataModelWrapper, {'model': data_model}))

    return wraps


def get_dir_vec(angle):
    x = np.cos(angle)
    z = -np.sin(angle)
    return np.array([x, 0, z])


def get_closest_heading(info, env):
    env = env.unwrapped.env_list[env.unwrapped.cur_env_idx] if isinstance(env, MultiMapEnv) else env.unwrapped

    i, j = env.get_grid_coords(info['Simulator']['cur_pos'])
    tile = env._get_tile(i, j)
    
    if tile is None or not tile['drivable']:
        return None

    curves = tile['curves']
    curve_headings = curves[:, -1, :] - curves[:, 0, :]

    curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
    dir_vec = get_dir_vec(info['Simulator']['cur_angle'])

    dot_prods = np.dot(curve_headings, dir_vec)

    # Closest curve = one with largest dotprod
    closest = np.argmax(dot_prods)
    return curve_headings[closest]