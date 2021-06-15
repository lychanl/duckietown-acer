import argparse
from ast import parse
import gym
import gym_duckietown
from gym_duckietown import simulator, graphics
import numpy as np


ANGLE_THRESHOLD = 0.4


def get_obj_info(obj, info):
    cp = info['Simulator']['cur_pos']
    dist = np.sqrt(np.square(obj.pos - cp).sum())
    
    # dir_vec = [np.cos(info['Simulator']['cur_angle']), -np.sin(info['Simulator']['cur_angle'])]
    # obj_vec = (info['Simulator']['cur_pos'] - obj.pos)[[0, 2]]
    
    obj_angle = np.arctan2(
        -(obj.pos[2] - cp[2]),
        (obj.pos[0] - cp[0])
    )
    angle = obj_angle - info['Simulator']['cur_angle']
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi

    return obj.kind, dist, angle


def get_objects_info(env, info):
    if hasattr(env.unwrapped, 'env_list'):
        env = env.unwrapped.env_list[env.unwrapped.cur_env_idx]

    obj_infos = [get_obj_info(obj, info) for obj in env.objects]
    filtered = [i for i in obj_infos if -ANGLE_THRESHOLD < i[2] < ANGLE_THRESHOLD]

    if not filtered:
        return np.array([0., 0., 0.])

    closest = np.argmin(np.array(filtered)[:,1])

    kind = 1 if filtered[closest][0] == 'duckie' else 2
    return (kind, *filtered[closest][1:])

def get_drive_curve_info(env):
    if hasattr(env.unwrapped, 'env_list'):
        env = env.unwrapped.env_list[env.unwrapped.cur_env_idx]
    env = env.unwrapped

    i, j = env.get_grid_coords(env.cur_pos)
    tile = env._get_tile(i, j)
    if tile['drivable']:
        curves = env._get_tile(i, j)['curves']
        curve_headings = curves[:, -1, :] - curves[:, 0, :]
        curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
        dirVec = simulator.get_dir_vec(env.cur_angle)
        dot_prods = np.dot(curve_headings, dirVec)
        pts = curves[np.argmax(dot_prods)]

        closest = graphics.bezier_closest(pts, env.cur_pos)
        tangent = graphics.bezier_tangent(pts, closest)
        angle = np.math.atan2(tangent[0], -tangent[2])
        d = min(1 - closest, 0.1)
        if d < 0.001:
            return 0
        nxt = closest + d
        nxt_tangent = graphics.bezier_tangent(pts, nxt)
        nxt_angle = np.math.atan2(nxt_tangent[0], -nxt_tangent[2])

        diff = nxt_angle - angle
        if diff < -np.pi:
            diff += 2* np.pi
        elif diff > np.pi:
            diff -= 2 * np.pi
        
        return diff / d
    
    return 0

def make_dataset(env_name: str, obs_scale: int, size: int):
    env = gym.make(env_name, full_transparency=True)
    env = gym.wrappers.ResizeObservation(env, shape=(480 // obs_scale, 640 // obs_scale))

    x = np.zeros((size, *env.observation_space.shape), dtype=env.observation_space.dtype)
    y = np.zeros((size, 3 + 3))

    for i in range(size):
        print(f"\r{i}/{size}", end='')
        env.reset()
        obs, reward, done, info = env.step([0, 0])

        x[i] = obs
        y[i, 0] = info['Simulator']['lane_position']['dist']
        y[i, 1] = info['Simulator']['lane_position']['angle_rad']
        y[i, 2] = get_drive_curve_info(env)
        y[i, 3:] = get_objects_info(env, info)

    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='MultiMap-v0')
    parser.add_argument('--size', type=int, default=250000)
    parser.add_argument('--obs_scale', type=int, default=12)

    args = parser.parse_args()

    x, y = make_dataset(
        args.env,
        args.obs_scale,
        args.size
    )

    with open(args.env + '_dataset_x.npy', 'wb') as file:
        np.save(file, x)

    with open(args.env + '_dataset_y.npy', 'wb') as file:
        np.save(file, y)
