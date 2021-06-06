import argparse
import json
from math import inf
import gym
from gym.wrappers.monitor import Monitor
import gym_duckietown
import numpy as np
import os
import time

from agent import AGENTS


def get_dir_vec(angle):
    x = np.cos(angle)
    z = -np.sin(angle)
    return np.array([x, 0, z])


def get_closest_heading(info, env):
    i, j = env.unwrapped.get_grid_coords(info['Simulator']['cur_pos'])
    tile = env.unwrapped._get_tile(i, j)
    curves = tile['curves']
    curve_headings = curves[:, -1, :] - curves[:, 0, :]

    curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
    dir_vec = get_dir_vec(info['Simulator']['cur_angle'])

    dot_prods = np.dot(curve_headings, dir_vec)

    # Closest curve = one with largest dotprod
    closest = np.argmax(dot_prods)
    return curve_headings[closest]


def get_dist_change(info):
    return info['Simulator'].get('lane_position', {'dot_dir': 0})['dot_dir'] * info['Simulator']['robot_speed']


def test(env, agent, episodes, save_path=None):
    env = gym.make(env, full_transparency=True)
    env = agent.wrap(env, eval=True)
    env.unwrapped.randomizer.randomization_config['frame_skip']['high'] = 2

    if save_path:
        env = Monitor(env, save_path, video_callable=lambda x: True, force=True)

    steps = []
    rewards = []
    dists = []
    total_steps_proximity = []
    total_steps_invalid_lane = []
    total_direction_changes = []

    for i in range(episodes):
        reward = 0
        step = 0
        dist = 0

        steps_proximity = 0
        steps_invalid_lane = 0
        direction_changes = 0

        env.seed(i)
        obs = env.reset()
        env.render()

        if i == 0:
            agent.prepare(env)

        done = False

        previous_heading = None
        dist_dir = 1

        while not done:
            action = agent.act(obs)
            obs, rew, done, info = env.step(action)
            env.render()

            closest_heading = get_closest_heading(info, env)

            if previous_heading is not None:
                if np.dot(closest_heading, previous_heading) < 0:
                    direction_changes += 1
                    dist_dir *= -1

            previous_heading = closest_heading

            if abs(info['Simulator'].get('lane_position', {'dist': 1})['dist']) > env.unwrapped.road_tile_size * 0.2:
                steps_invalid_lane += 1
            if info['Simulator'].get('proximity_penalty') > 0:
                steps_proximity += 1

            dist += dist_dir * get_dist_change(info)

            reward += rew
            step += 1

            w = agent.wait(env)
            if w > 0:
                time.sleep(w)

        dist = np.abs(dist)
        print(f'Run: {i}: steps: {step} reward: {reward} dist: {dist} ' +
            f'proximity: {steps_proximity} invalid_lane: {steps_invalid_lane} direction_changes: {direction_changes}')

        steps.append(step)
        rewards.append(reward)
        dists.append(dist)
        total_steps_proximity.append(steps_proximity)
        total_steps_invalid_lane.append(steps_invalid_lane)
        total_direction_changes.append(direction_changes)
        

    print(f'Mean: steps: {np.mean(steps)} reward: {np.mean(rewards)} dist: {np.mean(dists)} ' +
        f'proximity: {np.mean(total_steps_proximity)} invalid_lane: {np.mean(total_steps_invalid_lane)} ' +
        f'direction_changes: {np.mean(total_direction_changes)}')

    return [{
        'steps': s, 'rewards': r, 'dist': d, 'steps_proximity': p, 'steps_invalid_lane': i, direction_changes: 'c'
    } for s, r, d, p, i, c in zip(steps, rewards, dists, total_steps_proximity, total_steps_invalid_lane, total_direction_changes)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='Duckietown-loop_empty-v0')
    parser.add_argument('--agent', type=str, choices=AGENTS.keys(), required=True)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    agent = AGENTS[args.agent](path=args.model_path)

    infos = test(args.env, agent, args.episodes, args.save_path)
    
    if args.save_path:
        with open(os.path.join(args.save_path, 'tests.json'), 'w') as file:
            json.dump({
                    'episodes': infos,
                    'agent': args.agent,
                    'env': args.env,
                    'model': args.model_path
                },
                file  
            )
