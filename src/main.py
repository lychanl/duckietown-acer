import argparse
import gym
import logging
from wrappers import RescaleObsToFloatWrapper

from experiment import Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='Duckietown-loop_empty-v0')
    parser.add_argument('--algo', type=str, default='fastacer')
    parser.add_argument('--obs_scale', type=int, default=20)
    parser.add_argument('--no_grayscale', action='store_true', default=False)
    parser.add_argument('--center', action='store_true', default=False)
    parser.add_argument('--time_limit', type=int, default=1500)
    parser.add_argument('--reward_scale', type=int, default=1)

    parser.add_argument('--memory_size', type=int, default=int(5e5))
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--actor_lr', type=float, default=1e-5)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--std', type=float, default=0.4)
    parser.add_argument('--batches_per_env', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor_beta_penalty', type=float, default=0.1)

    parser.add_argument('--max_time_steps', type=int, default=1000000)
    parser.add_argument('--evaluate_time_steps_interval', type=int, default=10000)
    parser.add_argument('--num_evaluation_runs', type=int, default=5)
    parser.add_argument('--num_parallel_envs', type=int, default=5)

    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--asynchronous', action='store_true', default=False)

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    experiment = Experiment(
        args.env,
        wrappers=[
            (gym.wrappers.ResizeObservation, {'shape': (480 // args.obs_scale, 640 // args.obs_scale)})
        ] + ([] if args.no_grayscale else [(gym.wrappers.GrayScaleObservation, {})]) + [
            (RescaleObsToFloatWrapper, {'scale': 1/128, 'x0': 128} if args.center else {'scale': 1/256}),
            (gym.wrappers.TimeLimit, {'max_episode_steps': args.time_limit}),
            (gym.wrappers.TransformReward, {'f': lambda x: x / args.reward_scale})
        ],
        algorithm=args.algo,
        asynchronous=args.asynchronous,
        algorithm_parameters={
            'memory_size': args.memory_size,
            'actor_layers': (256, 256),
            'critic_layers': (256, 256),
            'c': args.c,
            'actor_lr': args.actor_lr,
            'critic_lr': args.critic_lr,
            'std': args.std,
            'batches_per_env': args.batches_per_env,
            'gamma': args.gamma,
            'actor_beta_penalty': args.actor_beta_penalty
        },
        max_time_steps=args.max_time_steps,
        evaluate_time_steps_interval=args.evaluate_time_steps_interval,
        num_evaluation_runs=args.num_evaluation_runs,
        log_tensorboard=args.tensorboard,
        do_checkpoint=False,
        num_parallel_envs=args.num_parallel_envs,
        log_dir='C:/dt_logs'
    )

    experiment.run()
