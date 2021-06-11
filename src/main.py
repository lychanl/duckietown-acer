
import gym_duckietown



if __name__ == '__main__':
    import tftools  # prepare cnn override

    from agent import ActorAgent
    import argparse
    import gym
    import json
    import logging
    from wrappers import RescaleObsToFloatWrapper
    from test import test
    import tools

    from experiment import Experiment

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='Duckietown-loop_empty-v0')
    parser.add_argument('--algo', type=str, default='fastacer')
    parser.add_argument('--obs_scale', type=int, default=20)
    parser.add_argument('--no_grayscale', action='store_true', default=False)
    parser.add_argument('--center', action='store_true', default=False)
    parser.add_argument('--time_limit', type=int, default=1500)
    parser.add_argument('--reward_scale', type=int, default=1)
    parser.add_argument('--dir_change_penalty', type=float, default=None)

    parser.add_argument('--memory_size', type=int, default=int(5e5))
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--actor_lr', type=float, default=1e-5)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--std', type=float, default=0.4)
    parser.add_argument('--batches_per_env', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor_beta_penalty', type=float, default=0.001)

    parser.add_argument('--max_time_steps', type=int, default=1000000)
    parser.add_argument('--evaluate_time_steps_interval', type=int, default=10000)
    parser.add_argument('--num_evaluation_runs', type=int, default=5)
    parser.add_argument('--num_parallel_envs', type=int, default=5)

    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--asynchronous', action='store_true', default=False)

    parser.add_argument('--save_model', action='store_true', default=False)

    parser.add_argument('--filters', type=int, nargs='+', default=[16, 32, 32, 64, 64])
    parser.add_argument('--kernels', type=int, nargs='+', default=[8, 8, 4, 4, 1])
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 4, 1, 2, 1])
    parser.add_argument('--layers', type=int, nargs='+', default=[256, 256])

    parser.add_argument('--data_model_path', type=str, default=None)

    args = parser.parse_args()

    cnn_params = {
        'filters': args.filters,
        'kernels': args.kernels,
        'strides': args.layers
    }

    wrappers_params = {
        'no_grayscale': args.no_grayscale,
        'obs_scale': args.obs_scale,
        'center': args.center,
        'time_limit': args.time_limit,
        'reward_scale': args.reward_scale,
        'dir_change_penalty': args.dir_change_penalty,
        'data_model_path': args.data_model_path
    }

    logging.getLogger().setLevel(logging.INFO)
    experiment = Experiment(
        args.env,
        wrappers=tools.wrappers(**wrappers_params),
        obs_scale=args.obs_scale,
        no_grayscale=args.no_grayscale,
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
        cnn_params=cnn_params,
        data_model_path=args.data_model_path,
        max_time_steps=args.max_time_steps,
        evaluate_time_steps_interval=args.evaluate_time_steps_interval,
        num_evaluation_runs=args.num_evaluation_runs,
        log_tensorboard=args.tensorboard,
        do_checkpoint=False,
        num_parallel_envs=args.num_parallel_envs,
        log_dir=args.log_dir
    )

    experiment.run()

    test_results = test(
        args.env,
        ActorAgent(actor=experiment.runner._agent._actor, wrappers_params=wrappers_params),
        10
    )

    if args.save_model:
        experiment.runner._agent._actor.save_weights(experiment.runner._log_dir / 'model' / 'weights')

        model_params = {
            'wrappers_params': wrappers_params,
            'cnn_params': cnn_params,
            'args': (
                experiment.runner._agent._actor_layers,
                experiment.runner._agent._actor_beta_penalty,
                list(float(b) for b in experiment.runner._agent._actions_bound),
                experiment.runner._agent._std
            )
        }
        with open(experiment.runner._log_dir / 'model' / 'params', 'w') as params_file:
            json.dump(model_params, params_file)
        with open(experiment.runner._log_dir / 'test', 'w') as test_file:
            json.dump(test_results, test_file)
