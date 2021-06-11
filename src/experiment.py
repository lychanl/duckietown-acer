
import gym
import gym_duckietown
import tools
import runners
from dataset_regression import load_or_build_data_model, DataModelVecWrapper


class Experiment:
    def __init__(
            self, env_name: str, wrappers: list, obs_scale: int, no_grayscale: bool,
             num_parallel_envs: int, asynchronous: bool,
            algorithm: str, algorithm_parameters: dict, cnn_params: list, data_model_path: str,
            max_time_steps: int, evaluate_time_steps_interval: int, num_evaluation_runs: int,
            log_tensorboard: bool, do_checkpoint: bool, log_dir: str):

        data_model = None
        if data_model_path:
            data_model = load_or_build_data_model(
                data_model_path, input_shape=(480 // obs_scale, 640 // obs_scale, 3 if no_grayscale else 1)
            )

        runners._get_env = self.get_env_getter(wrappers, data_model)
        algorithm_parameters = dict(algorithm_parameters)
        algorithm_parameters['num_parallel_envs'] = num_parallel_envs

        if not data_model_path:
            from tftools import override_cnn
            override_cnn(**cnn_params)

        def build_runner():            
            return runners.Runner(
                env_name,
                algorithm=algorithm,
                asynchronous=asynchronous,
                algorithm_parameters=algorithm_parameters,
                max_time_steps=max_time_steps,
                evaluate_time_steps_interval=evaluate_time_steps_interval,
                num_evaluation_runs=num_evaluation_runs,
                log_tensorboard=log_tensorboard,
                num_parallel_envs=num_parallel_envs,
                do_checkpoint=do_checkpoint,
                log_dir=log_dir
            )
        self._runner = None
        self._build_runner = build_runner

    @property
    def runner(self):
        if not self._runner:
            self._runner = self._build_runner()
        return self._runner

    def run(self):
        self.runner.run()

    def get_env_getter(self, wrappers, data_model):
        def get_env(env_id: str, num_parallel_envs: int, asynchronous: bool = True):
            def builder():
                env = gym.make(env_id)
                if 'MultiMap' in env_id:
                    for e in env.unwrapped.env_list:
                        e.unwrapped.randomizer.randomization_config['frame_skip']['high'] = 2
                else:
                    env.unwrapped.randomizer.randomization_config['frame_skip']['high'] = 2
                for wrapper, params in wrappers:
                    env = wrapper(env, **params)
                return env
            builders = [builder for _ in range(num_parallel_envs)]
            env = gym.vector.AsyncVectorEnv(builders) if asynchronous else gym.vector.SyncVectorEnv(builders)
            if data_model:
                env = DataModelVecWrapper(env, data_model)
            return env
        return get_env
