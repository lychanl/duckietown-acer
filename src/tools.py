import gym
from wrappers import RescaleObsToFloatWrapper

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


def wrappers(no_grayscale: bool, obs_scale: int, center: bool, time_limit: int = None, reward_scale: int = None, eval: bool = False):
    return [
        (gym.wrappers.ResizeObservation, {'shape': (480 // obs_scale, 640 // obs_scale)})
    ] + ([] if no_grayscale else [(gym.wrappers.GrayScaleObservation, {})]) + [
        (RescaleObsToFloatWrapper, {'scale': 1/128, 'x0': 128} if center else {'scale': 1/256})
    ] + ([] if eval else [
        (gym.wrappers.TimeLimit, {'max_episode_steps': time_limit}),
        (gym.wrappers.TransformReward, {'f': lambda x: x / reward_scale})
    ])
