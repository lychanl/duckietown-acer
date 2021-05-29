import gym
import gym_duckietown

import agent as agent_


if __name__ == '__main__':
    env = gym.make('Duckietown-loop_pedestrians-v0')
    env = gym.wrappers.TimeLimit(env, 60 * 30)

    agent = agent_.NoOpAgent()

    for i in range(5):
        reward = 0
        step = 0
        obs = env.reset()
        env.render()
        done = False

        while not done:
            action = agent.act(obs)
            obs, rew, done, info = env.step(action)
            env.render()
            reward += rew
            step += 0

        print(f'Run: {i}, steps: {step}, reward: {reward}')
