import gym
import numpy as np
from gym import spaces

class MyGame(gym.Env):
    def __init__(self, config):
        self.grid_size = config['grid_size']
        self.action_space = spaces.Discrete(4) # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=5, shape=(2,))
        self.reset()

    def reset(self):
        self.position = [0, 0]
        self.goal_position = [self.grid_size-1, self.grid_size-1]
        self.goal_reward = 10
        self.out_of_bounds_reward = -1
        self.action_penalty = -0.1
        self.current_step = 0
        self.max_steps = 50
        return np.array(self.position)

    def step(self, action):
        if action == 0: # up
            self.position[0] -= 1
        elif action == 1: # down
            self.position[0] += 1
        elif action == 2: # left
            self.position[1] -= 1
        elif action == 3: # right
            self.position[1] += 1

        if self.position[0] < 0 or self.position[0] >= self.grid_size or self.position[1] < 0 or self.position[1] >= self.grid_size:
            reward = self.out_of_bounds_reward
            done = True
        elif self.position == self.goal_position:
            reward = self.goal_reward
            done = True
        else:
            reward = self.action_penalty
            done = False

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        print("New State:")
        self._render(reward)

        return np.array(self.position), reward, done, {}

    def _render(self, reward=None):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.position == [i, j]:
                    print("A", end=" ")
                elif self.goal_position == [i, j]:
                    print("G", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

        if reward is not None:
            print("Reward:", reward)

# Use gym.make() to create the environment
env = gym.make('my_env_module:MyGame', config={'grid_size': 5})

num_episodes = 3
for episode in range(num_episodes):
    print("Episode:", episode+1)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env._render(reward=reward)
    print("Episode finished\n")