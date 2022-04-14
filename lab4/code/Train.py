import torch
import numpy as np
import gym
import time
from DQN import DQN
from DuelingDQN import DuelingDQN
from DDQN import DDQN


def calc_reward(pos):
    if pos >= 0.5:
        return 100
    elif pos <= -0.5:
        return -1
    else:
        return (10 * (pos + 0.5)) ** 2


class Train:
    def __init__(self, category, iter_time=1000, visual=False):
        self.category = category
        self.iter_time = iter_time
        self.visual = visual

    def train(self):
        memory_size = 2000
        env = gym.make('MountainCar-v0').unwrapped
        torch.manual_seed(2022)  # in order to reproduce the result

        dpn = None
        if self.category == 'DuelingDQN':
            dqn = DuelingDQN(action_num=env.action_space.n, state_num=env.observation_space.shape[0])
        elif self.category == 'DDQN':
            dqn = DDQN(action_num=env.action_space.n, state_num=env.observation_space.shape[0])
        else:
            dqn = DQN(action_num=env.action_space.n, state_num=env.observation_space.shape[0])

        episode_step = []
        episode_time = []
        avg_reward = []

        for i in range(self.iter_time):
            print('Episode: %s' % i)
            start_time = time.time()
            s = env.reset()
            total_reward = 0
            num = 0
            while True:
                # if i == 30:
                #     env.render()  # in order to debug

                if self.visual:
                    env.render()

                a = dqn.move(s)
                s_, reward, done, info = env.step(a)

                # pos, vel = s_
                reward_ = calc_reward(s_[0])

                dqn.save_buffer(s, a, reward_, s_)
                total_reward += reward
                s = s_

                if dqn.memory_counter > memory_size:
                    dqn.learn()
                    num += 1

                if done or total_reward <= -3000:
                    end_time = time.time()
                    print('reward=', total_reward, ' time=', end_time - start_time)
                    episode_step.append(-total_reward)
                    episode_time.append(end_time - start_time)
                    avg_reward.append(total_reward / (i + 1))
                    break

                    # if i >= 100 and np.average(episode_reward[-100:]) >= -100.0:
                    #     break

        return episode_step, episode_time, avg_reward
