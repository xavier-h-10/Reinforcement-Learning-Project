import torch
import numpy as np
import gym
import time
from DQN import DQN
from DuelingDQN import DuelingDQN

# 超参数
batch_size = 32  # 样本数量
LR = 0.01  # 学习率
eps = 0.9  # greedy policy
discount = 0.9  # reward discount
update_iter = 20  # 目标网络更新频率
memory_size = 2000  # 记忆库容量


class Train:
    def __init(self, category):
        self.category = category

    def train(self):
        memory_size = 2000
        env = gym.make('MountainCar-v0').unwrapped
        torch.manual_seed(2022)  # in order to reproduce the result
        action_num = env.action_space.n
        state_num = env.observation_space.shape[0]

        dpn = None
        if self.category == 'DuelingDQN':
            dqn = DuelingDQN()
        else:
            dqn = DQN()

        episode_reward = []
        episode_time = []

        for i in range(1000):
            print('Episode: %s' % i)
            start_time = time.time()
            s = env.reset()
            total_reward = 0
            num = 0
            while True:  # 开始一个episode (每一个循环代表一步)
                if i == 30:
                    env.render()  # 显示实验动画

                a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
                s_, r, done, info = env.step(a)  # 执行动作，获得反馈

                pos, vel = s_
                new_r = 0
                if -0.4 < s_[0] < 0.5:
                    new_r = 10 * (s_[0] + 0.4) ** 3
                elif s_[0] >= 0.5:
                    new_r = 100
                else:
                    new_r = -1
                # new_r = r

                dqn.store_transition(s, a, new_r, s_)  # 存储样本
                total_reward += r  # 逐步加上一个episode内每个step的reward

                s = s_

                if dqn.memory_counter > memory_size:  # 如果累计的transition数量超过了记忆库的固定容量2000
                    # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
                    dqn.learn()
                    num += 1

                if done:
                    # round()方法返回total_reward的小数点四舍五入到2个数字
                    print('episode%s---reward_sum: %s' % (i, round(total_reward, 2)))
                    episode_reward.append(total_reward)
                    print(num)
                    break  # 该episode结束

            end_time = time.time()
            episode_time.append(time, time)

            # if i >= 100 and np.average(episode_reward[-100:]) >= -100.0:
            #     break
        return episode_reward, episode_time
