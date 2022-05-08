import gym
import numpy as np
import torch

from A5.A3C.utils import v_wrap
from DDPG import DDPG

# define hyper parameters (from paper)
episodes = 1000
EP_STEPS = 200
alpha = 1e-4
beta = 1e-3
gamma = 0.99
tau = 0.001
input_dims = 3
n_actions = 1
env_id = 'Pendulum-v1'

if __name__ == '__main__':
    env = gym.make(env_id)
    agent = DDPG(alpha, beta, input_dims, tau, n_actions, gamma)

    score_history = []
    for i in range(episodes):
        s = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            # 乱七八糟改了一番，不报错了，但可能有问题。。。
            action = agent.choose_action(v_wrap(s, np.float32)).detach().numpy()
            s_, reward, done, _ = env.step(action)
            agent.remember(s, action, reward, s_, done)
            agent.learn()
            score += reward
            s = s_
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score,
              '100 games average score %.1f' % np.mean(score_history[-100:]))
