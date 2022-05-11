import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym
import random
import torch

from agent import Agent
from utils import export_video


def train(max_episode=1000, max_step=300, use_noise=True):
    scores = []
    best_score = -10000.0
    for i in range(max_episode):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_step):
            action = agent.choose_action(state, use_noise=use_noise)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)

        print('Episode {} Score: {:.2f} Average Score: {:.2f}'.format(i + 1, score, np.mean(scores)))

        if score < best_score:
            best_score = score
            torch.save(agent.actor.state_dict(), 'ddpg_actor.pth')
            torch.save(agent.critic.state_dict(), 'ddpg_critic.pth')

    return scores


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    env.seed(2022)
    agent = Agent(state_num=3, action_num=1, seed=2022)
    scores = train(max_episode=1000, use_noise=False)
    np.save("ddpg_no_noise.npy", np.array(scores))
    export_video(env=env, agent=agent, max_episode=1000)
