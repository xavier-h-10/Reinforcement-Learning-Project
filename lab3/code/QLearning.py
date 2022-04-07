from CliffWalking import CliffWalking
import numpy as np


class QLearning:
    def __init__(self, shape, alpha=0.1, gamma=0.9, eps=0.1, episode=1000):
        self.env = CliffWalking(shape, alpha, eps)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reward = []
        self.avg_reward = []
        self.num = episode

    def work(self):
        episode = 1
        while episode <= self.num + 10:
            state = self.env.start
            num = 0
            total_reward = 0
            while not self.env.is_terminal(state):
                num += 1
                action = self.env.eps_greedy(state)
                next_state, reward = self.env.move(state, action)
                total_reward += reward

                self.env.Q[state[0], state[1], action] += self.alpha * (
                        reward + self.gamma * np.max(self.env.Q[next_state[0], next_state[1], :])
                        - self.env.Q[state[0], state[1], action])
                state = next_state
            self.reward.append(total_reward)
            print("episode=", episode, " num=", num)
            episode += 1

        for i in range(self.num):
            self.avg_reward.append(np.mean(self.reward[i:i + 10]))

    def draw_path(self):
        policy = np.zeros((self.env.n, self.env.m))
        state = self.env.start
        while not self.env.is_terminal(state):
            action = self.env.eps_greedy(state)
            policy[state[0], state[1]] = action + 1
            next_state, reward = self.env.move(state, action)
            state = next_state

        self.env.draw_path(policy, 'Q-Learning Policy $\epsilon=' + str(self.eps) + '$', 'Q-Learning.png')

    def draw_heatmap(self):
        self.env.draw_heatmap('Q-Learning $\epsilon=' + str(self.eps) + '$','QLearning heatmap.png')
