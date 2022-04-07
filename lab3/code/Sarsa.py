from CliffWalking import CliffWalking
import numpy as np


class Sarsa:
    def __init__(self, shape, alpha, eps):
        self.env = CliffWalking(shape, alpha, eps)
        self.alpha = alpha
        self.eps = eps

    def work(self):
        episode = 0
        while True:
            episode += 1
            print("episode=", episode)
            state = self.env.start
            action = self.env.eps_greedy(state)

            num = 0
            while not self.env.is_terminal(state):
                num += 1
                if num < 100:
                    print(num, " ", state, " ", action)

                next_state, reward = self.env.move(state, action)
                next_action = self.env.eps_greedy(next_state)
                self.env.Q[state[0], state[1], action] += self.alpha * (
                        reward + self.eps * self.env.Q[next_state[0], next_state[1], next_action]
                        - self.env.Q[state[0], state[1], action])
                state = next_state
                action = next_action


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    shape = [4, 12]
    epsilon_set = [0, 0.1]

    # for epsilon in epsilon_set:
    #     sarsa = Sarsa(shape=shape, alpha=0.1, eps=epsilon)
    #     sarsa.work()


