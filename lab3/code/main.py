from Sarsa import Sarsa
from QLearning import QLearning
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    shape = [4, 12]
    epsilon_set = [0, 0.1, 0.5]
    np.set_printoptions(precision=2)

    for idx, epsilon in enumerate(epsilon_set):
        sarsa = Sarsa(shape=shape, alpha=0.1, gamma=0.9, eps=epsilon, episode=2000)
        q_learning = QLearning(shape=shape, alpha=0.1, gamma=0.9, eps=0.1, episode=2000)

        print("##### SARSA Method, eps=", epsilon, " #####")
        sarsa.work()
        sarsa.draw_path()

        print("##### Q-Learning Method, eps=", epsilon, " #####")
        q_learning.work()
        q_learning.draw_path()

        plt.ylim(-500, 0)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.title('Cliff-Walkng $\epsilon=' + str(epsilon) + '$')
        plt.plot(sarsa.reward, label='SARSA')
        plt.plot(q_learning.reward, label='Q-Learning')
        plt.legend()
        plt.show()
        plt.savefig(str(idx) + '.png', dpi=1000)
