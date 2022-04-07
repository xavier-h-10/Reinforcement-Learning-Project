from Sarsa import Sarsa
from QLearning import QLearning
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    shape = [4, 12]
    epsilon_set = [0, 0.1, 0.3]
    np.set_printoptions(precision=2)

    for idx, epsilon in enumerate(epsilon_set):
        sarsa = Sarsa(shape=shape, alpha=0.1, gamma=0.95, eps=epsilon, episode=2000)
        q_learning = QLearning(shape=shape, alpha=0.1, gamma=0.95, eps=epsilon, episode=2000)

        print("##### SARSA Method, eps=", epsilon, " #####")
        sarsa.work()
        sarsa.draw_path()
        sarsa.draw_heatmap()

        print("##### Q-Learning Method, eps=", epsilon, " #####")
        q_learning.work()
        q_learning.draw_path()
        q_learning.draw_heatmap()

        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.title('Cliff-Walking $\epsilon=' + str(epsilon) + '$')
        plt.plot(sarsa.avg_reward, label='SARSA')
        plt.plot(q_learning.avg_reward, label='Q-Learning')
        plt.legend()
        plt.savefig(str(idx) + '.png', dpi=1000)
        plt.show()
