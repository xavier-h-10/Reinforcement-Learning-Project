import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CliffWalking:
    def __init__(self, shape, alpha, eps):
        self.n = shape[0]
        self.m = shape[1]
        self.start = [self.n - 1, 0]
        self.terminal = [self.n - 1, self.m - 1]
        self.action = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.Q = np.zeros((self.n, self.m, 4))
        self.eps = eps
        self.usual_reward = -1
        self.cliff_reward = -100

    def move(self, state, idx):
        next_state = np.array(state) + np.array(self.action[idx])
        if not (0 <= next_state[0] <= self.n - 1 and 0 <= next_state[1] <= self.m - 1):
            next_state = state
        reward = self.usual_reward
        if next_state[0] == self.n - 1 and 0 < next_state[1] < self.m - 1:
            next_state = self.start
            reward = self.cliff_reward
        return next_state, reward

    def eps_greedy(self, pos):
        p = np.random.random()
        if p < self.eps:
            return np.random.choice(4)
        else:
            idx = []
            mx = max(self.Q[pos[0], pos[1], :])
            for i in range(4):
                if self.Q[pos[0], pos[1], i] == mx:
                    idx.append(i)
            return np.random.choice(idx)

    def is_terminal(self, state):
        return state[0] == self.terminal[0] and state[1] == self.terminal[1]

    def draw_path(self, policy, title, filename='1.png'):
        policy = policy.astype(np.int16)
        print(policy)
        fig_pos = [(0.5, 1.0), (0.5, 0.0), (0.0, 0.5), (1.0, 0.5)]
        fig = plt.figure(figsize=(self.m, self.n))
        if title is not None:
            plt.title(title)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        for x in range(self.n):
            for y in range(self.m):
                ax = fig.add_subplot(self.n, self.m, x * self.m + y + 1)
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])

                if x == self.n - 1 and 1 <= y <= self.m - 2:
                    ax.spines['bottom'].set_linewidth(3)
                    ax.spines['top'].set_linewidth(3)
                    ax.spines['left'].set_linewidth(0)
                    ax.spines['right'].set_linewidth(0)
                    ax.set_facecolor('grey')
                else:
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['top'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    ax.spines['right'].set_linewidth(2)

                if policy[x][y] != 0 and not (x == self.n - 1 and y == 0):
                    ax.annotate('', fig_pos[policy[x][y] - 1], xytext=(0.5, 0.5), xycoords='axes fraction',
                                arrowprops={'width': 1.5, 'headwidth': 10, 'color': 'black'})

                if x == self.n - 1 and y == 1:
                    ax.spines['left'].set_linewidth(3)

                if x == self.n - 1 and y == self.m - 2:
                    ax.spines['right'].set_linewidth(3)

                if x == self.n - 1 and y == 0:
                    ax.text(0.5, 0.5, 'S', horizontalalignment='center',
                            verticalalignment='center', fontdict={'size': 25})

                if x == self.n - 1 and y == self.m - 1:
                    ax.text(0.5, 0.5, 'G', horizontalalignment='center',
                            verticalalignment='center', fontdict={'size': 25})

        plt.savefig(filename, dpi=1000)
        plt.show()

    def draw_heatmap(self, title, filename='2.png'):
        data = np.mean(self.Q, axis=2)
        ax = sns.heatmap(data)
        if title is not None:
            ax.set_title(title)
        plt.savefig(filename, dpi=1000)
        plt.show()
