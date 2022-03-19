import numpy as np
import matplotlib.pyplot as plt
import random


class GridWorld:
    def __init__(self, shape, terminal):
        self.n = shape[0]
        self.m = shape[1]
        self.value = np.zeros([self.n, self.m])
        self.action = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.terminal = terminal
        self.reward = -1
        self.prob = 0.25
        self.discount_factor = 1.0
        self.policy = np.zeros_like(self.value)

    def is_terminal(self, x, y):  # judge whether the given grid is terminal
        return [x, y] in self.terminal

    def draw_value(self, filename='1.png'):  # draw value matrix
        # print(self.value)
        fig = plt.figure(figsize=(6, 6))
        plt.subplots_adjust(wspace=0, hspace=0)
        for x in range(self.n):
            for y in range(self.m):
                ax = fig.add_subplot(self.n, self.m, x * self.m + y + 1)
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])

                ax.spines['bottom'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)
                ax.text(0.5, 0.5, round(self.value[x][y], 2), horizontalalignment='center', verticalalignment='center',
                        fontdict={'size': 14})
        plt.savefig(filename, dpi=1000)
        plt.show()

    def draw_policy(self, filename='2.png'):  # draw policy/action matrix
        fig_pos = [(0.5, 1.0), (0.5, 0.0), (0.0, 0.5), (1.0, 0.5)]
        fig = plt.figure(figsize=(6, 6))
        plt.subplots_adjust(wspace=0, hspace=0)
        for x in range(self.n):
            for y in range(self.m):
                ax = fig.add_subplot(self.n, self.m, x * self.m + y + 1)
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])

                ax.spines['bottom'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)

                if self.is_terminal(x, y):
                    ax.set_facecolor('grey')
                else:
                    pos = np.array([x, y]) + np.array(self.action[int(self.policy[x][y])])
                    value = self.value[pos[0], pos[1]]

                    for idx in range(len(self.action)):
                        next_pos = np.array([x, y]) + np.array(self.action[idx])
                        if next_pos[0] < 0 or next_pos[0] >= self.n or next_pos[1] < 0 or next_pos[1] >= self.m:
                            continue
                        elif abs(self.value[next_pos[0], next_pos[1]] - value) < 0.0001:
                            ax.annotate('', fig_pos[idx], xytext=(0.5, 0.5), xycoords='axes fraction',
                                        arrowprops={'width': 1.5, 'headwidth': 10, 'color': 'black'})
        plt.savefig(filename, dpi=1000)
        plt.show()

    def generate_episode(self):
        mx = self.n * self.m - 1
        res = []
        num = 0
        while num < mx:
            now = random.randint(0, mx)
            now_x = now / self.m
            now_y = now - self.m * now_x
            res.append([now_x, now_y])
            if self.is_terminal(now_x, now_y):
                break
        return res
