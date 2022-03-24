import numpy as np
import matplotlib.pyplot as plt
import random


class GridWorld:
    def __init__(self, shape, terminal):
        self.n = shape[0]
        self.m = shape[1]
        self.value = np.zeros([self.n, self.m])
        self.num = np.zeros([self.n, self.m])
        self.action = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.terminal = terminal
        self.reward = -1
        self.prob = 0.25
        self.discount_factor = 1.0
        self.policy = np.zeros_like(self.value)

    def is_terminal(self, x, y):  # judge whether the given grid is terminal
        return [x, y] in self.terminal

    def in_bound(self, x, y):  # judge whether the position is in bound
        return 0 <= x < self.n and 0 <= y < self.m

    def get_policy(self):
        n = self.n
        m = self.m
        for x in range(n):
            for y in range(m):
                if self.is_terminal(x, y):
                    continue
                else:
                    tmp = -float('inf')
                    tmp_idx = 0
                    for idx in range(len(self.action)):
                        next_pos = np.array([x, y]) + np.array(self.action[idx])
                        if not self.in_bound(next_pos[0], next_pos[1]):
                            continue
                        elif self.value[next_pos[0], next_pos[1]] > tmp:
                            tmp_idx = idx
                            tmp = self.value[next_pos[0], next_pos[1]]
                    self.policy[x][y] = tmp_idx

    def draw_value(self, filename='1.png', title=None):  # draw value matrix
        # print(self.value)
        fig = plt.figure(figsize=(6, 6))
        if title is not None:
            plt.title(title)
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

    def draw_policy(self, filename='2.png', title=None):  # draw policy/action matrix
        fig_pos = [(0.5, 1.0), (0.5, 0.0), (0.0, 0.5), (1.0, 0.5)]
        fig = plt.figure(figsize=(6, 6))
        if title is not None:
            plt.title(title)
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

    def clear(self):
        self.value[:] = 0
        self.num[:] = 0
        self.policy[:] = 0

    def generate_episode(self):
        res = []
        now_x = random.randint(0, self.n - 1)
        now_y = random.randint(0, self.m - 1)
        gain = 0
        while True:
            res.append({'pos': (now_x, now_y), 'gain': gain})
            action = self.action[random.randint(0, 3)]
            if self.is_terminal(now_x, now_y):
                gain += 1
                break
            if self.in_bound(now_x + action[0], now_y + action[1]):  # if not in bound, remain in the current position
                now_x += action[0]
                now_y += action[1]
            gain -= 1
        return res
