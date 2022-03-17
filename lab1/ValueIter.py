import numpy as np


class ValueIteration:
    def __init__(self, shape):
        self.n = shape[0]
        self.m = shape[1]
        self.value = np.zeros([self.n, self.m])
        self.action = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.terminal = [[0, 1], [5, 5]]
        self.reward = -1
        self.prob = 0.25
        self.discount_factor = 1.0

    def calc_value(self, x, y):
        v = np.zeros(range(self.action))
        for idx in range(self.action):
            [next_x, next_y] = [x, y] + self.action[idx]
            if next_x < 0 or next_x >= self.n or next_y < 0 or next_y >= self.m:
                v[idx] = float('-inf')

        return np.max(v)

    def is_terminal(self, x, y):
        return [x, y] in self.terminal

    def iter(self, theta=0.0001):
        is_converge = False
        iter_num = 0
        while not is_converge:
            iter_num += 1
            for x in range(self.n):
                for y in range(self.m):
                    if self.is_terminal(x, y):
                        continue
                    else:
                        origin = self.calc_value(x, y)

            break


if __name__ == '__main__':
    env = ValueIteration([6, 6])
    env.iter()
