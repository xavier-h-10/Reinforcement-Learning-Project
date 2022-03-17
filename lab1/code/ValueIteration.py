import numpy as np
import matplotlib.pyplot as plt
from GridWorld import GridWorld


class ValueIteration:
    def __init__(self, shape, terminal):
        self.gridworld = GridWorld(shape, terminal)

    def calc_value(self, x, y):
        v = -float('inf')
        n = self.gridworld.n
        m = self.gridworld.m
        for action in self.gridworld.action:
            next_pos = np.array([x, y]) + np.array(action)
            if next_pos[0] < 0 or next_pos[0] >= n or next_pos[1] < 0 or next_pos[1] >= m:
                continue
            else:
                v = max(v, (self.gridworld.reward + self.gridworld.discount_factor * self.gridworld.value[
                    next_pos[0], next_pos[1]]) * self.gridworld.prob)

        return v

    def iteration(self, theta=0.001):
        is_converge = False
        iter_num = 0
        n = self.gridworld.n
        m = self.gridworld.m
        while not is_converge:
        #    print(f"######## iter_time = {iter_num} ########")
            iter_num += 1
            delta = 0
            tmp_value = np.zeros_like(self.gridworld.value)
            for x in range(n):
                for y in range(m):
                    if self.gridworld.is_terminal(x, y):
                        continue
                    else:
                        tmp_value[x][y] = self.calc_value(x, y)
                        delta = max(delta, abs(tmp_value[x][y] - self.gridworld.value[x][y]))

            # print(tmp_value)
            self.gridworld.value = tmp_value
            if delta < theta:
                is_converge = True

    def get_policy(self):
        n = self.gridworld.n
        m = self.gridworld.m
        for x in range(n):
            for y in range(m):
                if self.gridworld.is_terminal(x, y):
                    continue
                else:
                    tmp = -float('inf')
                    tmp_idx = 0
                    for idx in range(len(self.gridworld.action)):
                        next_pos = np.array([x, y]) + np.array(self.gridworld.action[idx])
                        if next_pos[0] < 0 or next_pos[0] >= n or next_pos[1] < 0 or next_pos[1] >= m:
                            continue
                        elif self.gridworld.value[next_pos[0], next_pos[1]] > tmp:
                            tmp_idx = idx
                            tmp = self.gridworld.value[next_pos[0], next_pos[1]]
                    self.gridworld.policy[x][y] = tmp_idx


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    env = ValueIteration(shape=[6, 6], terminal=[[0, 1], [5, 5]])
    env.iteration()
    env.get_policy()

    env.gridworld.draw_value('value_iter_1.png')
    env.gridworld.draw_policy('value_iter_2.png')
