import numpy as np
import matplotlib.pyplot as plt
from Gridworld import GridWorld


class PolicyIteration:
    def __init__(self, shape, terminal):
        self.gridworld = GridWorld(shape, terminal)

    def policy_evaluation(self, theta=0.001):
        is_converge = False
        iter_num = 0
        n = self.gridworld.n
        m = self.gridworld.m
        while not is_converge:
            # print(f"######## iter_time = {iter_num} ########")
            iter_num += 1
            delta = 0
            tmp_value = np.zeros_like(self.gridworld.value)
            for x in range(n):
                for y in range(m):
                    if self.gridworld.is_terminal(x, y):
                        continue
                    else:
                        tmp_value[x][y] = self.gridworld.calc_value(x, y)
                        delta = max(delta, abs(tmp_value[x][y] - self.gridworld.value[x][y]))

            # print(tmp_value)
            self.gridworld.value = tmp_value
            if delta < theta:
                is_converge = True

    def policy_improvement(self):
        policy_stable = True
        n = self.gridworld.n
        m = self.gridworld.m
        for x in range(n):
            for y in range(m):
                if self.gridworld.is_terminal(x, y):
                    continue
                else:
                    tmp_idx = 0
                    tmp = -float('inf')
                    for idx in range(len(self.gridworld.action)):
                        next_pos = np.array([x, y]) + np.array(self.gridworld.action[idx])
                        if next_pos[0] < 0 or next_pos[0] >= n or next_pos[1] < 0 or next_pos[1] >= m:
                            continue
                        elif self.gridworld.value[next_pos[0], next_pos[1]] > tmp:
                            tmp_idx = idx
                            tmp = self.gridworld.value[next_pos[0], next_pos[1]]
                    if self.gridworld.policy[x][y] != tmp_idx:
                        policy_stable = False
                        self.gridworld.policy[x][y] = tmp_idx
        return policy_stable


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    env = PolicyIteration(shape=[6, 6], terminal=[[0, 1], [5, 5]])
    while True:
        env.policy_evaluation()
        if env.policy_improvement():
            break

    env.gridworld.draw_value('policy_iter_1.png')
    env.gridworld.draw_policy('policy_iter_2.png')
