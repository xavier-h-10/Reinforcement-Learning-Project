import numpy as np
import matplotlib.pyplot as plt


class PolicyIteration:
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

    def calc_value(self, x, y):
        v = 0
        for action in self.action:
            next_pos = np.array([x, y]) + np.array(action)
            if next_pos[0] < 0 or next_pos[0] >= self.n or next_pos[1] < 0 or next_pos[1] >= self.m:
                v += (self.reward + self.discount_factor * self.value[x, y]) * self.prob
            else:
                v += (self.reward + self.discount_factor * self.value[next_pos[0], next_pos[1]]) * self.prob

        return v

    def is_terminal(self, x, y):
        return [x, y] in self.terminal

    def policy_evaluation(self, theta=0.001):
        is_converge = False
        iter_num = 0
        while not is_converge:
            # print(f"######## iter_time = {iter_num} ########")
            iter_num += 1
            delta = 0
            tmp_value = np.zeros_like(self.value)
            for x in range(self.n):
                for y in range(self.m):
                    if self.is_terminal(x, y):
                        continue
                    else:
                        tmp_value[x][y] = self.calc_value(x, y)
                        delta = max(delta, abs(tmp_value[x][y] - self.value[x][y]))

            # print(tmp_value)
            self.value = tmp_value
            if delta < theta:
                is_converge = True

    def policy_improvement(self):
        policy_stable = True
        for x in range(self.n):
            for y in range(self.m):
                if self.is_terminal(x, y):
                    continue
                else:
                    tmp_idx = 0
                    tmp = -float('inf')
                    for idx in range(len(self.action)):
                        next_pos = np.array([x, y]) + np.array(self.action[idx])
                        if next_pos[0] < 0 or next_pos[0] >= self.n or next_pos[1] < 0 or next_pos[1] >= self.m:
                            continue
                        elif self.value[next_pos[0], next_pos[1]] > tmp:
                            tmp_idx = idx
                            tmp = self.value[next_pos[0], next_pos[1]]
                    if self.policy[x][y] != tmp_idx:
                        policy_stable = False
                        self.policy[x][y] = tmp_idx
        return policy_stable

    def draw_value(self):
        print(self.value)

    def draw_policy(self):
        pos = ['U', 'D', 'L', 'R']
        for x in range(self.n):
            for y in range(self.m):
                if self.is_terminal(x, y):
                    print('. ', end='')
                else:
                    now = pos[int(self.policy[x][y])]
                    print(now + ' ', end='')
            print('')


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    env = PolicyIteration(shape=[6, 6], terminal=[[0, 1], [5, 5]])
    while True:
        env.policy_evaluation()
        if env.policy_improvement():
            break

    env.draw_value()
    env.draw_policy()
