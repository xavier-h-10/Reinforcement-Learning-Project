from GridWorld import GridWorld
import numpy as np


class TemporalDifference:
    def __init__(self, shape, terminal):
        self.gridworld = GridWorld(shape, terminal)

    def work(self, iter_time=10, alpha=0.1, gamma=0.9):
        self.gridworld.clear()
        for i in range(iter_time):
            episode, gain = self.gridworld.generate_episode()
            now_return = 0
            for idx in range(len(episode) - 1):
                item = episode[idx]
                next_item = episode[idx + 1]
                now_value = self.gridworld.value[item[0], item[1]]
                next_value = self.gridworld.value[next_item[0], next_item[1]]

                if item != next_item:
                    now_return -= 1
                self.gridworld.value[item[0], item[1]] = now_value + alpha * (
                            now_return + gamma * next_value - now_value)

        print("###### Temporal-Difference(0) Method ######")
        print(self.gridworld.value)
        self.gridworld.get_policy()
        self.gridworld.draw_value('TD_value.png')
        self.gridworld.draw_policy('TD_policy.png')


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    env = TemporalDifference(shape=[6, 6], terminal=[[0, 1], [5, 5]])
    env.work(iter_time=100000, alpha=0.1, gamma=0.7)
