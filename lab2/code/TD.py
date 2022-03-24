from GridWorld import GridWorld
import numpy as np


class TemporalDifference:
    def __init__(self, shape, terminal):
        self.gridworld = GridWorld(shape, terminal)

    def work(self, iter_time=10000, alpha=0.1, gamma=1.0):
        self.gridworld.clear()
        for i in range(iter_time):
            episode = self.gridworld.generate_episode()
            for idx in range(1, len(episode)):
                item = episode[idx - 1]['pos']
                next_item = episode[idx]['pos']
                now_value = self.gridworld.value[item[0], item[1]]
                next_value = self.gridworld.value[next_item[0], next_item[1]]
                gain = -1
                if item == next_item:
                    gain = 0
                self.gridworld.value[item[0], item[1]] = now_value + alpha * (
                        gain + gamma * next_value - now_value)

        print("###### Temporal-Difference(0) Method ######")
        print(self.gridworld.value)
        self.gridworld.get_policy()
        self.gridworld.draw_value('TD_value.png',
                                  title="TD(0) Method, $iter\_time=" + str(iter_time) + "$")
        self.gridworld.draw_policy('TD_policy.png',
                                   title="TD(0) Method, $iter\_time=" + str(iter_time) + "$")


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    env = TemporalDifference(shape=[6, 6], terminal=[[0, 1], [5, 5]])
    env.work(iter_time=50000, alpha=0.1, gamma=1.0)
