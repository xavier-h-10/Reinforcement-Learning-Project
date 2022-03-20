from GridWorld import GridWorld
import numpy as np


class MonteCarlo:
    def __init__(self, shape, terminal):
        self.gridworld = GridWorld(shape, terminal)

    def first_visit(self, iter_time=100000):
        self.gridworld.clear()
        for i in range(iter_time):
            episode, gain = self.gridworld.generate_episode()
            episode = set(episode)  # for first visit, we should remove duplicated items
            for item in episode:
                if self.gridworld.is_terminal(item[0], item[1]):
                    continue
                self.gridworld.num[item] += 1
                self.gridworld.value[item] += (1 / self.gridworld.num[item]) * (gain - self.gridworld.value[item])

        print("###### First-Visit MC Method ######")
        print(self.gridworld.value)
        self.gridworld.get_policy()
        self.gridworld.draw_value('first_visit_value.png')
        self.gridworld.draw_policy('first_visit_policy.png')

    def every_visit(self, iter_time=100000):
        self.gridworld.clear()
        for i in range(iter_time):
            episode, gain = self.gridworld.generate_episode()
            for item in episode:
                if self.gridworld.is_terminal(item[0], item[1]):
                    continue
                self.gridworld.num[item] += 1
                self.gridworld.value[item] += (1 / self.gridworld.num[item]) * (gain - self.gridworld.value[item])

        print("###### Every-Visit MC Method ######")
        print(self.gridworld.value)
        self.gridworld.get_policy()
        self.gridworld.draw_value('every_visit_value.png', title="$iter\_time=" + str(iter_time) + "$")
        self.gridworld.draw_policy('every_visit_policy.png', title="$iter\_time=" + str(iter_time) + "$")


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    env = MonteCarlo(shape=[6, 6], terminal=[[0, 1], [5, 5]])
    env.first_visit(iter_time=100000)
    env.every_visit(iter_time=100000)
