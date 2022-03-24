from GridWorld import GridWorld
import numpy as np


class MonteCarlo:
    def __init__(self, shape, terminal):
        self.gridworld = GridWorld(shape, terminal)

    def first_visit(self, iter_time=100000):
        self.gridworld.clear()
        for i in range(iter_time):
            episode = self.gridworld.generate_episode()
            vis = np.zeros((self.gridworld.n, self.gridworld.m))  # for first visit, we should remove duplicated items
            total_gain = episode[-1]['gain'] + 1
            for item in episode:
                pos = item['pos']
                gain = total_gain - item['gain']
                if self.gridworld.is_terminal(pos[0], pos[1]) or vis[pos[0], pos[1]] is True:
                    continue
                vis[pos[0], pos[1]] = True
                self.gridworld.num[pos] += 1
                self.gridworld.value[pos] += (1 / self.gridworld.num[pos]) * (gain - self.gridworld.value[pos])

        print("###### First-Visit MC Method ######")
        print(self.gridworld.value)
        self.gridworld.get_policy()
        self.gridworld.draw_value('first_visit_value.png',
                                  title="First-Visit MC Method, $iter\_time=" + str(iter_time) + "$")
        self.gridworld.draw_policy('first_visit_policy.png',
                                   title="First-Visit MC Method, $iter\_time=" + str(iter_time) + "$")

    def every_visit(self, iter_time=100000):
        self.gridworld.clear()
        for i in range(iter_time):
            episode = self.gridworld.generate_episode()
            total_gain = episode[-1]['gain'] + 1
            for item in episode:
                pos = item['pos']
                gain = total_gain - item['gain']
                if self.gridworld.is_terminal(pos[0], pos[1]):
                    continue
                self.gridworld.num[pos] += 1
                self.gridworld.value[pos] += (1 / self.gridworld.num[pos]) * (gain - self.gridworld.value[pos])

        print("###### Every-Visit MC Method ######")
        print(self.gridworld.value)
        self.gridworld.get_policy()
        self.gridworld.draw_value('every_visit_value.png',
                                  title="Every-Visit MC Method, $iter\_time=" + str(iter_time) + "$")
        self.gridworld.draw_policy('every_visit_policy.png',
                                   title="Every-Visit MC Method, $iter\_time=" + str(iter_time) + "$")


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    env = MonteCarlo(shape=[6, 6], terminal=[[0, 1], [5, 5]])
    env.first_visit(iter_time=50000)
    env.every_visit(iter_time=50000)
