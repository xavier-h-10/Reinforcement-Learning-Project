import numpy as np


class GridEnv:
    def __init__(self, shape=None, target=None):
        if not len(shape) == 2:
            raise ValueError('Shape argument must be a list/tuple of length 2')

        # Declaration
        self.shape = shape
        self.X = shape[0]
        self.Y = shape[1]
        self.table = np.zeros((self.X, self.Y))
        self.target = target

        self.actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.prob = np.array([0.25, 0.25, 0.25, 0.25])
        self.actions_num = 4

    def calc_num(self, x=0, y=0):
        return x * (self.X - 1) + self.Y

    def is_terminal(self, state):
        if not len(state) == 2:
            raise ValueError('state argument must be a list of 2')
        num = self.calc_num(state[0], state[1])
        if num in self.target:
            return True
        else:
            return False






