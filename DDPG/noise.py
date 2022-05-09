import random
import numpy as np
import copy


# use Ornstein-Uhlenbeck process
class OUNoise:

    def __init__(self, size, seed=2022, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        # reset the internal state to mean (mu)
        self.state = copy.copy(self.mu)

    def sample(self):
        # update internal state and return it as a noise sample
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state