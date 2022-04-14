import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, N_STATES=2, N_ACTIONS=3):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(N_STATES, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, N_ACTIONS)
        )

    def forward(self, x):
        return self.model(x)
