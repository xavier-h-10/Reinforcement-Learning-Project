import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, state_num, action_num):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_num, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_num)
        )

    def forward(self, x):
        return self.model(x)
