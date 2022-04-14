import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, state_action=2, action_num=3):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_action, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_num)
        )

    def forward(self, x):
        return self.model(x)
