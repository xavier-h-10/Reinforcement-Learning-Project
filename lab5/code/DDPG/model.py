import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorNet(nn.Module):
    def __init__(self, state_num=3, action_num=1, seed=2022, layer1_units=400, layer2_units=300):
        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_num, layer1_units)
        self.layer2 = nn.Linear(layer1_units, layer2_units)
        self.layer3 = nn.Linear(layer2_units, action_num)

        self.layer1.weight.data.uniform_(*hidden_init(self.layer1))
        self.layer2.weight.data.uniform_(*hidden_init(self.layer2))
        self.layer3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        # model = nn.Sequential(
        #     self.layer1,
        #     nn.ReLU(),
        #     self.layer2,
        #     nn.ReLU(),
        #     self.layer3,
        #     nn.Tanh()
        # )
        # return model
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return torch.tanh(self.layer3(x))


class CriticNet(nn.Module):
    def __init__(self, state_num=3, action_num=1, seed=2022, layer1_units=400, layer2_units=300):
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_num, layer1_units)
        self.layer2 = nn.Linear(layer1_units + action_num, layer2_units)
        self.layer3 = nn.Linear(layer2_units, 1)
        self.layer1.weight.data.uniform_(*hidden_init(self.layer1))
        self.layer2.weight.data.uniform_(*hidden_init(self.layer2))
        self.layer3.weight.data.uniform_(-3e-4, 3e-4)

    def forward(self, state, action):
        # (state,action) to Q-values
        x = F.relu(self.layer1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.layer2(x))
        return self.layer3(x)
