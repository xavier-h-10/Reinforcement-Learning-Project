import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor


class Net(nn.Module):
    def __init__(self, state_num=2, action_num=3):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_num, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_num)
        )

    def forward(self, x):
        return self.model(x)


class DDQN:
    def __init__(self, action_num=3, state_num=2, memory_size=2048, lr=0.01, eps=0.9, update_iter=25, batch_size=32,
                 discount=0.9):
        self.eval_net, self.target_net = Net(), Net()
        self.num = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_size, state_num * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.action_num = action_num
        self.state_num = state_num
        self.eps = eps
        self.memory_size = memory_size
        self.update_iter = update_iter
        self.batch_size = batch_size
        self.discount = discount

    def move(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # with probability 1-eps select a random action, otherwise a_t=argmax
        if np.random.uniform() < self.eps:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def save_buffer(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # every C step reset
        if self.num % self.update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.num += 1

        idx = np.random.choice(self.memory_size, self.batch_size, replace=False)
        memory = self.memory[idx, :]
        state = FloatTensor(memory[:, :self.state_num])
        action = LongTensor(memory[:, self.state_num:self.state_num + 1].astype(int))
        reward = FloatTensor(memory[:, self.state_num + 1:self.state_num + 2])
        state_ = FloatTensor(memory[:, -self.state_num:])

        q_eval = self.eval_net(state).gather(1, action)
        q_next = self.target_net(state_).detach()

        # difference between DQN and DDQN
        q_eval_ = self.eval_net(state).detach()
        q_next = q_eval_.gather(1, torch.max(q_eval_, 1)[1].unsqueeze(1)).squeeze(1)
        mx = q_next.reshape(self.batch_size, 1)
        # set y_j = r_j + gamma*max
        q_target = reward + self.discount * mx
        loss = self.loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
