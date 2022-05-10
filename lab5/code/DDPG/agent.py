import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim


from buffer import ReplayBuffer
from model import ActorNet, CriticNet
from noise import OUNoise

# Reference: https://arxiv.org/pdf/1509.02971.pdf
buffer_size = 1000000
batch_size = 64
lr_actor = 1e-4
lr_critic = 1e-3
weight_decay = 0.01
gamma = 0.99
tau = 0.001  # soft update parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():

    def __init__(self, state_num, action_num, seed):

        self.state_num = state_num
        self.action_num = action_num
        self.seed = random.seed(seed)

        self.actor = ActorNet(state_num, action_num, seed).to(device)
        self.actor_target = ActorNet(state_num, action_num, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = CriticNet(state_num, action_num, seed).to(device)
        self.critic_target = CriticNet(state_num, action_num, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

        self.noise = OUNoise(action_num, seed)
        self.memory = ReplayBuffer(action_num, buffer_size, batch_size, seed)
        print("current device:", device)

    # every batch size, we use experience replay memory
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > batch_size:
            records = self.memory.sample()
            self.learn(records, gamma)

    def choose_action(self, state, use_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if use_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, records, gamma):
        state, action, reward, next_state, done = records

        # update critic
        action_next = self.actor_target(next_state)
        Q_target_next = self.critic_target(next_state, action_next)
        Q_target = reward + (gamma * Q_target_next * (1 - done))
        Q_expected = self.critic(state, action)
        critic_loss = F.mse_loss(Q_expected, Q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        action_pred = self.actor(state)
        actor_loss = -self.critic(state, action_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        self.soft_update(self.critic, self.critic_target, tau)
        self.soft_update(self.actor, self.actor_target, tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
