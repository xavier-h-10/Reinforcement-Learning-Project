import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
from Net import ActorNet, CriticNet
from buffer import ReplayBuffer
from noise import OUActionNoise

#####################  hyper parameters  ####################
EPISODES = 200
EP_STEPS = 200
lr_actor = 0.001
LR_CRITIC = 0.002
gamma = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
RENDER = True
ENV_NAME = 'Pendulum-v1'

device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')


class DDPG(object):
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=100000, fc1_dims=400, fc2_dims=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNet(alpha, input_dims, fc1_dims, fc2_dims, n_actions=n_actions)
        self.critic = CriticNet(beta, input_dims, fc1_dims, fc2_dims, n_actions=n_actions)
        # off-policy
        self.target_actor = ActorNet(alpha, input_dims, fc1_dims, fc2_dims, n_actions=n_actions)
        self.target_critic = CriticNet(beta, input_dims, fc1_dims, fc2_dims, n_actions=n_actions)

        self.update_network_parameters(tau=1)

    # at = mu + Nt
    def choose_action(self, s):
        # put the actor into the evaluation mode
        # 前面使用了batch normalization, 这里要使用eval()
        self.actor.eval()
        s = T.tensor(s, dtype=T.float).to(device)
        mu = self.actor.forward(s).to(device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(device)
        self.actor.train()

        return mu_prime

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # sample a random minibatch of N transitions from R
        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        done = T.tensor(done).to(device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        # yi = ri + gamma * Q'
        target = []
        for j in range(self.batch_size):
            target.append(rewards[j] + self.gamma * critic_value_[j] * done[j])
        target = T.tensor(target).to(device)
        target = target.view(self.batch_size, 1)

        # update critic by minimizing the loss
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # update the actor
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # update target networks
        self.update_network_parameters()

    # theta^Q' = tau * theta^Q + (1-tau) * theta^Q'
    # theta^mu' = tau * theta^mu + (1-tau) * theta^mu'
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # get all the names of the parameters
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
