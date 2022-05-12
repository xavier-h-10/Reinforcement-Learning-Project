import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from utils import v_wrap, set_init
from Adam import SharedAdam
import gym
import pygame
import matplotlib.pyplot as plt

# set hyper parameters
lr = 1e-4
env_id = 'Pendulum-v1'
# number of actions and states, can get as follows
# env.observation_space.shape[0]
# env.action_space.shape[0]
n_actions = 1
input_dims = 3
# total number of games
N_GAMES = 5000
# in one episode, the max step is 200
MAX_EP_STEP = 200
# when reaches T_MAX, update the global network
T_MAX = 5


class ACNet(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ACNet, self).__init__()

        self.gamma = gamma

        # policy and value network
        self.p1 = nn.Linear(input_dims, 200)
        self.p = nn.Linear(200, n_actions)
        self.v1 = nn.Linear(input_dims, 200)
        self.v = nn.Linear(200, 1)
        self.sigma = nn.Linear(200, n_actions)
        # important to get a better reward
        set_init([self.p1, self.p, self.v1, self.v, self.sigma])

        self.distribution = torch.distributions.Normal

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        # normalize rewards
        self.rewards.append((reward + 8.1) / 8.1)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        # p1 = F.relu(self.p1(state))
        # v1 = F.relu(self.v1(state))
        # p = self.p(p1)
        # v = self.v(v1)
        p1 = F.relu6(self.p1(state))
        v1 = F.relu6(self.v1(state))
        p = 2 * torch.tanh(self.p(p1))
        v = self.v(v1)
        sigma = F.softplus(self.sigma(p1)) + 0.001

        return p, sigma, v

    # both calc may not be good enough
    # pass the done and next_state
    def calc_reward(self, done, s_):
        v = self.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
        # if done, the reward is 0
        rwd = v * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            rwd = reward + self.gamma * rwd
            batch_return.append(rwd)

        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return

    def calc_loss(self, done, s, a, s_):
        self.train()
        returns = self.calc_reward(done, s_)

        p, sigma, v = self.forward(s)
        # v = v.squeeze()
        critic_loss = (returns - v) ** 2

        dist = self.distribution(p, sigma)
        log_probs = dist.log_prob(a)
        # entropy may affect the result a little
        # actor_loss = -log_probs * (returns - v)
        # obtain a faster convergence
        entropy = 0.5 * 0.5 * math.log(2 * math.pi) + torch.log(dist.scale)
        actor_loss = -(log_probs * (returns - v).detach() + 0.005 * entropy)

        total_loss = (critic_loss + actor_loss).mean()
        return total_loss

    def choose_action(self, s):
        # state = torch.tensor(np.array([observation]), dtype=torch.float)
        # p, sigma, _ = self.forward(state)
        self.training = False
        p, sigma, _ = self.forward(s)

        action = self.distribution(p.view(1, ).data, sigma.view(1, ).data)
        action = action.sample().numpy()
        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, gamma,
                 lr, name, global_ep_idx, global_ep_r, res_queue, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ACNet(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = "w%02i" % name
        self.episode_idx = global_ep_idx
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        # get original class, lift Epoch restriction
        self.env = gym.make(env_id).unwrapped
        self.optimizer = optimizer

    def run(self):
        # initialize thread step counter
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            s = self.env.reset()
            score = 0.
            self.local_actor_critic.clear_memory()
            for t in range(MAX_EP_STEP):
                if self.name == 'w00':
                    self.env.render()
                action = self.local_actor_critic.choose_action(v_wrap(s[None, :]))
                s_, reward, done, _ = self.env.step(action.clip(-2, 2))
                if t == MAX_EP_STEP - 1:
                    done = True

                score += reward
                self.local_actor_critic.remember(s, action, reward)

                # update the global net and assign it to the local net
                if t_step % T_MAX == 0 or done:
                    # vwrap converts array to tensor
                    loss = self.local_actor_critic.calc_loss(done, v_wrap(np.vstack(self.local_actor_critic.states)),
                                                             v_wrap(np.array(self.local_actor_critic.actions)), s_)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # calculate local gradients and push local parameters to global
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(), self.global_actor_critic.parameters()
                    ):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    # pull global parameters
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()

                    if done:
                        # protect current variable, no other thread can try to access it
                        with self.episode_idx.get_lock():
                            self.episode_idx.value += 1
                        with self.global_ep_r.get_lock():
                            if self.global_ep_r.value == 0.:
                                self.global_ep_r.value = score
                            else:
                                # makes reward stable and much easier to observe the convergence trend
                                self.global_ep_r.value = self.global_ep_r.value * 0.99 + score * 0.01
                        self.res_queue.put(self.global_ep_r.value)
                        print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % self.global_ep_r.value)
                        break

                t_step += 1
                s = s_
        self.res_queue.put(None)


if __name__ == '__main__':
    global_actor_critic = ACNet(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.95, 0.999))
    # integer value
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    workers = [Agent(global_actor_critic, optim, input_dims, n_actions, gamma=0.9,
                     lr=lr, name=i, global_ep_idx=global_ep, global_ep_r=global_ep_r, res_queue=res_queue,
                     env_id=env_id)
               for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    np.save('data.npy', res)

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
