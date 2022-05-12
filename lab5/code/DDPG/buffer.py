from collections import deque, namedtuple
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:

    def __init__(self, action_num, buffer_size, batch_size, seed):
        self.action_num = action_num
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.record = namedtuple("Reward", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.record(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        records = random.sample(self.memory, k=self.batch_size)

        state = torch.from_numpy(np.vstack([e.state for e in records if e is not None])).float().to(device)
        action = torch.from_numpy(np.vstack([e.action for e in records if e is not None])).float().to(device)
        reward = torch.from_numpy(np.vstack([e.reward for e in records if e is not None])).float().to(device)
        next_state = torch.from_numpy(np.vstack([e.next_state for e in records if e is not None])).float().to(
            device)
        done = torch.from_numpy(np.vstack([e.done for e in records if e is not None]).astype(np.uint8)).float().to(
            device)

        return (state, action, reward, next_state, done)

    def __len__(self):
        return len(self.memory)