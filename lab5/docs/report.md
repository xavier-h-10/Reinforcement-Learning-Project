# Report for Assignment 5

### 519021910913 黄喆敏

### Directory Structure

````
lab4
├── code
│   ├── DDQN.py           			#  Double DQN model
│   ├── DQN.py            			#  DQN model
│   ├── DuelingDQN.py    			  #  Dueling DQN model
│   ├── Train.py                #  call DQN model to train
│   └── main.py                 #  external interface
└── docs
    ├── assets
    │   ├── avg_reward.png
    │   ├── double_dqn.png
    │   ├── dqn.png
    │   ├── dueling_dqn.png
    │   ├── step.png
    │   └── time.png
    └── report.md
````

All codes are placed in `./code` directory. You can directly run `main.py` to see the results of three DQN-based models.

 If you want to visualize the result of `MountainCar`, modify `self.visual` to `true`.

The package requirement is as follows:

- PyTorch 1.11.0. We don't have to install `cuda` , because the neural network is not very deep and we don't need to use GPU.
- Python >= 3.7.0
- gym <= 0.22.0. 
- pygame 2.1.2. If you want to visualize the result of `MountainCar`, please install it.



### Introduction





### DDPG



- Optimizer: Adam

- Soft target updates: 

I have also noted that the author used **Ornstein-Uhlenbeck Process** to generate noise. From the author's perspective, it can generate temporally correlated noise in order to explore well in physical environments that have momentum.



### SAC





















### References

[<a name="ref1">[1]</a>] https://zhuanlan.zhihu.com/p/54670989

