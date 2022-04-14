from Train import Train
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dqn = Train('dqn')
    dqn_reward, dqn_time = dqn.train()

    dueling_dqn = Train('dqn')
    dueling_dqn_reward, dueling_dqn_time = dueling_dqn.train()
