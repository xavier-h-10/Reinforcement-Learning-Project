from Train import Train
import matplotlib.pyplot as plt


def draw(x, y, z, x_label, y_label, filename, ylim=None):
    plt.plot(x, label='DQN')
    plt.plot(y, label='DuelingDQN')
    plt.plot(z, label='DoubleDQN')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    plt.savefig(filename, dpi=1000)
    plt.show()


if __name__ == '__main__':
    double_dqn = Train('ddqn', iter_time=300, visual=False)
    double_dqn_step, double_dqn_time, double_dqn_avg_reward = double_dqn.train()

    dqn = Train('dqn', iter_time=300, visual=False)
    dqn_step, dqn_time, dqn_avg_reward = dqn.train()

    dueling_dqn = Train('dueling_dqn', iter_time=300, visual=False)
    dueling_dqn_step, dueling_dqn_time, dueling_dqn_avg_reward = dueling_dqn.train()

    draw(dqn_step, dueling_dqn_step, double_dqn_step, 'episode', 'step', 'step.png')
    draw(dqn_time, dueling_dqn_time, double_dqn_time, 'episode', 'time', 'time.png')
    draw(dqn_avg_reward[5:], dueling_dqn_avg_reward[5:], double_dqn_avg_reward[5:], 'episode', 'average reward',
         'avg_reward.png', ylim=[-100, 20])
