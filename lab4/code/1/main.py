import torch
import torch.nn as nn
import numpy as np
import gym
from Net import Net

# 超参数
batch_size = 32  # 样本数量
LR = 0.01  # 学习率
eps = 0.9  # greedy policy
discount = 0.9  # reward discount
update_iter = 20  # 目标网络更新频率
memory_size = 2000  # 记忆库容量
env = gym.make('MountainCar-v0').unwrapped
action_num = env.action_space.n
state_action = env.observation_space.shape[0]

print(action_num)
print(state_action)


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):  # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((memory_size, state_action * 2 + 2))  # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < eps:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, action_num)
        return action

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % memory_size  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % update_iter == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        idx = np.random.choice(memory_size, batch_size)  # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[idx, :]  # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :state_action])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, state_action:state_action + 1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, state_action + 1:state_action + 2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -state_action:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + discount * q_next.max(1)[0].view(batch_size, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(batch_size, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数


dqn = DQN()  # 令dqn=DQN类

for i in range(400):  # 400个episode循环
    print('<<<<<<<<<Episode: %s' % i)
    s = env.reset()  # 重置环境
    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
    num = 0
    while True:  # 开始一个episode (每一个循环代表一步)
        if i == 30:
            env.render()  # 显示实验动画

        a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)  # 执行动作，获得反馈

        pos, vel = s_
        new_r = 0
        if -0.4 < s_[0] < 0.5:
            new_r = 10 * (s_[0] + 0.4) ** 3
        elif s_[0] >= 0.5:
            new_r = 100
        elif s_[0] <= -0.4:
            new_r = -0.1

        dqn.store_transition(s, a, new_r, s_)  # 存储样本
        episode_reward_sum += new_r  # 逐步加上一个episode内每个step的reward

        s = s_

        if dqn.memory_counter > memory_size:  # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.learn()
            num += 1

        if done:  # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            print(num)
            break  # 该episode结束
