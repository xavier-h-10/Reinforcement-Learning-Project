import gym
from DQN import DQN
import torch
import torch.nn as nn
import numpy as np

# 超参数
# batch_size = 32  # 样本数量
# LR = 0.01  # 学习率
# eps = 0.9  # greedy policy
# discount = 0.9  # reward discount
# frequency = 20  # 目标网络更新频率
# capacity = 2000  # 记忆库容量
# env = gym.make('MountainCar-v0').unwrapped  # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
# action_num = env.action_space.n
# state_num = env.observation_space.shape[0]
#
# print("env info: ", action_num, state_num)
# print(action_num)
# print(state_num)




class Net(nn.Module):
    def __init__(self, state_num, action_num):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_num, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_num)
        )

    def forward(self, x):
        return self.model(x)


class DQN(object):
    def __init__(self, action_num=3, state_num=2, learning_rate=0.01, discount=0.9, eps=0.9, capacity=2000, batch_size=32,
                 frequency=20):
        self.action_num = action_num
        self.state_num = state_num
        self.learning_rate = learning_rate
        self.eps = eps
        self.discount = discount
        self.capacity = capacity
        self.batch_size = batch_size
        self.frequency = frequency

        self.eval_net = Net(state_num, action_num)
        self.target_net = Net(state_num, action_num)
        self.num = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((capacity, state_num * 2 + 2))  # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < self.eps:  # 生成一个在[0, 1)内的随机数，如果小于eps，选择最优动作
            actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]  # 输出action的第一个数
        else:
            action = np.random.randint(0, self.action_num)  # 这里action随机等于0或1 (action_num = 2)
        return action  # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % self.capacity  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.num % self.frequency == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.num += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        idx = np.random.choice(self.capacity, self.batch_size)  # 在[0, 2000)内随机抽取32个数，可能会重复
        memory = self.memory[idx, :]  # 抽取32个索引对应的32个transition，存入memory
        state = torch.FloatTensor(memory[:, :self.state_num])
        # 将32个s抽出，转为32-bit floating point形式，并存储到state中，state为32行4列
        action = torch.LongTensor(memory[:, self.state_num:self.state_num + 1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到action中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，action为32行1列
        reward = torch.FloatTensor(memory[:, self.state_num + 1:self.state_num + 2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到state中，reward为32行1列
        state_ = torch.FloatTensor(memory[:, -self.state_num:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到state中，state_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(state).gather(1, action)
        # eval_net(state)通过评估网络输出32行每个state对应的一系列动作值，然后.gather(1, action)代表对每行对应索引action的Q值提取进行聚合
        q_next = self.target_net(state_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个state_对应的一系列动作值
        q_target = reward + self.discount * q_next.max(1)[0].view(self.batch_size, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(batch_size, 1)的形状；最终通过公式得到目标值
        loss = self.loss(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数


if __name__ == '__main__':
    capacity = 2000
    env = gym.make('MountainCar-v0').unwrapped
    action_num = env.action_space.n
    state_num = env.observation_space.shape[0]

    dqn = DQN()

    for i in range(400):  # 400个episode循环
        print('<<<<<<<<<Episode: %s' % i)
        s = env.reset()  # 重置环境
        total_reward = 0  # 初始化该循环对应的episode的总奖励
        num = 0
        while num <= 500:  # 开始一个episode (每一个循环代表一步)
            if i == 399:
                env.render()

            a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
            s_, r, done, info = env.step(a)  # 执行动作，获得反馈

            pos, vel = s_
            new_r = r

            dqn.store_transition(s, a, new_r, s_)  # 存储样本
            total_reward += new_r  # 逐步加上一个episode内每个step的reward

            s = s_  # 更新状态

            if dqn.memory_counter > capacity:
                dqn.learn()
                num += 1

            if done:
                print('episode%s---reward_sum: %s' % (i, round(total_reward, 2)))
                print(num)
                break  # 该episode结束
