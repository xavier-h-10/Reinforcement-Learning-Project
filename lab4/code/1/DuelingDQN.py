import numpy as np
import torch
import torch.nn as nn

batch_size = 32  # 样本数量
LR = 0.01  # 学习率
eps = 0.9  # greedy policy
discount = 0.9  # reward discount
update_iter = 20  # 目标网络更新频率
memory_size = 2000  # 记忆库容量


class Net(nn.Module):
    def __init__(self, state_num=2, action_num=3):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(state_num, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.advantage = nn.Linear(16, action_num)
        self.value = nn.Linear(16, 1)

    def forward(self, x):
        y = self.features(x)
        advantage = self.advantage(y)
        value = self.value(y)

        return value + advantage - advantage.mean()


class DuelingDQN:
    def __init__(self, action_num=3, state_num=2):
        self.eval_net, self.target_net = Net(), Net()
        self.num = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_size, state_num * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss = nn.MSELoss()

        self.action_num = action_num
        self.state_num = state_num

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < eps:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.num % update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.num += 1

        idx = np.random.choice(memory_size, batch_size, replace=False)  # 在[0, 2000)内随机抽取32个数，可能会重复
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
        q_target = reward + discount * q_next.max(1)[0].view(batch_size, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(batch_size, 1)的形状；最终通过公式得到目标值
        loss = self.loss(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数
