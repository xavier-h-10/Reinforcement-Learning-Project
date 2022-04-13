class Net(nn.Module):
    def __init__(self):  # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()  # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 24)  # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

        self.fc2 = nn.Linear(24, 24)

        self.out = nn.Linear(24, N_ACTIONS)  # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):  # 定义forward函数 (x为状态)
        y = F.relu(self.fc1(x))  # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        z = F.relu(self.fc2(y))
        actions_value = self.out(z)  # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value  # 返回动作值