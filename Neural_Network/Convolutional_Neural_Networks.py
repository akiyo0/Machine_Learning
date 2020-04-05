import torch as t
import torch.nn as nn
import torchsnooper
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        # nn.Module(单元)子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #输入图片为单通道,6输出,卷积核5
        #self.add_module("conv1", nn.Conv2d(1, 6, 5)) #等价于
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层 y=Wx+b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84 ,10)
    
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 数据降维 reshape '-1'表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

#@torchsnooper.snoop()
def main():
    net = Net()
    print(net)

    input = Variable(t.randn(1, 1, 32, 32))
    output = net(input)

    net.zero_grad()
    output.backward(Variable(t.ones(1, 10)))

    output = net(input)
    target = Variable(t.arange(0, 10, dtype=t.float32))
    #target = Variable(t.arange(0, 10))
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(loss)

    print("********")
    net.zero_grad()
    print(net.conv1.bias.grad)
    loss.backward()
    print(net.conv1.bias.grad)

    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)
    
    # 新建一个优化器，指定要调整的参数和学习率
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad() # 在训练过程中 先梯度清零
    output = net(input) 
    loss = criterion(output, target) # 计算损失
    loss.backward() # 反向传播
    optimizer.step() # 更新参数

    output = net(input) 
    loss = criterion(output, target)

    print(loss)

if __name__ == '__main__':
    main()


# https://xmfbit.github.io/2017/02/25/pytorch-tutor-01/