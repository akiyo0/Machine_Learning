import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # nn.Module(单元)子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #输入图片为单通道,6输出,卷积核5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层 y=Wx+b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84 ,10)
    
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape '-1'表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()
print(net)