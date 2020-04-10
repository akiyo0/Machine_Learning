import torch as t
import torch.nn as nn
from torch import optim
import torchvision as tv
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    show = ToPILImage()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) #归一化
        )
    ])

    root = '/Users/caujoeng/Documents/dataset'
    trainset = tv.datasets.CIFAR10(root, train=True, download=False, transform=transform)
    trainloader = t.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = tv.datasets.CIFAR10(root, train=False, download=False, transform=transform)
    testloader = t.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes=('plane','car','bird','cat','deer','dog','frog','horse',
            'ship','truck')

    '''
    (data, label) = trainset[100]
    print(classes[label])

    plt.imshow(show((data + 1)/2).resize((100, 100)))
    plt.axis('off')
    plt.show()
    dataiter = iter(trainloader)
    images,labels = dataiter.next()
    print(' '.join('%11s'%classes[labels[j]] for j in range(4)))

    plt.imshow(show(tv.utils.make_grid((images+1)/2)).resize((400,100)))
    plt.axis('off')
    plt.show()
    '''
    
    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss() #交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = Variable(data[0]), Variable(data[1])
            print(inputs, labels)
if __name__ == '__main__':
    main()