import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FCM(nn.Module):
    def __init__(self):
        super(FCM, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x

learning_rate = 0.1
epochs = 10

net = FCM()
oprimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        logits = net(data)
        loss = criteon(logits, target)

        oprimizer.zero_grad()
        loss.backward()

        oprimizer.step()



