import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.conv4 = nn.Conv2d(64, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 128, (3, 3))
        self.conv6 = nn.Conv2d(128, 128, (3, 3))
        self.conv7 = nn.Conv2d(128, 64, (3, 3))
        self.conv8 = nn.Conv2d(64, 64, (3, 3))
        self.conv9 = nn.Conv2d(64, 32, (3, 3))
        self.conv10 = nn.Conv2d(32, 32, (3, 3))
        self.conv11 = nn.Conv2d(32, 2, (3, 3))
        
        self.dropout = nn.Dropout2d(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        poolx = self.maxpool(x)
        #2
        y = F.relu(self.conv3(poolx))
        y = self.dropout(y)
        y = F.relu(self.conv4(y))
        pooly = self.maxpool(y)
        #3
        z = F.relu(self.conv5(pooly))
        z = self.dropout(z)
        z = F.relu(self.conv6(z))
        #4
        p = self.upsample(z)
        p = torch.cat((y, p), 0)
        p = F.relu(self.conv7(p))
        p = self.dropout(p)
        p = F.relu(self.conv8(p))
        #5
        q = self.upsample(p)
        q = torch.cat((x, q), 0)
        q = F.relu(self.conv9(q))
        q = self.dropout(q)
        q = F.relu(self.conv10(q))
        #6
        r = F.relu(self.conv11(q))
        r = r.reshape() #???
        r = r.permute(1, 0)

