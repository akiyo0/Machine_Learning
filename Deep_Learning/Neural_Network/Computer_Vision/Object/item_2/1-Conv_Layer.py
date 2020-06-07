import torch 
from torch import nn
import torch.nn.functional as F

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])

print(corr2d(X, K))

def comp_conv2d(conv2d, X):
    X = X.view((1, 1) + X.shape) # 代表批量大小和通道数为(1, 1)
    Y = conv2d(X) #这个函数需要四维输入包括(批量大小)和(通道数)
    return Y.view(Y.shape[2:])

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)

