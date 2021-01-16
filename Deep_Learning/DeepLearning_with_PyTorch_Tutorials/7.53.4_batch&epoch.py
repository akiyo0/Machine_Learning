import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loss = 0
correct = 0

criterion = nn.CrossEntropyLoss()

for data, target in test_loss:
    data = data.view(-1, 28*28)
    data, target = data.to(device)
    logits = net(data)
    test_loss += criterion(logits, target).item()

    pred = logits.argmax(dim=1)
    correct += pred.eq(target).float().sum().item()

test_loss /= len(test_loader.dataset)
