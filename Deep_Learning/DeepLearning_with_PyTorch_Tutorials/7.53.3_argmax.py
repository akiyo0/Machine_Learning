import torch
import torch.nn.functional as F

logits = torch.rand(4, 10)
label_A = logits.argmax(dim=1)
print(logits.argmax(dim=1))

pred = F.softmax(logits, dim=1) #torch.Size([4, 10])
label_B = pred.argmax(dim=1) #torch.Size([4])
print(label_B)

correct = torch.eq(label_A, label_B)

print(correct.sum().float().item()/4)
