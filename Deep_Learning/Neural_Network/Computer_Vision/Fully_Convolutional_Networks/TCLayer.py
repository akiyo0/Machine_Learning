import torch
import torch.nn as nn

###########
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
###########

a = torch.ones((1, 1, 2, 2))

conv = nn.ConvTranspose2d(1, 1, 3, 1, 0, bias=False)

ones = torch.ones((1, 1, 3, 3))

print(a)

conv.weight = nn.Parameter(ones)

out = conv(a)
print(out)
ax = sns.heatmap(np.array(out.tolist()).reshape(4, 4), center=0, fmt="d",cmap='YlGnBu')
plt.show()
#print(out)