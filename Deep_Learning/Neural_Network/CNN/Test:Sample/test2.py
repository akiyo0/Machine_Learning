import torch as t
import torch.nn as nn
#import torchsnooper
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open('/Users/caujoeng/Desktop/weka.jpg')
print(np.asarray(image).shape)
x = TF.to_tensor(image)
x.unsqueeze_(0)
plt.imshow(TF.to_pil_image(x[0]))