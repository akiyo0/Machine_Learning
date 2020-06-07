import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

voc_dir = "./VOC2012"
train_features, train_labels = d2l.read_voc_images(voc_dir, max_num=100)

n = 5
imgs = train_features[0:n] + train_labels[0:n]
d2l.show_images(imgs, 2, n)

colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
for i, colormap in enumerate(d2l.VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

y = d2l.voc_label_indices(train_labels[0], colormap2label)
y[105:115, 130:140], d2l.VOC_CLASSES[1]

imgs = []
for _ in range(n):
    imgs += d2l.voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)

crop_size = (320, 480)
max_num = 100
voc_train = d2l.VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
voc_test = d2l.VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)

batch_size = 64
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True,
                             num_workers=num_workers)

for X, Y in train_iter:
    print(X.dtype, X.shape)
    print(y.dtype, Y.shape)
    break