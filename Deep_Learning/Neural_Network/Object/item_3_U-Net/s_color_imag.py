from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

TRAIN_PATH = '/Volumes/TimeMachine/Github/Machine_Learning/Deep_Learning/Neural_Network/Object/item_3_U-Net/DATASET/training'
TEST_PATH = '/Volumes/TimeMachine/Github/Machine_Learning/Deep_Learning/Neural_Network/Object/item_3_U-Net/DATASET/test'

IMG_HEIGHT = IMG_WEIGHT = 128

img = imread(TRAIN_PATH + '/images/' + "21_training.tif")[:,:,:3]
plt.imshow(img)
plt.show()

img = resize(img, (IMG_HEIGHT, IMG_WEIGHT), mode='constant', preserve_range=True)
mask = imread(TRAIN_PATH + '/mask/' +  "21_training_mask.gif" )
img = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WEIGHT), mode='constant', preserve_range=True), axis=-1)

print(img)

plt.imshow(img, cmap='gray')
plt.show()