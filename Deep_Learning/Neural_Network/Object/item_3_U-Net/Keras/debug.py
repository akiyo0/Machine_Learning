import os
import sys
sys.path.insert(1, '/Volumes/TimeMachine/Github/Machine_Learning/Deep_Learning/Neural_Network/Object/item_3_U-Net/Keras/lib')
dir = os.path.dirname(os.path.realpath(__file__))

import configparser
from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images

conf = configparser.RawConfigParser()
conf.read(dir + '/configuration.txt')
path_data = conf.get('data_paths', 'path_local')

train_imgs_original_url = path_data + conf.get('data_paths', 'train_imgs_original')
train_masks_url = path_data + conf.get('data_paths', 'train_groundTruth')

train_imgs_original = load_hdf5(train_imgs_original_url)
train_masks = load_hdf5(train_masks_url) #masks always the same

print(train_imgs_original_url)
print(train_masks_url)

visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train').show()  #check original imgs train