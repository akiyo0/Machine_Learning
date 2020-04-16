import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

ROOT_DIR = '/Users/caujoeng/Documents/dataset'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_DOC = 'train_anno.csv'
VAL_DOC = 'val_anno.csv'
SPECIES = ['rabbit', 'rat', 'chicken']
CLASS = ['mammal', 'bird']

class MyDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + "doesn't exist!")
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        
        image = Image.open(image_path).convert('RGB')
        label_species = int(self.file_info.iloc[idx]['species'])
        label_class = int(self.file_info.iloc[idx]['class'])

        sample = {'image':image, 'species':label_species, 'class':label_class}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample

def main():
    train_transforms = transforms.Compose([
        transforms.Resize((500,500)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    val_transforms = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor()])
    train_dataset = MyDataset(
        root_dir=ROOT_DIR + TRAIN_DIR, 
        annotations_file=ROOT_DIR + TRAIN_DOC,
        transforms=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    #####
    test_dataset = MyDataset(
        root_dir=ROOT_DIR + VAL_DIR,
        annotations_file=ROOT_DIR + VAL_DOC,
        transform=val_transforms)
    test_loader = DataLoader(dataset=test_dataset)

    data_loaders = {'train': train_loader,'val':test_loader}

    def visualize_dataset():
        print(len(train_dataset))
        idx = random.randint(0, 1000)
        sample = train_loader.dataset[idx]
        print(idx, sample['image'].shape, 
              SPECIES[sample['species']], CLASS[sample['class']])
        img = sample['image']
        plt.imshow(transforms.ToPILImage()(img))
        plt.show()

if __name__ == '__main__':
    main()

