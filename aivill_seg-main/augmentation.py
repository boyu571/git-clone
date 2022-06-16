import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt



# Load Data
path = 'src/gantry_crane/'
src_image = path + '/img'
save_aug_path = path + '/augmentation'

file_names = os.listdir(src_image)

file_path = []
for file in file_names :
    filepath = src_image + '/' + file
    file_path.append(filepath)


# Augmentation
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1000,800), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((800,600)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.4),
    transforms.ToTensor(),
    ])


# Make list of augmented images
aug_img = []
for i in file_path :
    img = plt.imread(i)
    aug_img.append(my_transforms(img))


# Save augmented images
img_num = 1
for img in aug_img :
    save_image(img, save_aug_path + '/img'+str(img_num)+'.jpg')
    img_num +=1