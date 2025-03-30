from Custom_datasets import CustomDataset
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(270),
    transforms.RandomCrop(224),
    transforms.ColorJitter(0.3),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1,0.1,0.1],std=[0.8,0.8,0.8])

])
dataset = CustomDataset(csv_file='Cats_dogs_dataset/cats_dogs.csv',root_dir='Cats_dogs_dataset/cats_dogs_resized' ,transform=my_transforms)
img_num = 0
for img,label in dataset:
    save_image(img, 'Resized/resized' + str(img_num) + '.png')
    img_num += 1