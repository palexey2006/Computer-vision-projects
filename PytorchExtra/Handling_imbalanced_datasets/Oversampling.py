import numpy as np
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
# two methods to deal with imbalanced datasets
# 1) Oversampling using Data augmentation
# 2) Class weighting

img_path = 'Balanced_dog_dataset_using_oversampling/Swedish elkhound/elkhound1.jpg'
img = Image.open(img_path).convert('RGB')
img = np.array(img)

transform = A.Compose([
    A.HorizontalFlip(p=0.9),
    A.RandomRotate90(p=0.2),
    A.Resize(720,1280),
    A.RandomCrop(width=640,height=640),
    A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.1,rotate_limit=25, p=0.7),
    A.GaussianBlur(blur_limit=2, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.9),
    A.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.4, hue=0.1,p=0.7),
    A.CLAHE(p=0.5),
])

for i in range(50):
    augmentation = transform(image=img)
    augmented_image = augmentation['image']
    augmented_image = transforms.ToTensor()(augmented_image)
    save_image(augmented_image, './Balanced_dog_dataset_using_oversampling/Swedish elkhound/elkhound_' + str(i) + '.jpg')