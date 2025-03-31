import cv2
import numpy as np
import albumentations as A
from torchvision.utils import save_image
from Custom_datasets import CustomDataset
from torch.utils.data import Subset
import torchvision.transforms as transforms
# List of transforms using Albumentations
transform = A.Compose([
    A.Resize(270,270),
    A.RandomCrop(224,224),
    A.Rotate(limit=25,p=0.7),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RGBShift(r_shift_limit=25,g_shift_limit=25,b_shift_limit=25,p=0.9),
    A.OneOf(
        [
            A.Blur(blur_limit=3,p=0.7),
            A.ColorJitter(brightness=0.25,p=0.3)
        ]
    ),
])

dataset = CustomDataset(csv_file='butterfly/Training_set.csv',root_dir='butterfly/train',)

indices = list(range(10))
sliced_df = Subset(dataset, indices)

img_num = 1
for image,label in sliced_df:
    image = np.array(image)
    #applying transforms for these images
    augmentations = transform(image=image)
    augmented_image = augmentations['image']
    augmented_image = transforms.ToTensor()(augmented_image)
    save_image(augmented_image, 'butterfly/Augmented_images/augmented_butterfly' + str(img_num) + '.jpg')
    img_num += 1

