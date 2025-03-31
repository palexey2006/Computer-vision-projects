import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import os
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir,transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        img = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            img = self.transform(img)
        return img, y_label