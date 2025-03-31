import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from sklearn.preprocessing import LabelEncoder
import os
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir,transform=None):
        self.root_dir = root_dir
        # basically csv file
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        # in that case we will need label_encoder because
        #   labels in csv file and image names don't match
        self.label_encoder = LabelEncoder()

        self.annotations.iloc[:,1] = self.label_encoder.fit_transform(self.annotations.iloc[:,1])
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        img = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            img = self.transform(img)
        return img, y_label