import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    load_checkpoint,
)
from Loss import YoloLoss
seed = 123
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Hyperparams
learning_rate = 1e-7
batch_size = 4
weight_decay = 0
epochs = 1000
num_workers = 2
pin_memory = True
load_model = False
Load_model_file = 'saved_model.pth.tar'
img_dir = 'archive/images/'
labels_dir = 'archive/labels/'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img,bboxes = t(img), bboxes
        return img,bboxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (inputs, targets) in enumerate(loop):
        x, y = inputs, targets
        outputs = model(x)
        loss = loss_fn(outputs, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
    loss_fn = YoloLoss()
    if load_model:
        load_checkpoint(torch.load(Load_model_file),model,optimizer)
    train_dataset = VOCDataset(
        'archive/8examples.csv',
        transform=transform,
        img_dir=img_dir,
        label_dir=labels_dir,
    )
    test_dataset = VOCDataset(
        'archive/test.csv',
        transform=transform,
        img_dir=img_dir,
        label_dir=labels_dir,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,drop_last=True)

    for epoch in range(epochs):

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5,threshold=0.5)
        mean_avg_precision = mean_average_precision(pred_boxes, target_boxes,iou_threshold=0.5,box_format='midpoint')
        print(f'epoch: {epoch + 1}/{epochs}')
        print(f'Mean Average Precision: {mean_avg_precision}')
        if epoch % 10 == 0:
            for x, y in train_loader:
                x = x.to(device)
                for idx in range(3):
                    bboxes = cellboxes_to_boxes(model(x))
                    bboxes = non_max_suppression(bboxes[idx],iou_threshold=0.5, threshold=0.5, box_format="midpoint")
                    plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)

        train(train_loader, model, optimizer, loss_fn)

if __name__ == '__main__':
    main()
