import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# two methods to deal with imbalanced datasets
# 1) Oversampling using Data augmentation
# 2) Class weighting

#Class weighting
# It is easier to do but in practice everyone usually use oversampling

# here I multiplied weight for swedish elkhound by 50,
# because there are 50 times less images of swedish elkhound breed than golder retriever breed
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,50.0]))