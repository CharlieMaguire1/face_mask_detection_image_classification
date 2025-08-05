##### CW_Folder/Code/model3_mobilenetv2.py #####

# Partially inspired by DataCamp: https://app.datacamp.com/learn/courses/intermediate-deep-learning-with-pytorch
# Modified by Charlie Maguire and other sources implemented are cited in the code

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchmetrics import Accuracy, F1Score 
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models

from datasets import Model_3_Dataset
from config import img_train_path, img_test_path, random_seed, batch_size
from data import get_dataloaders

# Random seed imported for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Setting device to use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
# ======= Preparation of DataLoaders for training and validation =======
training_imgs, training_targets = img_train_path

dataloader_train, dataloader_validation = get_dataloaders(training_imgs,
                                                          training_targets,
                                                          batch_size = batch_size,
                                                          val_percent = 0.2,
                                                          num_workers = 0)

# ======== Preparation of DataLoader for testing =====================
test_imgs, test_targets = img_test_path
dataset_test = Model_3_Dataset(test_imgs, test_targets)
dataloader_test = DataLoader(dataset_test, batch_size = batch_size, 
                             shuffle = False, num_workers = 0)

# ==== Checking the sizes of the modifieed datasets
print(f"Train size: {len(dataloader_train.dataset)}")
print(f"Val size: {len(dataloader_validation.dataset)}")
print(f"Test size: {len(dataset_test)}")

# Inspired by DataCamp's:
#   - https://campus.datacamp.com/courses/intermediate-deep-learning-with-pytorch/images-convolutional-neural-networks?ex=5
# And tonylin's Github page on )MobileNetV2:
#   -https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py

class MobileNetV2CNN(nn.Module):
    def __init__(self, num_classes = 3, pretrained = True):
        super().__init__()
        
        # Loading up the pretrained MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(pretrained = pretrained)
        
        # Classifier head (the last layer)
        input_features = self.backbone.classifier[1].input_features
        
        # Classifier for 3 claaes
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace = False),
            nn.Linear(input_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x) # Forward pass through MobileNetV2 backbone
                                               