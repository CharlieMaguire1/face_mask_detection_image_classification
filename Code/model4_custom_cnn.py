##### CW_Folder/Code/model4_custom_cnn.py #####

# Partially inspired by DataCamp: https://app.datacamp.com/learn/courses/intermediate-deep-learning-with-pytorch
# Modified by Charlie Maguire and other sources implemented are cited in the code

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Accuracy, F1Score 
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import Model_4_Dataset
from config import img_train_path, img_test_path, random_seed

# Random seed imported for reproducibility
torch.manual_seed(random_seed)

