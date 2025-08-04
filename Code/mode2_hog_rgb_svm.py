##### CW_Folder/Code/model2_hog_rgb_svm.py #####

# Mostly Inspired by DataScienceBase: https://www.datasciencebase.com/supervised-ml/algorithms/support-vector-machines/pytorch-example/
# Partially Based oy 
# Modified/created by Charlie Maguire and other sources implemented are cited in the code

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Accuracy, F1Score 
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import Model_2_Dataset
from config import img_train_path, img_test_path, random_seed

# Random seed imported for reproducibility
torch.manual_seed(random_seed)

# ======= Preparation of DataLoaders for training and validation =======
training_imgs, training_targets = img_train_path
dataset_train = Model_2_Dataset(training_imgs, training_targets)

training_size = int(0.8 * len(dataset_train)) # Training set is 80% of train dataset
validation_size = len(dataset_train) - training_size # Validation set is 20% of train dataset

training_split, validation_split = random_split(dataset_train, [training_size, 
                                                                validation_size])

dataloader_train = DataLoader(training_split, batch_size = 32, shuffle = True, num_workers = 4)
dataloader_validation = DataLoader(validation_split, batch_size = 32, num_workers = 4)

# ======== Preparation of DataLoader for testing =====================
test_imgs, test_targets = img_test_path
dataset_test = Model_2_Dataset(test_imgs, test_targets)

dataloader_test = DataLoader(dataset_test, batch_size = 32, num_workers = 4)

# ==== Checking the sizes of the modifieed datasets
print(f"Train size: {len(training_split)}")
print(f"Val size: {len(validation_split)}")
print(f"Test size: {len(dataset_test)}")

# ========= Definition of the Custom SVM Model (nn.Module subclass) ========
class SVM_ImageClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.svm = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.svm(x)
    
# ========== Hinge Loss and SGD ========== 
# This is the loss function for the custom SVM
# The code is based on DataScienceBase: 3. Define the Hinge Loss and Optimizer:
#   - https://www.datasciencebase.com/supervised-ml/algorithms/support-vector-machines/pytorch-example/
def hinge_loss(targets, predictions):
    """
    This function is about maximising the margin
    Hinge loss: max(0, 1 - y * f(x))
    Targets should be -1 or 1
    Predictions are the outputs of the SVM model
    """
    return torch.mean(torch.clamp(1 - targets * predictions, min = 0))


# =========== Instantiate Model, Loss and Prepare Optimiser ==========
model = SVM_ImageClassifier()
criterion = hinge_loss()
optimiser = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9,
                      weight_decay = 0.0005 ) # Defining the optimiser (SGD for SVM)


# ========= TRAINING LOOP ============ 
epochs = 1000
C = 1.0 # Regularisation for L2 Normalisation

for epoch in range(epochs)