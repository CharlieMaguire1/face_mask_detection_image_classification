##### CW_Folder/Code/model1_sift_bovw_mlp.py #####

# Partially inspired by DataCamp: https://app.datacamp.com/learn/courses/intermediate-deep-learning-with-pytorch
# Modified by Charlie Maguire and other sources implemented are cited in the code

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import optuna
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.utils.data import Dataset, DataLoader, random_split
from optuna import Trial

from datasets import Model_1_Dataset
from config import img_train_path, img_test_path, random_seed, batch_size
from data_functions import get_dataloaders
from utils import train_NN, evaluate_NN, print_metrics

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
dataset_test = Model_1_Dataset(test_imgs, test_targets)
dataloader_test = DataLoader(dataset_test, batch_size = batch_size, 
                             shuffle = False, num_workers = 0)

# ==== Checking the sizes of the modifieed datasets
print(f"Train size: {len(dataloader_train.dataset)}")
print(f"Val size: {len(dataloader_validation.dataset)}")
print(f"Test size: {len(dataset_test)}")

# ========= Definition of the 3 layer MLP (nn.Module subclass) ========
# Creating a custom neural netwrok subclass
class SIFT_MLPNet(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(200 , 128) # Input from the 200 clusters used by BoVW vocab
        self.fc2 = nn.Linear(128, 64) # 1st hidden layer to 2nd hidden layer
        self.out = nn.Linear(64, 3) # The output is 3 classes 
        
        # He/ Kaiming initialisation to stabilise gradients
        init.kaiming_uniform_(self.fc1.weight, nonlinearity= "relu") 
        init.kaiming_uniform_(self.fc2.weight, nonlinearity = "relu")
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x) # Output is raw logits so no softmax is used here
        return x
 
    
# ============ Instantiate Model, Loss Function and Prepare Optimiser ==========
# Instantiate the model
model = SIFT_MLPNet().to(device)

# Set up the criterion for losses
criterion = nn.CrossEntropyLoss()

# Set up the optimiser
optimiser = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0005)

# ================= Training Loop =============
accuracy = Accuracy(task = "multiclass", num_classes = 3).to(device)

for epoch in range(1000):
    loss_average, accuracy_train = train_NN(model = model, dataloader_train = dataloader_train,
                                            criterion = criterion, optimiser = optimiser,
                                            accuracy = accuracy, device = device)
    
    print(f"Epoch {epoch+1}, Loss: {loss_average:.4f}, Train Accuracy: {accuracy_train:.4f}")
        
        
# ============== EVALUATION (VALIDATION) LOOP ================
accuracy = Accuracy(task = "multiclass", num_classes = 3)
f1_score = F1Score(task = "multiclass", num_classes = 3, average = "macro").to(device)
precision = Precision(task = "multiclass", num_classes = 3, average = "macro").to(device)
recall = Recall(task = "multiclass", num_classes = 3, average = "macro").to(device)

accuracy_validation, f1_score_validation, precision_validation, recall_validation = evaluate_NN(
    model = model, dataloader = dataloader_validation, accuracy = accuracy, f1_score = f1_score,
    precision = precision, recall = recall, device = device
)

print_metrics("Validation", accuracy_validation, f1_score_validation, precision_validation,
              recall_validation)

# ============= HYPERPARAMETER TUNING  =============================
# Inspired by DataCamp's, Hyperparameter optimization with Optuna, within their Deep Reinforcement Learning in Python course: 
# https://campus.datacamp.com/courses/deep-reinforcement-learning-in-python/proximal-policy-optimization-and-drl-tips?ex=10

# Creating the study object
study = optuna.create_study(direction = "maximize")

# Defining the objective function
def objective(trial: Trial):
    
    lr = trial.suggest_float("lr", 0.0001, 0.1, log = True)
    weight_decay = trial_suggest_float("weight_decay", 0.000001, 0.01, log = True)
    
    # Instantiating the MLP model 
    model = SIFT_MLPNet().to(device)
    
    # Setting up the optimiser and loss
    optimiser = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics
    accuracy = Accuracy(task = "multiclass", num_classes = 3).to(device)
    f1_score = F1Score(task = "multiclass", num_classes = 3, average = "macro").to(device)
    precision = Precision(task = "multiclass", num_classes = 3, average = "macro").to(device)
    recall = Recall(task = "multiclass", num_classes = 3, average = "macro").to(device)   
    

    # Training the model
    for epoch in range(10):
    loss_average, accuracy_train = train_NN(model = model, dataloader_train = dataloader_train,
                                            criterion = criterion, optimiser = optimiser,
                                            accuracy = accuracy, device = device)
    
    # Evaluate the model:
    accuracy_validation, f1_score_validation, precision_validation, recall_validation = evaluate_NN(
    model = model, dataloader = dataloader_validation, accuracy = accuracy, f1_score = f1_score,
    precision = precision, recall = recall, device = device)
    
    return f1_score_validation.item()

# Running the Optuna tuning






# ================== EVALUATION (TESTING) LOOP ====================
accuracy = Accuracy(task = "multiclass", num_classes = 3)
f1_score = F1Score(task = "multiclass", num_classes = 3, average = "macro").to(device)
precision = Precision(task = "multiclass", num_classes = 3, average = "macro").to(device)
recall = Precision(task = "multiclass", num_classes = 3, average = "macro").to(device)

accuracy_test, f1_score_test, precision_test, recall_test = evaluate_NN(
    model = model, dataloader = dataloader_test, accuracy = accuracy, f1_score = f1_score,
    precision = precision, recall = recall, device = device
)

print_metrics("Test", accuracy_test, f1_score_test, precision_test,
              recall_test)


