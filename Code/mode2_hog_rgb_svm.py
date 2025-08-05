##### CW_Folder/Code/model2_hog_rgb_svm.py #####

# Mostly Inspired by DataScienceBase: https://www.datasciencebase.com/supervised-ml/algorithms/support-vector-machines/pytorch-example/
# Partially Based oy 
# Modified/created by Charlie Maguire and other sources implemented are cited in the code

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
import copy
import json
import numpy as np
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from datasets import Model_2_Dataset
from config import img_train_path, img_test_path, random_seed, batch_size
from utils import load_features_and_targets, train_SVM, validate_SVM, test_SVM, EarlyStoppingMin, EarlyStoppingMax
from class_distribution import print_class_distribution

# Random seed imported for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Setting device to use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up the scaler for the SVM hinge loss
scaler = StandardScaler()

# ======= Preparation of DataLoaders for training and validation =======
# Instantiating the dataset subclass
training_imgs, training_targets = img_train_path
dataset_train = Model_2_Dataset(training_imgs, training_targets)

# Splitting the Validation set from the Training set
training_size = int(0.8 * len(dataset_train)) # Training set is 80% of train dataset
validation_size = len(dataset_train) - training_size # Validation set is 20% of train dataset

# Randomising the splits
training_split, validation_split = random_split(dataset_train, [training_size, 
                                                                validation_size])

# Loading the features and targets from the data splits
features_train, targets_train = load_features_and_targets(training_split)
features_validation, targets_validation = load_features_and_targets(validation_split)

# Fitting the scaler on the training features only 
scaler.fit(features_train.numpy())
features_train_scaled = torch.tensor(scaler.transform(features_train.numpy()), dtype = torch.float32)
features_validation_scaled = torch.tensor(scaler.transform(features_validation.numpy()), dtype = torch.float32)

# Converting target datatypes to torch.long
targets_train = targets_train.type(torch.long)
targets_validation = targets_validation.type(torch.long)

# Structure the scaled data with TensorDataset
# Inspired by: https://campus.datacamp.com/courses/introduction-to-deep-learning-with-pytorch/training-a-neural-network-with-pytorch?ex=2
training_split_scaled = TensorDataset(features_train_scaled, targets_train)
validation_split_scaled = TensorDataset(features_validation_scaled, targets_validation)

# Applying the DataLoader for batching
dataloader_train = DataLoader(training_split_scaled, batch_size = batch_size, shuffle = True, num_workers = 0)
dataloader_validation = DataLoader(validation_split_scaled, batch_size = batch_size, num_workers = 0)

# ======== Preparation of DataLoader for testing =====================
# Instantiating the dataset subclass
test_imgs, test_targets = img_test_path
dataset_test = Model_2_Dataset(test_imgs, test_targets)

# Loading the 
features_test, targets_test = load_features_and_targets(dataset_test)

# Transforming the test features
features_test_scaled = torch.tensor(scaler.transform(features_test.numpy()), dtype = torch.float32)

targets_test = targets_test.type(torch.long)

dataset_test_scaled = TensorDataset(features_test_scaled, targets_test)

dataloader_test = DataLoader(dataset_test_scaled, batch_size = batch_size, num_workers = 0)

# ==== Checking the sizes of the modifieed datasets
print(f"Train size: {len(training_split)}")
print(f"Val size: {len(validation_split)}")
print(f"Test size: {len(dataset_test)}")

# ========= Definition of the Custom SVM Model (nn.Module subclass) ========
class SVM_ImageClassifier(nn.Module):
    def __init__(self, num_features, num_classes = 3):
        super().__init__()
        self.svm = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.svm(x)
    

# =========== Instantiate Model, Prepare Criterion and Prepare Optimiser ==========
# Incorporating 

# Instantiate model
model = SVM_ImageClassifier(num_features = features_train.shape[1])
model.to(device)

# Using MultiMarginLoss for multiclass SVM
criterion = nn.MultiMarginLoss()

# Defining the optimiser (SGD for SVM)
optimiser = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9,
                      weight_decay = 0.0) # Weight decay is zero to do L2


# ============== TRAINING LOOP ==================
# The code is based loosely based on DataScienceBase: 4. Train a model:
#   - https://www.datasciencebase.com/supervised-ml/algorithms/support-vector-machines/pytorch-example/
# And DataCamp: 

accuracy = Accuracy(task = "multiclass", num_classes = 3) # Set up micro accuracy metric
C = 1.0 # L2 Regularisation (will be tuned later)

# Training the model with reusable code
train_SVM(model, dataloader_train, criterion, optimiser, C, accuracy)

      
# ============== VALIDATION LOOP ================== 
# Inspired by DataCamp: 
#   -https://campus.datacamp.com/courses/intermediate-deep-learning-with-pytorch/training-robust-neural-networks?ex=6

# Setting up the rest of the metrics
accuracy = Accuracy(task = "multiclass", num_classes = 3)
f1_score = F1Score(task = "multiclass", num_classes = 3, average = "macro")
recall = Recall(task = "multiclass", num_classes = 3, average = "macro")   
precision = Precision(task = "multiclass", num_classes = 3, average = "macro")   

# Validating the model with reusable code
validate_SVM(model, dataloader_validation, accuracy, f1_score, precision, recall) 


# ============= HYPERPARAMETER TUNING  =============================
# Inspired by DataCamp's, Hyperparameter optimization with Optuna, within their Deep Reinforcement Learning in Python course: 
# https://campus.datacamp.com/courses/deep-reinforcement-learning-in-python/proximal-policy-optimization-and-drl-tips?ex=10

# Creating the study object
study = optuna.create_study(direction = "maximize")

# Defining the objective function for the optimisation process
def objective(trial: optuna, optuna.Trial):
    
    # Tuning C
    C = trial.suggest_float("C", 0.01, 10.0, log = True)
    lr = trial.suggest_float("lr", 0.0001, 0.1, log = True)
    
    # Instantiate SVM model
    model = SVM_ImageClassifier(num_features = features_train.shape[1])
    
    # Setting up the optimiser and loss
    optimiser = optim.SGD(model.parameters(), lr = lr)
    criterion = nn.MultiMarginLoss()
    
    # Metrics
    accuracy = Accuracy(task = "multiclass", num_classes = 3)
    f1_score = F1Score(task = "multiclass", num_classes = 3, average = "macro")
    recall = Recall(task = "multiclass", num_classes = 3, average = "macro")   
    precision = Precision(task = "multiclass", num_classes = 3, average = "macro")   

    # Training the model
    train_SVM(model, dataloader_train, criterion, optimiser, C, accuracy, epochs = 10)
    
    # Validating the model and getting the metrics for optimisation
    metrics = validate_SVM(model, dataloader_validation, accuracy, f1_score, precision, recall) 

    return metrics.item()


# ===================== TESTING LOOP ==============================
# Checking the class imbalance
print_class_distribution(training_split_scaled, "Training Set", class_targets=[0, 1, 2])
print_class_distribution(validation_split_scaled, "Validation Set", class_targets=[0, 1, 2])
print_class_distribution(dataset_test_scaled, "Test Set", class_targets=[0, 1, 2])  

# Reintantiate the metrics
accuracy = Accuracy(task = "multiclass", num_classes = 3)
f1_score = F1Score(task = "multiclass", num_classes = 3, average = "macro")
recall = Recall(task = "multiclass", num_classes = 3, average = "macro")   
precision = Precision(task = "multiclass", num_classes = 3, average = "macro") 

# Early Stopping Objects: One for loss and the other for metrics
loss_stopping = EarlyStoppingMin(patience = 5, min_delta = 0.001)
f1_stopping = EarlyStoppingMax(patience = 5, min_delta = 0.001)

# The best parameters to be used for Training loop
params_best = study.best_params

# Redo the model with the best parameter values
model_best = SVM_ImageClassifier(num_features = features_train.shape[1])
optimiser_best = optim.SGD(model_best.parameters(), lr = params_best["lr"])
C_best = params_best["C"]
criterion = nn.MultiMarginLoss() # Loss function

epoch_best = 0
f1_best = -1.0
model_state_best = None

# Retraining and revalidating the model with the training data 
for epoch in range(1000):
    train_SVM(model_best, dataloader_train, criterion, optimiser_best, C_best, accuracy, epochs = 1)
    
    # Running the validation to get the validation loss
    validation_metrics = validate_SVM(model_best, dataloader_validation, accuracy, f1_score, precision, recall)
    validation_loss = validation_metrics["loss"]
    validation_f1 = validation_metrics["macro_f1"]
    
    print(f"Epoch {epoch + 1} | Validation Loss: {validation_loss:.4f} | Macro F1: {validation_metrics}")
    
    # Then save the best model according to the F1 score
    if validation_f1 > f1_best:
        f1_best = validation_f1
        epoch_best = epoch + 1
        model_state_best = copy.deepcopy(model_best.state_dict())
    
    if loss_stopping.stop_early(validation_loss) and f1_stopping.stop_early(validation_f1):
        print(f"Early stopping done at epoch {epoch + 1}")
        break

# Restoring the best weights 
model_best.load_state_dict(model_state_best)

# Changing to evaluation mode before testing
model_best.eval()

# Saving the model weights
torch.save(model_state_best, "../Models/best_model2_hog_svm.pt")

# Saving the best parameters to the Models folder
with open("../Models/best_param_model2_hog_svm.json", "w") as file:
    json.dump(params_best, file, indent = 4)

# Testing the model with reusable code
test_SVM(model_best, dataloader_test, accuracy, f1_score, precision, recall) 
