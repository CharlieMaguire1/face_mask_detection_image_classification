##### CW_Folder/Code/model1_sift_bovw_mlp.py #####

# Partially inspired by DataCamp: https://app.datacamp.com/learn/courses/intermediate-deep-learning-with-pytorch
# Modified by Charlie Maguire and other sources implemented are cited in the code

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import Model_1_Dataset
from config import img_train_path, img_test_path, random_seed

# Random seed imported for reproducibility
torch.manual_seed(random_seed)

# ======= Preparation of DataLoaders for training and validation =======
training_imgs, training_targets = img_train_path
dataset_train = Model_1_Dataset(training_imgs, training_targets)

training_size = int(0.8 * len(dataset_train)) # Training set is 80% of train dataset
validation_size = len(dataset_train) - training_size # Validation set is 20% of train dataset

training_split, validation_split = random_split(dataset_train, [training_size, 
                                                                validation_size])

dataloader_train = DataLoader(training_split, batch_size = 32, shuffle = True, num_workers = 4)
dataloader_validation = DataLoader(validation_split, batch_size = 32, num_workers = 4)

# ======== Preparation of DataLoader for testing =====================
test_imgs, test_targets = img_test_path
dataset_test = Model_1_Dataset(test_imgs, test_targets)

dataloader_test = DataLoader(dataset_test, batch_size = 32, num_workers = 4)

# ==== Checking the sizes of the modifieed datasets
print(f"Train size: {len(training_split)}")
print(f"Val size: {len(validation_split)}")
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
model = SIFT_MLPNet()

# Set up the criterion for losses
criterion = nn.CrossEntropyLoss()

# Set up the optimiser
optimiser = optim.Adam(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)

# ================= Training Loop =============
accuracy = Accuracy(task = "multiclass", num_classes = 3)

for epoch in range(1000):
    model.train()
    accuracy.reset()
    loss_total = 0
    for features, targets in dataloader_train:
        optimiser.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()
        
        loss_total += loss.item()
        
        predictions = torch.argmax(outputs, dim = 1)
        accuracy.update(predictions, targets)
        
    loss_average = loss_total / len(dataloader_train)   
    accuracy_train = accuracy.compute()
    print(f"Epoch {epoch+1}, Loss: {loss_average:.4f}, Train Accuracy: {accuracy_train:.4f}")
        
        
# ========== Evaluation Loop (Validation) ===========
f1_score = F1Score(task = "multiclass", num_classes = 3, average = "macro")
precision = Precision(task = "multiclass", num_classes = 3, average = "macro")
recall = Precision(task = "multiclass", num_classes = 3, average = "macro")

# Change to evaluation mode
model.eval()

# Reset the metrics
accuracy.reset()
f1_score.reset()
precision.reset()
recall.reset()

# Evaluation loop
with torch.no_grad():
    for features, targets in dataloader_validation:
        outputs = model(features)
        predictions = torch.argmax(outputs, dim = 1)
        
        # Update the metrics
        accuracy.update(predictions, targets)
        f1_score.update(predictions, targets)
        precision.update(predictions, targets)
        recall.update(predictions, targets)

# Computing the metrics 
accuracy_validation = accuracy.compute()
f1_score_validation = f1_score.compute()
precision_validation = precision.compute()
recall_validation = recall.compute()

# Printing the metrics
print(f"Validation accuracy: {accuracy_validation:.4f}")
print(f"Validation F1 Score: {f1_score_validation:.4f}")
print(f"Validation precision: {precision_validation:.4f}")
print(f"Validation recall: {recall_validation:.4f}")