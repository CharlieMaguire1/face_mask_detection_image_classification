##### CW_Folder/Code/utils.py #####

# Modified by: Charlie Maguire with any incorporation of sources listed in the code

# Libraries needed:
import os
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics import Accuracy, F1Score, Precision, Recall


# Inspired by: 
#    Python documentation for os.listdir(): https://docs.python.org/3/library/os.html#os.listdir
#    Stack Overflow: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
def get_img_paths_and_targets(img_path_dir, target_path_dir):
    """
    This function takes directory paths for images and target labels
    And returns the individual image paths and the read integer target labels 
    """
    img_paths = [] # Start with an empty list
    targets = [] # Empty list for the labels/targets
    
    for filename in sorted(os.listdir(img_path_dir)): # Iterate through the image folder
        if filename.lower().endswith((".jpeg", ".jpg", "png")): # Set conditions for if statement
            img_path = os.path.join(img_path_dir, filename) # 
            target_filename = os.path.splitext(filename)[0] + ".txt" # Splitting filename and adding text file extension
            target_path = os.path.join(target_path_dir, target_filename)
            
            # Opening and reading the target labels
            try:
                with open(target_path, "r") as file: # Opening target label in .txt file
                    target = int(file.read().strip()) # Reading target as an integer
            except FileNotFoundError: # Print an error if target file is not found
                print("Target txt file not found")
                continue # Skip to the next file if found
            except ValueError: # Print an error if the target inside the file is invalid 
                print("Invalid target in file")
                continue # Skip to the next file if valid
        
            img_paths.append(img_path) # Append image path to the empty list instantiated before
            targets.append(target) # Append integer label to the empty list instantiated before
    
    return img_paths, targets

def load_features_and_targets(dataset):
    """
    This function takes a dataset (Dataset subclass instance) 
    To extract its torch tensors, which are the features and targets
    """
    features = []
    targets = []
    for feat, tar in dataset:
        features.append(feat)
        targets.append(tar)
    return torch.stack(features), torch.tensor(targets)
 
 
# ==================== EARLY STOPPING CLASSES AND METHODS =====================================

class EarlyStoppingMin:
    """
    This class will stop the training when the validation loss is not improving
    According to a number of epochs, which is patience.
    Inspired by StackOverflow code seen here:
        - https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience = 5, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta 
        self.counter = 0
        self.min_validation_loss = float("inf")
        
    def stop_early(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EarlyStoppingMax:
    """
    This class will stop the training when the validation metrics does not improving
    According to a number of epochs, which is patience.
    Inspired by StackOverflow code seen here:
        - https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience = 5, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta 
        self.counter = 0
        self.max_validation_metric = float("-inf")
        
    def stop_early(self, validation_metric):
        if validation_metric > self.max_validation_metric + self.min_delta:
            self.max_validation_metric = validation_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
            

 
 
# ===================== SUPPORT VECTOR MACHINE FUNCTIONS==============================================  
def train_SVM(model: Module , dataloader_train: DataLoader, criterion: Module, 
              optimiser: Optimizer, C: float, accuracy: Accuracy, epochs: int = 1000):
    """
    This function trains a custom SVM Model and prints the loss and accuracy for every epoch
    The code is based loosely based on DataScienceBase: 4. Train a model:
      - https://www.datasciencebase.com/supervised-ml/algorithms/support-vector-machines/pytorch-example/
    And DataCamp: 
    
    Args:
        model (torch.nn.Module): Custom SVM PyTorch model
        dataloader_train (torch.utils.data.DataLoader): Enables batching
        criterion (torch.nn.Module): Loss function (MultiMargin was chosen)
        optimiser (torch.optim.Optimizer): Optimisation algorithm (SGD was chosen)
        C (float): L2 Regularisation strength
        accuracy : Torchmetrics accuracy
        epochs (int, optional): Number of epochs for training. Defaults to 1000.

    Returns:
        A tuple of two floats: The final training loss and the final training average
    """
    model.train()
    for epoch in range(epochs):
        accuracy.reset()
        loss_total = 0
        for features, targets in dataloader_train:
            optimiser.zero_grad()
            outputs = model(features)
            
            # Applying L2 Regularisation manually
            l2_reg = C * torch.norm(model.svm.weight) ** 2
            
            loss = criterion(outputs, targets) + l2_reg # Calculate losses
            loss.backward() # Backpropagation
            optimiser.step() # Update the weights
            
            loss_total += loss.item() # Accumulate the loss
            
            predictions = torch.argmax(outputs, dim = 1)
            accuracy.update(predictions, targets) # Update the accuracy progression
            
        loss_average = loss_total/len (dataloader_train)    
        accuracy_train = accuracy.compute()
        print(f"Epoch {epoch+1}, Loss: {loss_average:.4f}, Train Accuracy: {accuracy_train:.4f}") 
        
    return loss_average, accuracy_train      
                    
def validate_SVM(model: Module, dataloader_validation: DataLoader, accuracy: Accuracy, 
                 f1_score: F1Score, precision: Precision, recall: Recall):
    """
    This function validates a custom SVM Model 
    And prints the accuracy, F1, precision, and recall.
    Inspired by DataCamp's:
     -https://campus.datacamp.com/courses/intermediate-deep-learning-with-pytorch/training-robust-neural-networks?ex=6
    
    Args:
        model (torch.nn.Module): Trained SVM PyTorch model
        dataloader_validation (torch.utils.data.DataLoader): The validation data split
        accuracy: Torchmetrics Accuracy 
        f1_score: Torchmetrics F1 Score
        precision: Torchmetrics Precision
        recall: Torchmetrics Recall

    Returns:
        A tuple of the four final validation metrics (four floats)
    """
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
    
    return accuracy_validation, f1_score_validation, precision_validation, recall_validation    
      
                    
def test_SVM(model: Module, dataloader_test: DataLoader, accuracy: Accuracy, f1_score: F1Score,
             precision: Precision, recall: Recall):
    """
    This function evaluates a trained SVM PyTorch model on the test split (data) 
    And prints out the accuracy, F1 score, precision, and recall
    Inspired by DataCamp's:
     -https://campus.datacamp.com/courses/intermediate-deep-learning-with-pytorch/training-robust-neural-networks?ex=6

    Args:
        model (torch.nn.Module): Trained SVM PyTorch model
        dataloader_test (torch.utils.data.DataLoader): The test data split
        accuracy: Torchmetrics Accuracy 
        f1_score: Torchmetrics F1 Score
        precision: Torchmetrics Precision
        recall: Torchmetrics Recall
        
    Returns:
        A tuple of the four final test metrics (four floats)
    """
    # Change to evaluation mode
    model.eval()

    # Reset the metrics
    accuracy.reset()
    f1_score.reset()
    precision.reset()
    recall.reset()
    
    with torch.no_grad():
        for features, targets in dataloader_test:
            outputs = model(features)
            predictions = torch.argmax(outputs, dim = 1)
            
            # Update the metrics
            accuracy.update(predictions, targets)
            f1_score.update(predictions, targets)
            precision.update(predictions, targets)
            recall.update(predictions, targets)

    # Computing the metrics 
    accuracy_test = accuracy.compute()
    f1_score_test = f1_score.compute()
    precision_test = precision.compute()
    recall_test = recall.compute()
    
    # Printing the metrics
    print(f"Test accuracy: {accuracy_test:.4f}")
    print(f"Test F1 Score: {f1_score_test:.4f}")
    print(f"Test precision: {precision_test:.4f}")
    print(f"Test recall: {recall_test:.4f}")   
    
    return accuracy_test, f1_score_test, precision_test, recall_test 
    
   
    
    