##### CW_Folder/Code/data.py #####

import torch
from torch.utils.data import random_split, DataLoader

from datasets import Model_3_Dataset
from config import batch_size

def get_dataloaders(training_imgs, training_targets, batch_size = batch_size, val_percent = 0.2,
                    num_workers = 0):
    
    # Creating the dataset
    dataset_train = Model_3_Dataset(training_imgs, training_targets)
    
    training_size = int((1 - val_percent) * len(dataset_train))
    validation_size = len(dataset_train) - training_size
    
    training_split, validation_split = random_split(dataset_train, [
        training_size, validation_size])
    
    dataloader_train = DataLoader(training_split, batch_size = batch_size,
                                  shuffle = True, num_workers = num_workers)
    dataloader_validation = DataLoader(validation_split, batch_size = batch_size,
                                  shuffle = False, num_workers = num_workers)
    
    print(f"Training size: {len(training_split)}")
    print(f"Validation size: {len(validation_split)}")
    
    return dataloader_train, dataloader_validation