##### CW_Folder/Code/dataset.py #####

# Mostly inspired by DataCamp: https://app.datacamp.com/learn/courses/intermediate-deep-learning-with-pytorch
# Created/modified by: Charlie Maguire with the incorporation of multiple sources listed below
# This file is for reproducible Dataset subclasses 

# Libraries needed
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

from transforms import compute_bovw_histogram, compute_hog_colour_histograms, model_transforms # Importing functions from transforms.py

# Dataset subclasses in preparation for DataLoaders/Batching in the model python files
# All subclases here are based on DataCamp's Training Robust Neural Networks in their Intermediate Deep Learning with PyTorch course:
#   -https://campus.datacamp.com/courses/intermediate-deep-learning-with-pytorch/training-robust-neural-networks?ex=1


# =========================== Model 1: SIFT (Grayscale conversion) + BovW + SVM ===============================================

# Some of the SIFT code is adapted from: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
class Model_1_Dataset(Dataset):
    def __init__(self, img_paths, targets, vocabulary, sift = None):
        self.img_paths = img_paths # A list of the image paths
        self.targets = targets # A list of the labels or targets [0, 1, 2]
        self.vocabulary = vocabulary # Visual words represented by K-Means clusters
        self.sift = cv2.SIFT_create() # Initialise new SIFT detector
        
    def __len__(self):
        return len(self.img_paths) # Returns the total number of samples
    
    # Incoporated some code from: https://www.geeksforgeeks.org/machine-learning/sift-interest-point-detector-using-python-opencv/
    # Also from Lab Tutorial 04 — IN3060/INM460 (Dr. Giacomo Tarroni)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx]) # An image gets loaded
        img = cv2.resize(img, (128, 128)) # Resizing image
        
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        
        gray_conversion = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Then converted to grayscale
        
        # SIFT feature extraction: detect keypoints and compute the descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray_conversion, None)
        
        # From the descriptors, compute a vector of the BovW histogram
        bovw_vector = compute_bovw_histogram(descriptors, self.vocabulary)
        
        # The output is a tensor consisting of the description vector with a label/class [0, 1, 2]
        return torch.tensor(bovw_vector, dtype = torch.float32), self.targets[idx]
  
    
# ============================= Model 2: HOG + Colour/Gradients + SVM ===========================================================

class Model_2_Dataset(Dataset):
    def __init__(self, img_paths, targets):
        self.img_paths = img_paths
        self.targets = targets
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.resize(img, (128, 128))
        
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        
        feature_vector = compute_hog_colour_histograms(img)
        return torch.tensor(feature_vector, dtype = torch.float32), self.targets[idx]
    

# ============================== Model 3: MobileNetV2 (Pretrained CNN) =============================================================

class Model_3_Dataset(Dataset):
    def __init__(self, img_paths, targets):
        self.img_paths = img_paths
        self.targets = targets
        self.transforms = model_transforms("mobilenet")
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transforms(img), self.targets[idx]
        

# ========================================= Model 4: Custom CNN ===================================================================

class Model_4_Dataset(Dataset):
    def __init__(self, img_paths, targets):
        self.img_paths = img_paths
        self.targets = targets
        self.transforms = model_transforms("custom_cnn")
     
    def __len__(self):
        return len(self.img_paths) 
        
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transforms(img), self.targets[idx]
        


















""" 
References:

1. OpenCV, 2024. Introduction to SIFT in OpenCV. [online] Available at: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html [Accessed 29 Jul. 2025]

scikit-learn developers, nd. sklearn.cluster.KMeans. scikit-learn. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html (Accessed: 29 July 2025).

2. Tarroni, G. (2025). Computer Vision - Lab Tutorial 04. MSc Data Science, City St George's.
"""