##### CW_Folder/Code/transforms.py #####

# Mostly inspired by DataCamp: https://app.datacamp.com/learn/courses/intermediate-deep-learning-with-pytorch
# Created/modified by: Charlie Maguire with any incorporation of sources listed in the code
# This file is for reusable functions for feature extraction methods and image transforms for PyTorch CNN models

# Libraries needed
import cv2
import numpy as np
import joblib
from torchvision import transforms
from skimage.feature import hog

# ====== These are transforms/ preprocessing for each CNN Model (Model 3 and 4) =====
# Inspired by the code on DataCamp's Images Convolutional Neural Networks section in Intermediate Deep Learning with PyTorch: 
#   -https://campus.datacamp.com/courses/intermediate-deep-learning-with-pytorch/images-convolutional-neural-networks?ex=8
#   -https://campus.datacamp.com/courses/intermediate-deep-learning-with-pytorch/images-convolutional-neural-networks?ex=2
# And for the pretrained Model 3 from Torchvision Docs: 
#   -https://docs.pytorch.org/vision/stable/models.html

def model_transforms(model):
    if model == "mobilenet":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [
                0,229, 0.224, 0.225])
        ])

    elif model == "custom_cnn":
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ])
    
    else:
        raise ValueError("That model name is invalid")


# ===== This feature extraction function is for Model 1: SIFT + BovW + MLP =====
# Incorporated kmeans from: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
vocabulary = joblib.load("../Models/kmeans_model.pkl") # This is the vocab (kmeans model) save in the Models directory

def compute_bovw_histogram(descriptors, vocabulary):
    """
    Create a Bag of visual words histogram from the quantisation and clustering of SIFT descriptors
    Made from a mix of lab 04 IN3060/INM460 material and scikit-learn KMeans clustering documentation
    """
    # An if statement basically saying if there is a descriptors found, assign it to the nearest cluster
    # And each cluster is a vector
    if descriptors is not None:
        visual_words = vocabulary.predict(descriptors)
        
        # Building an histogram from the kMeans clusters 
        histogram, _ = np.histogram(visual_words, 
                                    bins = np.arange(len(vocabulary.cluster_centers_) + 1)) # Bins are arrays of vector magnitudes
    
    # If descriptors are not found then make a zero vector 
    else:
        histogram = np.zeros(len(vocabulary.cluster_centers_))  
    return histogram   


# ===== This feature extraction function is for Model 2: HOG (RGB Channels) + SVM =====
# Incorporated some code from: https://www.geeksforgeeks.org/python/python-opencv-cv2-calchist-method/

def compute_hog_colour_histograms(img):
    """
    This function takes an image and extracts the hog features (grayscale) and normalised colour histograms
    Then concatenates them into a vector
    """
    img_colour = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Making sure all pictures are RGB colour images for colour histograms
    gray_conversion = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Coverting to grayscale for the HOG feature extraction
    
    hog_features = hog(gray_conversion, pixels_per_cell = (8, 8), cells_per_block = (2, 2),
                       visualize = True, feature_vector = True)
    
    # Computing colour histograms that are normalised
    colour_histogram = []
    for rgb_channel in cv2.split(img_colour):
        histogram = cv2.calcHist([rgb_channel], [0], None, [32], [0, 256])
        histogram_norm = cv2.normalize(histogram, None).flatten()
        colour_histogram.extend(histogram_norm)
        
    # Concatenation of the Colour Histograms and the HOG features for vectorisation
    features_vectorised = np.concatenate([np.array(hog_features), np.array(colour_histogram)])
    
    return features_vectorised

""" 
References:

PyTorch, 2024. Torchvision Models Documentation. Available at: https://pytorch.org/vision/stable/models.html (Accessed on 30 August 2025).

scikit-learn developers, nd. sklearn.cluster.KMeans. scikit-learn. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html (Accessed on 29 July 2025).
"""