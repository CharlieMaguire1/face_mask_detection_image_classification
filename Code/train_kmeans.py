##### CW_Folder/Code/train_kmeans.py #####

# Modified by: Charlie Maguire with any incorporation of sources listed below

# Libraries needed:
import cv2
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans # This is for the BovW histogram in transforms.py

from config import random_seed, img_train_dir # Importing variables from config.py


# ===== For Model 1: SIFT + BovW + MLP =====

# Incorporated kmeans from: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
sift = cv2.SIFT_create()
sift_descriptors = [] # Collecting all descriptors from images

for file in os.listdir(img_train_dir):
    img_path = os.path.join(img_train_dir, file)
    img = cv2.imread(img_path)
    if img is None:
        continue
    gray_conversion = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_conversion, None)
    if descriptors is not None:
        sift_descriptors.append(descriptors)
    
sift_descriptors = np.vstack(sift_descriptors)

k_means = KMeans(n_clusters = 200, random_state = random_seed)
k_means.fit(sift_descriptors)
    
joblib.dump(k_means, "../Models/kmeans_model.pkl") # Saving model
print("KMeans Saving Completed")

