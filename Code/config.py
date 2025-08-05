##### CW_Folder/Code/config.py #####

# Modified by: Charlie Maguire 

# Libraries needed:
import os
import sys
from utils import get_img_paths_and_targets # Helper function from utils.py

# Most of the code is done on VSCode then exported to Colab for training
# Therefore, an If statement is done to work locally and on Colab (cloud)
# sys.modules to see if Colab is running first and use os.path if not
if "google.colab" in sys.modules:
    dataset_root_dir = "/content/drive/MyDrive/CW_Folder/CW_Dataset"
else: 
    # If google.colab is not detected then search locally
    dataset_root_dir = os.path.join("..", "CW_Dataset")

# The folder locations for the images
img_train_dir = os.path.join(dataset_root_dir, "train", "images")
img_test_dir = os.path.join(dataset_root_dir, "test", "images")

# The folder location for the labels/targets
targets_train_dir = os.path.join(dataset_root_dir, "train", "labels")
targets_test_dir = os.path.join(dataset_root_dir, "test", "labels")

# The image path variables using a helper function
# The helper function is get_img_path
img_train_path = get_img_paths_and_targets(img_train_dir, targets_train_dir)
img_test_path = get_img_paths_and_targets(img_test_dir, targets_test_dir)


# ===== Other configuration settings =====

random_seed = 1 # reproducibility 
batch_size = 32 # reproducibility 

# This maps models according to torchvision transform settings in transforms.py
# Only used for the pretrained CNN and custom CNN (model 3 and 4)
model_transform_map = {
    "model_3": "mobilenet",
    "model_4": "custom_cnn"
}
