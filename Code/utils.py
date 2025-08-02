##### CW_Folder/Code/utils.py #####

# Modified by: Charlie Maguire with any incorporation of sources listed in the code

# Libraries needed:
import os

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


   
            
            
                
                
            