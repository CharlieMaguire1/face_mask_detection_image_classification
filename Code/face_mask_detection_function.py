##### CW_Folder/Code/face_mask_detection_function.py #####


import matplotlib.pyplot as plt
from skimage.feature import Cascade
from skimage import data


# Inspired by DataCamp's: https://campus.datacamp.com/courses/image-processing-in-python/advanced-operations-detecting-faces-and-features?ex=7

# Loading the pretrained face detector with skimage
face_cascade = data.lbp_frontal_face_cascade_filename()

# Initialising the detector 
detector_cascade = Cascade(face_cascade)

# Applying detector on the image
face_detected = detector_cascade.detect_multi_scale(
    img = image, scale_factor = 1, step_ratio = 1, 
    min_size = (10, 10), max_size = (200, 200))


# This is for my Classifier model

def MaskDetection(vid_path, skip_frame_by = 10):
    
    # Opening the video file
    capture = cv2.VideoCapture(vid_path)
    
    index_frame = 0
    
    while capture.isOpened():
        r, frame = capture.read()
        if not r:
            break
        
        # Process by skip_frame_by
        if index_frame % skip_frame_by == 0:
            # Converting to grayscale for face detection
            gray_conversion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = 
            
        # Prepare matplotlib figure
        fig, ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        for (x, y, w, h) in faces:
            # Bounding box
            rectangle = 
            
            # Cropping and preprocessing face (resizing)
            face_image = frame
            face_resized = cv2.resize
            face_tensor = 
            
            # Predictions
            target = face_mask_model.predict(face_tensor)
            
            # Adding the target
            ax.text(x, y - 10, target, color = "white", fontsize = 10, backgroundcolor = "black")
            
        plt.axis("off")
        plt.show
