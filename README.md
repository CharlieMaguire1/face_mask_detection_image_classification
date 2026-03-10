# Face Mask Classification with Classical CV and Deep Learning

**Status:** Comparative CV project | Core training pipelines implemented | Detection deployment stage partially complete

This project compares **classical computer vision** and **deep learning** approaches for a 3-class face mask classification task.

It explores handcrafted feature extraction, transfer learning, and custom neural network pipelines, then begins with extending the classifier toward a downstream video-based face-mask detection workflow.


---


## What this project does

The repository implements and experiments with four modelling routes:

- **Model 1:** SIFT + Bag of Visual Words + MLP
- **Model 2:** HOG + RGB colour histograms + linear SVM
- **Model 3:** MobileNetV2 transfer learning
- **Model 4:** Custom CNN scaffold

It also includes an unfinished **face-mask detection function** intended to run classification on detected faces from video frames.


---


## Main components

### Model 1 — SIFT + BoVW + MLP
- SIFT keypoint extraction
- KMeans-trained visual vocabulary
- Bag of Visual Words histograms
- MLP classifier in PyTorch

### Model 2 — HOG + colour histograms + SVM
- grayscale HOG descriptors
- RGB colour histogram features
- concatenated handcrafted feature vector
- custom multiclass linear SVM in PyTorch
- Optuna tuning and checkpoint saving

### Model 3 — MobileNetV2
- pretrained MobileNetV2 backbone
- custom classification head
- torchvision transforms and transfer-learning setup

### Model 4 — Custom CNN
- dataset and transform scaffolding included
- training pipeline not fully consolidated


---


## Additional work

The repo also includes early work toward **video-based face-mask detection**, where the intended pipeline was:

1. detect faces in video frames  
2. crop and preprocess the face region  
3. classify the face using the trained model  
4. render bounding boxes and class labels  

This deployment function is **not yet complete**.


---


## Repository structure

```text
face-mask-classification-cv/
├── Code/
│   ├── __init__.py
│   ├── class_distribution.py
│   ├── config.py
│   ├── data_functions.py
│   ├── datasets.py
│   ├── face_mask_detection_function.py
│   ├── model1_sift_bovw_mlp.py
│   ├── model2_hog_rgb_svm.py
│   ├── model3_mobilenetv2.py
│   ├── train_kmeans.py
│   ├── transforms.py
│   └── utils.py
├── README.md
├── requirements.txt
└── .gitignore
```


---


## Dataset format

The project expects image files and integer class labels stored as matching .txt files.

```text
CW_Dataset/
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```


---


## Technical highlights

This project includes:

- Reusable PyTorch Dataset subclasses
- Handcrafted feature extraction pipelines
- KMeans visual vocabulary training
- Custom training/evaluation utilities
- Multi-class metrics:

  - Accuracy
  - Macro F1
  - Precision
  - Recall
 
- Early stopping helpers
- Optuna-based hyperparameter tuning
- Checkpoint and parameter saving


---


## Current status

### Implemented

- Dataset loading and label parsing
- KMeans training for BoVW vocabulary
- SIFT + BoVW feature pipeline
- HOG + RGB histogram feature pipeline
- PyTorch-based MLP and linear SVM training loops
- MobileNetV2 dataset and training setup
- Metric tracking and utility functions
- Early stopping utilities
- Partial model saving/ tuning workflow

### Still incomplete

- Some of the scripts still contain unresolved bugs or inconsistencies
- Final comparative results are not fully consolidated in the repository
- The MobileNetV2 pipeline needs a further cleanup
- The custom CNN route was not fully implemented
- The video-based detection function is scaffolded but unfinished


---


## Limitations

This repository should be seen as a comparative CV experimentation project, not a finished production detector.

The known limitations are:

- Some of the scripts need consolidation after post-submission revisions
- Reproducibility would be better with a clearer run order
- Not all the model routes are completed end-to-end
- The result summaries are not yet centralised into one final comparison
- The downstream detection stage is not yet operational.


---


## Planned improvements

- Consolidate the final results for all model routes
- Fix the remaining import/logic inconsistencies
- Complete the best performing training pipeline cleanly
- Add confusion matrices and per-class performance
- Finish the face-mask video detection function
- Standardise the model saving/loading across all pipelines


---


## Skills demonstrated

- Python
- PyTorch
- torchvision
- torchmetrics
- OpenCV
- scikit-image
- scikit-learn
- SIFT/HOG feature extraction
- Bag of Visual Words
- Transfer learning
- Multi-class image classification
- Hyperparameter tuning with Optuna
- Resusable training/evaluation utility design


---


## Notes

This project is a MSc computer vision comparison project showing experimentation across both handcrafted and neural pipelines, with partial work towards deployment on video data.
  
