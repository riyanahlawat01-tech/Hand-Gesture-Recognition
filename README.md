## Hand Gesture Recognition Using Custom Convolutional Neural Network from Scratch ##

* A complete, end-to-end deep learning system built from the ground up to recognize 32 distinct hand gestures—comprising all 26 letters of the American Sign Language (ASL) alphabet and 6 common word-level gestures (hello, bye, yes, no, thankyou, perfect).

___________________________________________

# Key Highlights:

*Zero Pretrained Weights: Built entirely from scratch with randomly initialized weights (He Normal initialization). No transfer learning or external gesture detection APIs (like MediaPipe) were used.
*Custom Dataset Pipeline: 12,800 images captured via a custom webcam collection script across bilateral hands (left and right).
*High Performance: Achieved a peak validation accuracy of 99.24% (97.11% final validation accuracy).
*Real-Time Inference: Powered by OpenCV, running smoothly at ~13–18 FPS on standard consumer CPUs.

______________________________

# Project Structure & Architecture

├── dataset_loader.py       # TensorFlow data pipeline & data augmentation
├── collect_data.py         # Webcam ROI-based image collection utility
├── model.py                # Custom CNN architecture definition
├── train.py                # Training script with callbacks & optimizers
├── webcam_predict.py       # Real-time inference application via OpenCV
├── evaluate.py             # Quantitative evaluation & metrics report
└── requirements.txt        # Project dependencies

______________________________

# Model Architecture (model.py)

*The network consists of four convolutional blocks followed by a global average pooling layer and two fully connected dense layers:

-Conv Block 1: Conv2D(32, 3x3) + Batch Normalization + ReLU + MaxPool(2x2) $\rightarrow$ Output: (32, 32, 32)
-Conv Block 2: Conv2D(64, 3x3) + Batch Normalization + ReLU + MaxPool(2x2) $\rightarrow$ Output: (16, 16, 64)
-Conv Block 3: Conv2D(128, 3x3) + Batch Normalization + ReLU + MaxPool(2x2) $\rightarrow$ Output: (8, 8, 128)
-Conv Block 4: Conv2D(256, 3x3) + Batch Normalization + ReLU + MaxPool(2x2) $\rightarrow$ Output: (4, 4, 256)
-Head: Global Average Pooling $\rightarrow$ Dense (512, Dropout 0.5) $\rightarrow$ Dense (256, Dropout 0.3) $\rightarrow$ Softmax Output (32 classes)

--Total Parameters: 664,544 (~2.5 MB model size)

________________________________________________________________________________________________

# Dataset Summary

*Total Classes: 32 (26 ASL alphabets + 6 words)
*Images per Class: 400 images (200 left hand + 200 right hand)
*Total Dataset Size: 12,800 RGB images (64x64 pixels)
*Split: ~80% training (10,240 images) / ~20% validation (2,560 images)

___________________________________

# Training ConfigurationOptimizer: 

*Adam (Initial Learning Rate: $0.001$)
*Loss Function: Sparse Categorical Cross-Entropy
*Epochs / Batch Size: 35 epochs, batch size of 32
*Callbacks: ModelCheckpoint (saves best weights), EarlyStopping (patience = 10), ReduceLROnPlateau (factor = 0.5 after 5 stagnant epochs)

____________________________________________________________________________________________________

# Getting Started & Installation

1. Clone the Repositorygit clone https://github.com/your-username/hand-gesture-recognition.git

cd hand-gesture-recognition

2. Set Up Virtual EnvironmentIt is recommended to use Python 3.11 for seamless TensorFlow compatibility:python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependenciespip install -r requirements.txt

_________________________________________________________________

# Usage Guide

Step 1: Collect Custom Data (Optional)To capture your own dataset using the webcam region of interest (ROI) script:python collect_data.py
Step 2: Train the ModelTrain the custom CNN from scratch on your dataset:python train.py
Step 3: Evaluate PerformanceGenerate per-class accuracies and validation reports:python evaluate.py
Step 4: Run Real-Time Webcam InferenceLaunch the live gesture recognition interface:python webcam_predict.py

__________________________________________

# Controls inside webcam window:

* Press P to Pause/Resume prediction.
* Press S to capture and save a screenshot.
* Press Q to Quit the application.

________________________________________________________________________

# Results

*Final Validation Accuracy: 97.11% (Peak: 99.24% at Epoch 34)
*Final Validation Loss: 0.1620
*Inference Speed: ~13–18 FPS on CPU
---------------------------------------------------------------------------------------------------------------------------
