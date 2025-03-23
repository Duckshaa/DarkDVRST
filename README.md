# DarkDVRST
Low-light Violence Detection using Deep Learning and Image Processing
Project Overview
DarkDVRST (Dark Environment Deep-Learning based Violence Detection using Image Processing and Spatio-Temporal Analysis) is a comprehensive project focused on detecting violence in video footage captured in low-light environments. The model incorporates advanced deep learning techniques and several image processing strategies to improve detection accuracy, even in challenging conditions. The project is implemented using TensorFlow/Keras and Python.

Project Structure
The project is organized into several modules for better structure and maintainability. Each module is responsible for a specific part of the process, from data loading and preprocessing to model training and evaluation. The project includes the following files:

File Structure:
initialization.py # Imports, warnings suppression, environment setup
preprocessing.py # Data loading, augmentation, preprocessing functions
data_generator.py # Custom DataGenerator class
training.py # Model definition, training loop, callbacks
evaluation.py # Model evaluation and performance metrics
utils.py # Helper functions (e.g., video frame extraction, optical flow)

Features
Low-Light Enhancement: Uses RetinexNet to enhance low-light video data for better feature extraction.
Violence Detection: Implements violence detection using video data categorized as "fights" and "noFights".
Spatio-Temporal Analysis: Combines CNNs and Transformers for feature extraction and temporal analysis.
Multi-Camera Synchronization: Integrates multiple camera feeds with time synchronization for a broader analysis.
Behavioral Analysis: Classifies aggressive behavior using Optical Flow and machine learning classifiers.
Data Augmentation: Applies data augmentation techniques to increase the robustness of the model.

Setup Instructions
Prerequisites
Before running the code, ensure that you have the following installed:
Python 3.6 or higher
TensorFlow 2.x
OpenCV
numpy
scikit-learn
kagglehub (for downloading datasets)

The dataset used here is "sparshdrolia/violence", and it can be downloaded from Kaggle using the following command:
kaggle datasets download -d sparshdrolia/violence

Running the Project
Initialize the environment: Run initialization.py to set up the environment and import necessary libraries.
Preprocess the Data: Use preprocessing.py to extract and preprocess video frames.
Create Data Generators: data_generator.py contains the DataGenerator class to feed data into the model.
Train the Model: Use training.py to build and train the model using the training data.
Evaluate the Model: Once the model is trained, evaluate its performance using evaluation.py.

That's about it! Thank you :)
