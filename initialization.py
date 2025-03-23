import os
import random
import warnings
import cv2  # OpenCV for frame extraction
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
import kagglehub

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Kaggle Dataset Path (Modify if needed)
sparshdrolia_violence_path = kagglehub.dataset_download('sparshdrolia/violence')
