import numpy as np
from google.cloud import storage
import matplotlib.pyplot as plt
import time
import os
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle

X = np.loadtxt('flattened_images.txt')

X = X/255.0
