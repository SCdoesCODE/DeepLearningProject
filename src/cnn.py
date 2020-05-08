import numpy as np
from google.cloud import storage
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import cv2
import pandas
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

classes=[]
names_and_labels=[]
# Loading all labels from the CSV file
def init():
    global classes
    global names_and_labels

    # Loading class names
    #classes = np.fromfile("../resources/classes.txt", sep='\n')
    #print(classes)

    # Loading image names and the corresponding labels
    names_and_labels = pandas.read_csv('~/nih-chext-xrays/Data_Entry_2017.csv', usecols=["Image Index", "Finding Labels"])

def get_label(img_path):
    # Extracting the name from the image path
    img_name = tf.strings.split(img_path, os.path.sep)[-1].numpy().decode('utf-8')
    try:
        # Locating the proper entry and returning its label
        return names_and_labels.loc[names_and_labels['Image Index'] == img_name]['Finding Labels'].values[0]
    except:
        print("Error: Could not find image label.")
        return None

def decode_img(img):
    # convert compressed string to a uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    # convert to floats in range [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize image
    return tf.image.resize(img, [256, 256])

# Processing a given image path, returning the image and the corresponding label
def process_path(img_path):
    label = get_label(img_path)
    img = tf.io.read_file(img_path)
    img = decode_img(img)
    return img, label

init()

EPOCHS = 10
DATASET_SIZE = len(names_and_labels.index)
BATCH_SIZE = 100
TRAIN_SIZE = int(0.7 * DATASET_SIZE)
TEST_SIZE = DATASET_SIZE - TRAIN_SIZE

# Loading image paths
image_paths = tf.data.Dataset.list_files("/home/emil.elmarsson/nih-chext-xrays/images_*/images/*")

# Mapping image paths to the respective image and label
dataset = image_paths.map(lambda path: tf.py_function(func=process_path, inp=[path], Tout=(tf.float32, tf.string)), num_parallel_calls=1)

#dataset = dataset.shuffle()
#train_dataset = dataset.take(TRAIN_SIZE)
#test_dataset = dataset.skip(TRAIN_SIZE).take(TEST_SIZE)

plt.figure(figsize=(10,10))
for i, (img, label) in enumerate(dataset.take(25)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap="gray")
    plt.xlabel(label.numpy().decode('utf-8'))
plt.show()

# Create the model
#model = Sequential()
