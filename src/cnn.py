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
from tensorflow.keras import datasets, layers, models
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
    classes = open("../resources/classes.txt").read().splitlines()

    # Loading image names and the corresponding labels
    #names_and_labels = pandas.read_csv('~/nih-chext-xrays/Data_Entry_2017.csv', usecols=["Image Index", "Finding Labels"])
    names_and_labels = pandas.read_csv('/mnt/disks/new-disk/nih-chest-xrays/Data_Entry_2017.csv', usecols=["Image Index", "Finding Labels"])

init()

AUTOTUNE = tf.data.experimental.AUTOTUNE

LR = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_CLASSES = len(classes)
DATASET_SIZE = len(names_and_labels.index)
SHUFFLE_BUFFER_SIZE = 1024

def get_label(img_path):
    # Extracting the name from the image path
    img_name = tf.strings.split(img_path, os.path.sep)[-1].numpy().decode('utf-8')
    try:
        # Locating the proper entry and returning its label
        label_string = names_and_labels.loc[names_and_labels['Image Index'] == img_name]['Finding Labels'].values[0]
        return classes.index(label_string)
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

# Split the dataset into training and test data
def split_dataset(dataset, test_data_fraction):
    test_data_percent = round(test_data_fraction * 100)
    if not (0 <= test_data_percent <= 100):
        raise ValueError("Test data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > test_data_percent)
    test_dataset = dataset.filter(lambda f, data: f % 100 <= test_data_percent)

    # remove enumeration
    train_data = train_dataset.map(lambda f, data: data)
    test_data = test_dataset.map(lambda f, data: data)

    return train_data, test_data

def prepare_dataset(dataset, is_training=True):
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

# Loading image paths
#image_paths = tf.data.Dataset.list_files("/home/emil.elmarsson/nih-chext-xrays/images_*/images/*")
image_paths = tf.data.Dataset.list_files("/mnt/disks/new-disk/nih-chest-xrays/images_*/images/*")

# Mapping image paths to the respective image and label
dataset = image_paths.map(lambda path: tf.py_function(func=process_path, inp=[path], Tout=(tf.float32, tf.int64)), num_parallel_calls=AUTOTUNE)

# Splitting data
train_data, test_data = split_dataset(dataset=dataset, test_data_fraction=0.3)

# Caching, batching, etc.
train_data = prepare_dataset(train_data)
test_data = prepare_dataset(test_data)

# Create the model
model = Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='softmax'))
model.add(layers.Dense(NUM_CLASSES))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss="binary_crossentropy",
              metrics=['accuracy'])

history = model.fit(train_data, 
                    epochs=NUM_EPOCHS,
                    validation_data=test_data)