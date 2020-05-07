import numpy as np
from google.cloud import storage
import matplotlib.pyplot as plt
import time
import os
import cv2
import pandas
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

names_and_labels=[]
# Loading all labels from the CSV file
def load_names_and_labels():
    global names_and_labels
    names_and_labels = pandas.read_csv("~/nih-chext-xrays/Data_Entry_2017.csv", usecols=["Image Index", "Finding Labels"])

def get_label(img_path):
    # Extracting the name from the image path
    img_name = tf.strings.split(img_path, os.path.sep)[-1].numpy().decode('utf-8')
    try:
        # Locating the proper entry and returning its label
        return labels.loc[labels['Image Index'] == img_name]['Finding Labels'].values[0]
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
    print(img_path)
    label = get_label(img_path)
    img = tf.io.read_file(img_path)
    img = decode_img(img)
    return img, label

# Should always be at the top (loads all the labels)
load_names_and_labels()

image_paths = tf.data.Dataset.list_files("/home/emil.elmarsson/nih-chext-xrays/images_*/images/*")

# Mapping image paths to the respective image and label
# For some reason this doesn't work (the image paths are weird Tensor objects in the process path function)
dataset = image_paths.map(process_path)
for image, label in dataset.take(5):
    print("Shape:", image.numpy().shape)
    print("Label:", label)
