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

labels=[]
def load_labels():
    labels = pandas.read_csv("/home/emil.elmarsson/nih-chext-xrays/Data_Entry_2017.csv", usecols=["Image Index", "Finding Labels"])

def get_label(img_path):
    img_name = tf.strings.split(img_path, os.path.sep)[-1]
    for label in labels:
        if img_name == label[0]:
	    return label[1]
    return None

def decode_img(img):
    # convert compressed string to a uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    # convert to floats in range [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize image
    return tf.image.resize(img, [256, 256])

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

#image_paths = tf.data.Dataset.list_files("/home/emil.elmarsson/nih-chext-xrays/images_*/images/*")

#for image_path in image_paths.take(10):
#    print(image_path.numpy())

load_labels()
get_label("/home/emil.elmarsson/nih-chest-xrays/images_0001/00000001_000.png")
