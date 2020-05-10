import numpy as np
from google.cloud import storage
#import matplotlib.pyplot as plt
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
    # Making sure that multivalues are formatted correctly
    names_and_labels["Finding Labels"]=names_and_labels["Finding Labels"].apply(lambda x:x.split("|"))

init()

AUTOTUNE = tf.data.experimental.AUTOTUNE

LR = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_CLASSES = len(classes)
DATASET_SIZE = len(names_and_labels.index)
SHUFFLE_BUFFER_SIZE = 1024
IMG_HEIGHT = 256
IMG_WIDTH = 256

train_dir = "/mnt/disks/new-disk/nih-chest-xrays/images/"

train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.3)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=names_and_labels,
    directory=train_dir,
    x_col='Image Index',
    y_col='Finding Labels',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    subset='training')

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=names_and_labels,
    directory=train_dir,
    x_col='Image Index',
    y_col='Finding Labels',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    subset='validation')

# Make a test generator as well to evaluate the model after training. It seems we need to move some images into a separate directory for that.

# Create the model
model = Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,1)))
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
              loss="categorical_crossentropy",
              metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN, 
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS)
