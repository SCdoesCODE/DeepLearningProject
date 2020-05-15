import numpy as np
from numpy.random import default_rng
#from google.cloud import storage
#import matplotlib.pyplot as plt
import time
import glob
import os
from os.path import expanduser
from pathlib import Path
import cv2
import pandas
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, VGG16
import logging

# For suppressing annoying log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

set_tf_loglevel(logging.FATAL)

HOME = expanduser("~")

# The class names (diseases) of the model
classes=[]
# CSV object
names_and_labels=[]

# Loading all labels from the CSV file
def init():
    global classes
    global names_and_labels

    # Loading class names
    classes = open(HOME + "/DeepLearningProject/resources/classes.txt").read().splitlines()

    # Loading image names and the corresponding labels
    #names_and_labels = pandas.read_csv('~/nih-chext-xrays/Data_Entry_2017.csv', usecols=["Image Index", "Finding Labels"])
    names_and_labels = pandas.read_csv('~/nih-chest-xrays/Data_Entry_2017.csv', usecols=["Image Index", "Finding Labels"])
    # Making sure that multivalues are formatted correctly
    names_and_labels["Finding Labels"]=names_and_labels["Finding Labels"].apply(lambda x:x.split("|"))

init()

AUTOTUNE = tf.data.experimental.AUTOTUNE

LR = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_CLASSES = len(classes)
DATASET_SIZE = len(names_and_labels.index)
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2
SHUFFLE_BUFFER_SIZE = 1024
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

def decode_img(img):
    # convert compressed string to a uint8 tensor
    img = tf.image.decode_png(img, channels=CHANNELS)
    # convert color values to floats in range [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize image
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

# Processing a given image path, returning the image and the corresponding label
def process_path(img_path):
    img = tf.io.read_file(img_path)
    img = decode_img(img)
    return img

def prepare_dataset(dataset):
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# Takes an array of label names and returns the one-hot encoding label
def one_hot_encoding(label_names):
    one_hot = np.zeros(NUM_CLASSES)
    for label_name in label_names:
        index = classes.index(label_name)
        one_hot[index] = 1
    return one_hot

# One-hot encodes all labels using helper function above
def one_hot_encode_labels(labels):
    new_labels = np.zeros((len(labels), NUM_CLASSES))
    for i,label in enumerate(labels):
        new_labels[i] = one_hot_encoding(label)
    return new_labels

def create_data():
    image_path = HOME + "/nih-chest-xrays/images/"

    # Random list of file indices to ensure that the distribution of the training, validation and test datasets varies over different runs
    file_indices = default_rng().choice(DATASET_SIZE, size=DATASET_SIZE, replace=False)

    # Splitting the indices based on the fractions given to each dataset
    [train_indices,val_indices,test_indices] = np.split(file_indices, [int(DATASET_SIZE*TRAIN_FRAC), int(DATASET_SIZE*(TRAIN_FRAC+VAL_FRAC))])

    # Getting the filenames from the file indices
    train_names = names_and_labels.iloc[train_indices]['Image Index'].to_numpy()
    val_names = names_and_labels.iloc[val_indices]['Image Index'].to_numpy()
    test_names = names_and_labels.iloc[test_indices]['Image Index'].to_numpy()

    # Creating the full filepaths
    name_to_path = np.vectorize(lambda name: image_path + name)
    train_paths = name_to_path(train_names)
    val_paths = name_to_path(val_names)
    test_paths = name_to_path(test_names)

    # Mapping the filepaths to images
    train_images = tf.data.Dataset.from_tensor_slices(train_paths).map(process_path, num_parallel_calls=AUTOTUNE)
    val_images = tf.data.Dataset.from_tensor_slices(val_paths).map(process_path, num_parallel_calls=AUTOTUNE)
    test_images = tf.data.Dataset.from_tensor_slices(test_paths).map(process_path, num_parallel_calls=AUTOTUNE)

    # Mapping the filenames to labels
    train_labels = tf.data.Dataset.from_tensor_slices(one_hot_encode_labels(names_and_labels[names_and_labels['Image Index'].isin(train_names)]['Finding Labels'].to_numpy()))
    val_labels = tf.data.Dataset.from_tensor_slices(one_hot_encode_labels(names_and_labels[names_and_labels['Image Index'].isin(val_names)]['Finding Labels'].to_numpy()))
    test_labels = tf.data.Dataset.from_tensor_slices(one_hot_encode_labels(names_and_labels[names_and_labels['Image Index'].isin(test_names)]['Finding Labels'].to_numpy()))

    # Preparing data for training
    train_ds = prepare_dataset(tf.data.Dataset.zip((train_images, train_labels)))
    val_ds = prepare_dataset(tf.data.Dataset.zip((val_images, val_labels)))
    test_ds = tf.data.Dataset.zip((test_images, test_labels))

    return train_ds, val_ds, test_ds

def create_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,CHANNELS)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(NUM_CLASSES))
    return model

train_ds, val_ds, test_ds = create_data()

#model = create_model()
#model.summary()

base_model = VGG16(include_top=False)

model = base_model.output
model = GlobalAveragePooling2D()(model)
model = Dense(1024, activation='relu')(model)
predictions = Dense(NUM_CLASSES, activation='sigmoid')(model)

model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss="binary_crossentropy",
              metrics=['accuracy'])

# Saving the best model every epoch (based on val loss)
checkpoint_path = HOME + "/DeepLearningProject/models/nih_model.h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True)                      

# Early stopping based on validation loss
es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# Fitting
history = model.fit(train_ds, 
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=[cp_callback, es_callback])

test_ds = prepare_dataset(test_ds)

# Predicting test data
preds = model.evaluate(test_ds)