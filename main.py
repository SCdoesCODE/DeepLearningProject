"""
sentdex tutorial

https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/

annan bra tutorial som gör typ precis det vi vill göra?

https://towardsdatascience.com/detecting-covid-19-induced-pneumonia-from-chest-x-rays-with-transfer-learning-an-implementation-311484e6afc1


google cloud nih dataset
https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest


loada in datasetet på google cloud
https://towardsdatascience.com/automl-and-big-data-980e24fba6fa
"""

import numpy as np
import tensorflow as tf
#from tensorflow_io.bigquery import BigQueryClient
from keras.preprocessing import image
from google.cloud import storage
import matplotlib.pyplot as plt
import time
import os
import cv2

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="Deep Learning Project-98e1d6697904.json"

our_bucket_name = "nih-chest-x-rays"
bucket_name = "gcs-public-data--healthcare-nih-chest-xray"
project_name = "Deep Learning Project"
project_id = "deep-learning-project-275614"
source_blob_name = "source-blob-name"
destination_file_name = "png/"

def load_data(bucket_name):
    storage_client = storage.Client()
    private_bucket = storage_client.bucket(our_bucket_name, user_project = project_id)
    bucket = storage_client.bucket(bucket_name, user_project = project_id)

    for blob in bucket.list_blobs():
        if blob.name.endswith(".png"):
            image = cv2.imdecode(np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), 0).flatten()
            np.savetxt("flattened_image.txt", image)
            # [4:-4] removes the file ending and the folder from the path
            private_blob = private_bucket.blob("flattened_images/" + blob.name[4:-4] + ".txt")
            private_blob.upload_from_filename("flattened_image.txt")
            break

def load_image(source_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(our_bucket_name, user_project = project_id)
    blob = bucket.blob(source_name)
    array = np.asarray(blob.download_as_string().splitlines()).reshape(1024,1024)
    image = cv2.imdecode(np.float32(array), 0)
    plt.imshow(image)
    plt.show()

#load_data(bucket_name)
#upload_blob(our_bucket_name, "flattened_image.txt", "flattened_image.txt")
load_image("flattened_images/00000001_000.txt")
#strtest = "2.020000000000000000e+02\n2.080000000000000000e+02\n2.080000000000000000e+02"
#array = strtest.splitlines()
#nparray = np.asarray(array)
#print(nparray)

#np.savetxt("flattened_images.txt",X)
