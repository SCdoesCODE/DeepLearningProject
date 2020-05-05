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

bucket_name = "gcs-public-data--healthcare-nih-chest-xray"
project_name = "Deep Learning Project"
project_id = "deep-learning-project-275614"
source_blob_name = "source-blob-name"
destination_file_name = "png/"


from google.cloud import storage

import numpy as np
import cv2



def load_data(bucket_name):
    bucket = storage.Client().bucket(bucket_name, user_project = project_id)

    flattened_images = []
    iterator = 0
    for index,blob in enumerate(bucket.list_blobs()):
        if blob.name.endswith(".png"):
            flattened_images.append(cv2.imdecode(np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), 0).flatten())
            #image = cv2.imdecode(np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), 0)
            #plt.imshow(image, cmap='gray')  # graph it
            #plt.show()  # display!
            if(index%100 == 0):
                print(index)

    return flattened_images


X = load_data(bucket_name)


np.savetxt("flattened_images.txt",X)

#x = list(x)

print(X[0])
