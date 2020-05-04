import numpy as np
import tensorflow as tf
from tensorflow_io.bigquery import BigQueryClient
from keras.preprocessing import image
from google.cloud import storage
import matplotlib.pyplot as plt
import time
import os
import cv2

'''
PROJECT_ID = "deep-learning-project-275614"
DATASET_GCP_PROJECT_ID = "chc-nih-chest-xray"
DATASET_ID = "nih-chest-xray"
TABLE_ID = "chc-nih-chest-xray:nih_chest_xray.nih_chest_xray"

sess = tf.compat.v1.Session()

image_path = 'gs://gcs-public-data--healthcare-nih-chest-xray/png/00000001_000.png?userProject=deep-learning-project-275614'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)
image_array = sess.run(image)
'''

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/emilelm/Kurser/DeepLearningProject/Deep Learning Project-98e1d6697904.json"

bucket_name = "gcs-public-data--healthcare-nih-chest-xray"
project_name = "Deep Learning Project"
project_id = "deep-learning-project-275614"
source_blob_name = "source-blob-name"
destination_file_name = "gs://gcs-public-data--healthcare-nih-chest-xray/png/00000001_000.png"

'''
client = storage.Client()
bucket = client.bucket(bucket_name, user_project=project_id)
blob = storage.Blob('gs://gcs-public-data--healthcare-nih-chest-xray/png/00000001_000.png', bucket)
with open('image.png') as file_obj:
    client.download_blob_to_file(blob, file_obj)
'''

client = storage.Client()
bucket = client.bucket(bucket_name, user_project=project_id)
blob = bucket.blob(destination_file_name)
blob.download_to_filename('image.png')

'''
# Initialise a client
storage_client = storage.Client(project_name)
# Create a bucket object for our bucket
bucket = storage_client.bucket(bucket_name, user_project=project_id)
# Create a blob object from the filepath
blob = bucket.blob("gs://gcs-public-data--healthcare-nih-chest-xray/png/00000001_000.png")
# Download the file to a destination
blob.download_to_filename("image.png")
'''
#for blob in bucket.list_blobs():
#    print blob.name


#blob = bucket.blob(source_blob_name)
#blob.download_to_filename(destination_file_name)

#print(
#    "Blob {} downloaded to {} using a requester-pays request.".format(
#        source_blob_name, destination_file_name
#    )
#)

#filenames = tf.io.gfile.glob("gs://gcs-public-data--healthcare-nih-chest-xray")
'''
def run_benchmark(num_iterations):
  batch_size = 2048
  client = BigQueryClient()
  read_session = client.read_session(
      "projects/" + PROJECT_ID,
      DATASET_GCP_PROJECT_ID, TABLE_ID, DATASET_ID,
      ["title",
       "id",
       "num_characters",
       "language",
       "timestamp",
       "wp_namespace",
       "contributor_username",
       "PatientID"],
      [tf.string,
       tf.int64,
       tf.int64,
       tf.string,
       tf.int64,
       tf.int64,
       tf.string,
       tf.string],
      requested_streams=10
  )

  #print read_session.PatientID

  #dataset = read_session.parallel_read_rows(sloppy=True).batch(batch_size)

  itr = dataset.make_one_shot_iterator()

  n = 0
  mini_batch = 100
  for i in range(num_iterations // mini_batch):
    local_start = time.time()
    start_n = n
    for j in range(mini_batch):
      n += batch_size
      batch = itr.get_next()

    local_end = time.time()
    print('Processed %d entries in %f seconds. [%f] examples/s' % (
        n - start_n, local_end - local_start,
        (mini_batch * batch_size) / (local_end - local_start)))
    '''

#run_benchmark(10000)
