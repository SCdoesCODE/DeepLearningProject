import numpy as np
import tensorflow as tf
from tensorflow_io.bigquery import BigQueryClient
import matplotlib.pyplot as plt
import time
import os
import cv2

PROJECT_ID = "deep-learning-project-275614"
DATASET_GCP_PROJECT_ID = "chc-nih-chest-xray"
DATASET_ID = "nih-chest-xray"
TABLE_ID = "chc-nih-chest-xray:nih_chest_xray.nih_chest_xray"

#filenames = tf.io.gfile.glob("gs://gcs-public-data--healthcare-nih-chest-xray")

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
  '''
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

run_benchmark(10000)
