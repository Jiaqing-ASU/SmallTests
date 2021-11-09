# general template to evaluate text classification model

# download text classification dataset
import os
import psycopg2
import time

# If run out of GPU memory, you can uncomment the following line to 
# run inference on CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#if not os.path.isfile('model_deduplication_dataset.zip'):
#    model_deduplication_dataset_url = 'https://drive.google.com/uc?id=1nYzDDSJGkjCsVQI4gbSdC3DafQ7Ez107'
#    gdown.download(model_deduplication_dataset_url, output=None, quiet=False)
#    !unzip -qqq model_deduplication_dataset.zip

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# connect with the postgres
t_host = "localhost"
t_port = "5432"
t_dbname = "postgres"
t_user = "postgres"
t_pw = "postgres"
db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
db_cursor = db_conn.cursor()

A = np.zeros([100,1009375])

# load input from postgres
database_start = time.time()
try:
    for i in range(100):
        # execute the SELECT statement
        db_cursor.execute(""" SELECT array_data
                        FROM imdb WHERE id = %s """,(i,))
        blob = db_cursor.fetchone()
        A[i] = np.frombuffer(blob[0])
    # close the communication with the PostgresQL database
    db_cursor.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if db_conn is not None:
        db_conn.close()
database_end = time.time()
database_time = database_end - database_start

model_start = time.time()
# load the model from h5 file
model = tf.keras.models.load_model(
    'w2v_wiki500_yelp_embed_trainable.h5', custom_objects={'KerasLayer': hub.KerasLayer})
# get the weights from the 1st layer
layer0 = model.get_layer(index = 0)
weights = layer0.get_weights()
#weights = np.array(weights)
model_end = time.time()
model_time = model_end - model_start

matrix_start = time.time()
# matrix dot with weights in the 1st layer
#B = np.dot(A,weights)
B = tf.matmul(A,weights)
matrix_end = time.time()
matrix_time = matrix_end - matrix_start
# run the data on the model from the 2nd layer
inference_start = time.time()
output1 = model.layers[1](B)
output2 = model.layers[2](output1)
inference_end = time.time()
inference_time = inference_end - inference_start
#print(output2)
print('data loading time from postgres:', database_time)
print('model laoding time from local h5 file and getting 1st layer:', model_time)
print('matrix dot time between input and weights in 1st layer:', matrix_time)
print('inference time in 2nd and 3rd layer:', inference_time)