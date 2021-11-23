import io
import re
import string
import tqdm
import time

import numpy as np
import os
import psycopg2
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
 # Define the vocabulary size and number of words in a sequence.
vocab_size = 1009375
embedding_dim = 500
batch_size = 100
num_models = 6

class Word2Vec_MM():
    def __init__(self, input_weights):
        #self.weights = np.copy(tf.dtypes.cast(input_weights, tf.double))
        self.weights = np.copy(tf.dtypes.cast(input_weights, tf.float32))

    def predict(self, input_batch):
        return tf.matmul(input_batch, self.weights)

if __name__ == "__main__":

    loading_method  = input("Enter 1 for loading from Postgres OR 2 for loading from CSV file\n")
    # connect with the postgres
    t_host = "localhost"
    t_port = "5432"
    t_dbname = "postgres"
    t_user = "postgres"
    t_pw = "postgres"
    db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    db_cursor = db_conn.cursor()

    print("loading model")
    word2vec = tf.keras.models.load_model(
    'w2v_wiki500_yelp_embed_nontrainable.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    layer = word2vec.get_layer('keras_layer_1')
    weights= layer.get_weights()
    model_list = []
    for i in range(num_models):
        model_list.append(Word2Vec_MM(weights))
    
    loading_start = time.time()
    targets = np.zeros([100,1009375])
    if(loading_method == "1"):
        print ("loading inputs from Postgres")
        # load input from postgres
        try:
            for i in range(100):
                # execute the SELECT statement
                db_cursor.execute(""" SELECT array_data FROM imdb WHERE id = %s """,(i,))
                blob = db_cursor.fetchone()
                targets[i] = np.frombuffer(blob[0])
            # close the communication with the PostgresQL database
            db_cursor.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if db_conn is not None:
                db_conn.close()
        #targets = tf.dtypes.cast(targets, tf.double)
        targets = tf.dtypes.cast(targets, tf.float32)
    elif(loading_method == "2"):
        print ("loading inputs from CSV file")
        #targets = tf.dtypes.cast(np.load('inputs.npy'), tf.double)
        targets = tf.dtypes.cast(np.load('inputs.npy'), tf.float32)
    else:
        print("Invalid Input")
    loading_end = time.time()
    print('data loading time', loading_end-loading_start, ' seconds')
    #targets = tf.dtypes.cast(np.random.rand(100,vocab_size), tf.double)
    #targets = tf.dtypes.cast(np.random.rand(100,vocab_size), tf.float32)
    print("making inference")
    inference_start = time.time()
    for i in range(num_models):
        results = model_list[i].predict(targets)
        print(results)
    inference_end = time.time()
    print('inference time for ', num_models, ' models:', inference_end-inference_start, ' seconds')