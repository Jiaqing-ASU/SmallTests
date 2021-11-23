import io
import re
import string
import tqdm
import time
import psycopg2

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from keras.initializers import Constant

# Comment it if on GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
 # Define the vocabulary size and number of words in a sequence.
vocab_size = 1009375
embedding_dim = 500
num_models = 12

def create_word2vec_model(input_weights):
    word2vec = Sequential()
    embedding = layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=Constant(input_weights), trainable=False)
    word2vec.add(embedding)
    return word2vec

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
    weights = np.copy(tf.dtypes.cast(weights, tf.float32))
    model_list = []
    for i in range (num_models):
        model_list.append(create_word2vec_model(weights))
    
    loading_start = time.time()
    if(loading_method == "1"):
        print ("loading inputs from Postgres")
        # load input from postgres
        try:
            # execute the SELECT statement
            db_cursor.execute(""" SELECT array_data FROM imdb WHERE id = %s """,(0,))
            blob = db_cursor.fetchone()
            targets = np.frombuffer(blob[0])
            # close the communication with the PostgresQL database
            db_cursor.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if db_conn is not None:
                db_conn.close()
        targets = tf.dtypes.cast(targets, tf.float32)
    elif(loading_method == "2"):
        print ("loading inputs from CSV file")
        file = open("input_float_G.csv")
        targets = np.loadtxt(file, delimiter=",")
        targets = tf.dtypes.cast(targets, tf.float32)
    else:
        print("Invalid Input")
    loading_end = time.time()
    print('data loading time', loading_end-loading_start, ' seconds')

    print("making inference")
    inference_start = time.time()
    for i in range(num_models):
        results = model_list[i].predict(targets)
    inference_end = time.time()
    print('inference time for all', num_models, 'models', inference_end-inference_start, ' seconds')