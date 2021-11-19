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

num_models = 2

# Comment it if on GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    #model = tf.keras.models.load_model('extreme_classification_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    model_list = []
    for i in range(num_models):
        model_list.append(tf.keras.models.load_model('extreme_classification_model_double.h5', custom_objects={'KerasLayer': hub.KerasLayer})) 
    for i in range(num_models):
        for layer in model_list[i].layers:
            a,b = layer.get_weights()[0].shape
            #w = tf.dtypes.cast(tf.random.normal([a,b], stddev = 2, mean = 0, seed =1), tf.float32)
            #b = tf.dtypes.cast(tf.random.normal((layer.get_weights()[1].shape), stddev = 2, mean = 0, seed =1), tf.float32)
            w = tf.dtypes.cast(tf.random.normal([a,b], stddev = 2, mean = 0, seed =1), tf.double)
            b = tf.dtypes.cast(tf.random.normal((layer.get_weights()[1].shape), stddev = 2, mean = 0, seed =1), tf.double)
        layer.set_weights([w, b])
        model_list[i].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    loading_start = time.time()
    targets = np.zeros([1000, 597540])
    if(loading_method == "1"):
        print ("loading inputs from Postgres")
        # load input from postgres
        try:
            for i in range(1000):
                # execute the SELECT statement
                db_cursor.execute(""" SELECT array_data FROM extreme WHERE id = %s """,(i,))
                blob = db_cursor.fetchone()
                targets[i] = np.frombuffer(blob[0])
            # close the communication with the PostgresQL database
            db_cursor.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if db_conn is not None:
                db_conn.close()
        targets = tf.dtypes.cast(targets, tf.double)
        #targets = tf.dtypes.cast(targets, tf.float32)
    elif(loading_method == "2"):
        print ("loading inputs from CSV file")
        file = open("input_double.csv")
        #file = open("input_float.csv")
        targets = np.loadtxt(file, delimiter=",")
    else:
        print("Invalid Input")
    loading_end = time.time()
    print('data loading time', loading_end-loading_start, ' seconds')
    print("making inference")
    inference_start = time.time()
    for i in range(num_models):
        results = model_list[i].predict(targets)
    inference_end = time.time()
    print('inference time for ', num_models, ' models:', inference_end-inference_start, ' seconds')