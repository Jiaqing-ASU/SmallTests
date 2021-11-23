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

if __name__ == "__main__":
    
    # connect with the postgres
    t_host = "localhost"
    t_port = "5432"
    t_dbname = "postgres"
    t_user = "postgres"
    t_pw = "postgres"
    db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    db_cursor = db_conn.cursor()

    loading_start = time.time()
    targets = []
    #for i in range(598):
    #    if (i == 597):
    #        targets.append(np.zeros([1000, 540]))
    #    else:
    #        targets.append(np.zeros([1000, 1000]))
    print ("loading inputs from Postgres")
    # load input from postgres
    try:
        for i in range(1000):
            arr = None
            for j in range(299):
                # execute the SELECT statement
                db_cursor.execute("SELECT N" + str(j) + "Column FROM extreme_m WHERE id = " + str(i) + ";")
                #print("SELECT N" + str(j) + "Column FROM extreme_m WHERE id = " + str(i) + ";")
                blob = db_cursor.fetchone()
                this_target = np.frombuffer(blob[0])
                #print(this_target.shape)
                if(arr is None):
                    arr = this_target
                else:
                    arr = np.concatenate([arr,this_target],axis = 0)
            #print(arr.shape)
            #print(i)
            targets.append(tf.dtypes.cast(arr, tf.double))
            #targets.append(tf.dtypes.cast(this_target, tf.double))
            # close the communication with the PostgresQL database
        db_cursor.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if db_conn is not None:
            db_conn.close()
    #np.array(targets)
    print(len(targets))
    #targets = tf.dtypes.cast(targets, tf.double)
    #targets = tf.dtypes.cast(targets, tf.float32)
    loading_end = time.time()
    print('data loading time', loading_end-loading_start, ' seconds')