# general template to evaluate text classification model

# download text classification dataset
import os
import time

# If run out of GPU memory, you can uncomment the following line to 
# run inference on CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# load the model from h5 file
# change the file name while changing the weights of other models
model = tf.keras.models.load_model(
    'w2v_wiki500_yelp_embed_nontrainable.h5', custom_objects={'KerasLayer': hub.KerasLayer})
for layer in model.layers:
    try:
        if(layer.name == 'keras_layer_1'):
            layer.set_weights(tf.dtypes.cast(model.get_layer(name=layer.name).get_weights(),tf.double))
        else:
            layer.set_weights([tf.dtypes.cast(model.get_layer(name=layer.name).get_weights()[0],tf.double),tf.dtypes.cast(model.get_layer(name=layer.name).get_weights()[1],tf.double)])
    except:
        print("Could not transfer weights for layer {}".format(layer.name))
model.save('w2v_wiki500_yelp_embed_nontrainable_double.h5')