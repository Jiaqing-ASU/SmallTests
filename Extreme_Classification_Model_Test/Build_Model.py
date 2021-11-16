import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import os

#w1 = tf.Variable(tf.random.normal([597540,1000], stddev = 2, mean = 0, seed =1))
#b1 = tf.Variable(tf.random.normal([1000,], stddev = 2, mean = 0, seed =1))

#w2 = tf.Variable(tf.random.normal([1000,14588], stddev = 2, mean = 0, seed =1))
#b2 = tf.Variable(tf.random.normal([14588,], stddev = 2, mean = 0, seed =1))

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(597540,), sparse=True))
model.add(layers.Dense(units=1000, use_bias=True, activation='relu'))
model.add(layers.Dense(units=14588, use_bias=True, activation=None))

#model.layers[0].set_weights([w1, b1])
#model.layers[1].set_weights([w2, b2])

for layer in model.layers:
    a,b = layer.get_weights()[0].shape
    w = tf.random.normal([a,b], stddev = 2, mean = 0, seed =1)
    b = tf.random.normal((layer.get_weights()[1].shape), stddev = 2, mean = 0, seed =1)
    layer.set_weights([w, b])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.save('extreme_classification_model.h5')