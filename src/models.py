# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import InputLayer

def deep_cnn_sequential(features_shape, num_classes, act='relu'):
    """ CNN model for a single drum instrument training.
    May use the same model for e.g. snare, bass drum and hi-hat onset training.
    """
    model = keras.Sequential()

    model.add(InputLayer(name='inputs', input_shape=features_shape, dtype='float32'))
    # Block 1
    model.add(tf.keras.layers.Conv2D(10, (3, 7), activation='relu', padding='same', strides=1, name='block1_conv', input_shape=features_shape))
    model.add(tf.keras.layers.MaxPooling2D((3, 1), strides=(2,2), padding='same', name='block1_pool'))
    model.add(tf.keras.layers.BatchNormalization(name='block1_norm'))
    
    # Block 2
    model.add(tf.keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same', strides=1, name='block2_conv'))
    model.add(tf.keras.layers.MaxPooling2D((3, 1), strides=(2,2), padding='same', name='block2_pool'))
    model.add(tf.keras.layers.BatchNormalization(name='block2_norm'))

    # Flatten
    model.add(tf.keras.layers.Flatten(name='flatten'))
    
    # Fully connected layer 1
    model.add(tf.keras.layers.Dense(256, activation='relu', name='dense'))
    model.add(tf.keras.layers.BatchNormalization(name='dense_norm'))
    model.add(tf.keras.layers.Dropout(0.5, name='dropout'))
    
    # Prediction (Fully connected layer 2)
    # 2 predictions: onset or no onset
    model.add(tf.keras.layers.Dense(num_classes, activation=act, name='pred'))

    # Print network summary
    model.summary()

    return model
