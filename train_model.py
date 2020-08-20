import os

import numpy as np
import sklearn.preprocessing
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.concatenate([x_train, x_test])
y_train = np.concatenate([y_train, y_test])
x_train = tf.keras.utils.normalize(x_train, axis=1)


def one_hot(input_array):
    a = input_array
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(a) + 1))
    b = label_binarizer.transform(a)
    return b


x_train = np.expand_dims(x_train, axis=3)
x_train[0][0][0][0] = x_train[0][0][0]

y_train = one_hot(y_train)

# amount of dense layers
dense_layers = [1]
# number of nodes in dense and conv layers
layer_sizes = [128]
# amount of conv layers
conv_layers = [2]
# amount if images to process before doing backpropagation
batchSize = 32
# number of times the network passes through each piece of unique data
hm_epochs = 1


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for _ in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(10))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],)

            model.fit(x_train, y_train,
                      batch_size=32,
                      epochs=hm_epochs,
                      validation_split=0.15)

model.save("MNIST-digits-model")
