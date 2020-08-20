import cv2
import tensorflow as tf
import numpy as np
import os

def decode(x):
    return np.argmax(x)

def prepare(filepath):
    IMG_SIZE = 28
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model("MNIST-digits-model")

number_dirs = ["zero.png","one.png","two.png",
               "three.png","four.png","five.png",
               "six.png","seven.png","eight.png",
               "nine.png"]

for number in number_dirs:

    one_hot = model.predict([prepare(os.path.join("test-images",number))])[0]
    print(number + " : " + str(decode(one_hot)))


