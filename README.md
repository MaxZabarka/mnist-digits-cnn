# MNIST Digit Classification

MNIST digit classification with Python, Tensorflow, Keras

![This is a alt text.](/mnist_digits.png "MNIST Digits")

## Requirements
* Tensorflow
* Keras
* OpenCV2

## Network
**I found this network to be the one with best results:**

Conv(128,pool_size=(2,2))  
Conv(128,pool_size=(2,2))  
Dense(128)  
Dense(10)  
Batchsize = 32  
Epochs = 5

## Results:
**When run using 60000 training images, a batch size of 32, validation split of 15%, and 5 epochs. Training time of 35 seconds on GTX 1080**
* loss: 0.0035
* accuracy: 0.9988 
* val_loss: 0.0068
* val_accuracy: 0.9979



