# load_dataset.py
from __future__ import print_function
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

def shape(v):
  return v.shape

print (shape(X_train), shape(y_train))