# binary.classifer.py

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# for a single-input model with 2 classes (binary):
model = Sequential()
model.add(Dense(1, input_dim=784, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data
data = np.random.random((10000, 784))
labels = np.random.randint(2, size=(10000, 1))

# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=20, batch_size=64)
