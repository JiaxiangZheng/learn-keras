"""Sequential is a simple keras model representation, internally it use Model to construct.
But for the high level, it's stored as a linear list rather than a graph. For the Model itself,
it can be used for constructing models with complexity but lacks with simple usage.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Reshape, Dense

width, height = 8, 8

model = Sequential()
model.add(Reshape((width * height, ), input_shape=(width, height)))
model.add(Dense(4))

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
model.summary()

input = np.random.randn(1, width, height)
output = np.array([[0, 0, 1, 0]])
history = model.fit(input, output, nb_epoch=100, verbose=0)

predict = model.predict(input)

print(predict)
print(output)

diff = model.evaluate(input, output)
print(diff)