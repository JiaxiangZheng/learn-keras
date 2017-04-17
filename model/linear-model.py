# achieve 92% accuracy after 100 epoches
import numpy as np
import os as os
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(Dense(1, input_shape=(10, )))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# sgd = optimizers.Adagrad(lr=1.0)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

x = np.random.rand(5000, 10)
w = np.array([2, 1.4, 4.0, 1.0, 3.0, 2.0, 0.05, 10.0, 1000, 0.0])
y = x.dot(w) + 1.24

model.fit(x, y, verbose=1, nb_epoch=1, batch_size=2)
loss = model.evaluate(x, y)

print(loss, model.get_weights());
