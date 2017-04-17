# achieve 92% accuracy after 100 epoches
import numpy as np
import os as os
from keras.models import Model, model_from_json, save_model, load_model
from keras.layers import Input, Dense
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import Sequential
from keras import optimizers

filename = './dump-model.h5'

if not os.path.exists(filename):
  model = Sequential(name='dump-model')
  model.add(Dense(1, input_shape=(3, )))
  sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  # sgd = optimizers.Adagrad(lr=1.0)
  model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
else:
  model = load_model(filename)

print(model.optimizer.weights)

x = np.random.rand(500, 3)
w = np.array([2, 1.4, 4.0])
y = x.dot(w) + 1.24

model.fit(x, y, verbose=0, nb_epoch=1, batch_size=2)
loss = model.evaluate(x, y)

print(loss, model.get_weights());
print(model.optimizer.weights)
save_model(model, filename)
