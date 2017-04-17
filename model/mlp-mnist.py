# achieve 96% after 100 epoches
import numpy as np
from keras.models import Model, model_from_json
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.core import Lambda, Flatten, Dense, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import optimizers
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Reshape((784, ), input_shape=(28, 28)))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

# model.add(Dense(512))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax', input_shape=(784, )))
model.summary()

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, verbose=1, nb_epoch=12, batch_size=128, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

