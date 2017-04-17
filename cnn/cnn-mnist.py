# achieve 98% after 10 epoches
import numpy as np
from keras.models import Model, model_from_json
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense, Reshape
from keras.models import Sequential
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Reshape((1, 28, 28), input_shape=(28, 28)))

model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Conv2D(16, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, verbose=1, nb_epoch=2, batch_size=32, validation_data=(x_test, y_test))
sgd.lr = sgd.lr * 0.5;
history = model.fit(x_train, y_train, verbose=1, nb_epoch=2, batch_size=32, validation_data=(x_test, y_test))
sgd.lr = sgd.lr * 0.5;
history = model.fit(x_train, y_train, verbose=1, nb_epoch=2, batch_size=32, validation_data=(x_test, y_test))
sgd.lr = sgd.lr * 0.5;
history = model.fit(x_train, y_train, verbose=1, nb_epoch=2, batch_size=32, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

model.save('cnn-mnist.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

