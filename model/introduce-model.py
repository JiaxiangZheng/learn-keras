from keras.models import Model, model_from_json
from keras.layers import Input, Dense
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import Sequential

# Sequential Type
f = open('./mnist_cnn.json')
model = model_from_json(f.read())
model.summary()

# Model Type
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(input=inputs, output=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
