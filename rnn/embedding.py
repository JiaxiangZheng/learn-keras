import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Sequential

input_length = 5
input = np.random.randint(1000, size=(300, input_length))
model = Sequential()
model.add(Embedding(1000, 64, input_length=input_length))
model.compile('rmsprop', 'mse')

output = model.predict(input)
print(output.shape)
print(output[0, 0, :])
print(output[0, 1, :])