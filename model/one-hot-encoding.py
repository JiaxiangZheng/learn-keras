import numpy as np
import keras as keras
from keras.utils.np_utils import to_categorical

y = np.random.randint(0, 4, size=10)

print('keras version %s' % keras.__version__)

# perform one-hot encoding
# this will transform a vector of 10 elements to a matrix with shape (10, 4)
print(to_categorical(y))
