from keras.models import Model, model_from_json
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data()
idx = imdb.get_word_index()
idx2word = {v: k for k, v in idx.iteritems()}

print(' '.join([idx2word[o] for o in x_train[0]]))