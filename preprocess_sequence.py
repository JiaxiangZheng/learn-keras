# preprocess_sequence.py
from __future__ import print_function
from keras.preprocessing import sequence

np_sequences = sequence.pad_sequences([[1, 2], [3, 4], [5, 6]], maxlen=10, dtype='int32')
print(np_sequences)

sampling_table = sequence.make_sampling_table(10, sampling_factor=1e-5)
print(sampling_table)

skipgrams = sequence.skipgrams("the rain in Spain falls mainly on the plain", 2)
print(skipgrams)
