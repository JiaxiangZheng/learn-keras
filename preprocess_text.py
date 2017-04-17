# preprocess_text.py
from __future__ import print_function
from keras.preprocessing import text
from keras.preprocessing import image

seq = text.text_to_word_sequence('VGG-16 is my favorite image classification model to run because of its simplicity and accuracy. The creators of this model published a pre-trained binary that can be used in Caffe.')
print(seq)

img = image.load_img('FX6ROg9.jpg')
img = image.img_to_array(img)
print(img.shape)