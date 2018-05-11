from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf


# Read the data into a list of strings.
from scipy.spatial.distance import dice


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    text_file = open(filename, "r")
    text = text_file.read().split()
    return text


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def collect_data(vocabulary_size=10000):
    url = 'http://mattmahoney.net/dc/'
    filename = "textfiles/ptb.train.txt"
    vocabulary = read_data(filename)
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary


vocab_size = 10000
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocab_size)
print(data[:7])

window_size = 3
vector_dim = 300
epochs = 10

sampling_table = sequence.make_sampling_table(vocab_size)
print("start skipgram")
couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
print("skipgram done")
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(word_target)
print(word_context)



# create some input variables
input_target = Input((1,))
input_context = Input((1,))
print("input_target : ", input_target)

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
print("target :", target)
print("target shape before reshape : ", target.shape)
target = Reshape((vector_dim, 1))(target)
print("target shape after reshape : ", target.shape)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# setup a cosine similarity operation which will be output in a secondary model
similarity = merge([target, context], mode='cos', dot_axes=0)

# now perform the dot product operation to get a similarity measure
dot_product = merge([target, context], mode='dot', dot_axes=1)
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)
# create the primary training model
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')




arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))

save_file = open("saveFile.txt", "w")
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    save_file.write(str(arr_1) + " ")
    save_file.write(str(arr_2) + " ")
    save_file.write(str(arr_3) + "\n")
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))

gimme = model.predict(dictionary.get(40))
print("prediction von :" + dictionary.get(40) + " : " + reverse_dictionary.get(gimme))
save_file.close()
print("finished")