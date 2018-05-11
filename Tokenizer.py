from keras.models import Sequential
from keras.preprocessing import text as text
from keras.preprocessing import sequence as sq
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Input, Embedding, Reshape, merge, TimeDistributed, Activation
from keras.models import  Model
from keras.preprocessing.text import Tokenizer
import numpy as np

text_file = open("textfiles/ptb.train.txt", "r")
# params
nb_epoch = 3
# learn `batch_size words` at a time
batch_size = 1500
num_steps=30
hidden_size=500
vec_dim = 128
# half of window
window_size = 8
look_back = 2
num_words = 25000


tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ', char_level=False, oov_token=None)
text = text_file.read().split()

tokenizer.fit_on_texts(text)
print("Tokenizer :", tokenizer.word_index)
save_File = open("tokenizerFile.txt", "w")
save_File.write(str(tokenizer.texts_to_sequences(text)))
save_File.close()
print("skipgram starts")

couples, labels = sq.skipgrams(tokenizer.texts_to_sequences(text), 1500, window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
couples = np.array(couples)
couples = np.reshape(couples, (couples.shape[0], 1, couples.shape[1]))
labels = np.array(labels)


save_File = open("couples.txt", "w")
save_File.write(str(couples))
save_File.close()

save_File = open("labels.txt", "w")
save_File.write(str(labels))
save_File.close()

print("skipgram done")

nb_batch = len(labels) // batch_size
samples_per_epoch = batch_size * nb_batch


"""""

model = Sequential()
model.add(Embedding(tokenizer.num_words,hidden_size , input_length=2))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(tokenizer.num_words)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print("start training")
model.fit(couples,
          labels,
          epochs=100,
          batch_size=batch_size,
          verbose=0)
print("training done")
trainPredict = model.predict(couples)

print(trainPredict)
"""
print("finished")