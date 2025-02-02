import json

import keras
import keras.preprocessing.text as kpt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# extract data from a csv
# notice the cool options to skip lines at the beginning
# and to only take data from certain columns

training = np.genfromtxt('../web/dataset.csv', encoding="utf8", delimiter=',', skip_header=1, usecols=(1, 3), dtype=None)
numTweets = 700000
training = training[:numTweets]

# Cogemos todos los tweets (mensajes, son la entrada)
# YO QUIERO FILTRAR LAS STOPWORDS
stemmer = SnowballStemmer('english', True)
stop = set(stopwords.words('english'))
stop.add('its')
stop.add('im')
stop.add('mi')
train_x = [x[1] for x in training]

for n, s in enumerate(train_x):
    s = s.lower().strip()
    for sword in stop:
        s = s.replace(' ' + sword.lower() + ' ', ' ')

    nuevaS = ''
    for palabra in str(s).split(' '):
        palabra = stemmer.stem(palabra)
        nuevaS += (palabra + ' ')
        del palabra
    train_x[n] = nuevaS
    del nuevaS

del stemmer
np.save("train_x", train_x)
print("Tweets filtrados")
# Cogemos el valor de su sentimiento (0/1, son la salida)
train_y = np.asarray([y[0] for y in training])
np.save("train_y", train_y)
print("Arrays guardados")

# ¿Ahorrar memoria?
del training
# only work with the 3000 most popular words found in our dataset
max_words = 4000

# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
# Y liberamos RAM
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
del tokenizer

# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, 2)

model = Sequential()
# model.add(LSTM(256, return_sequences=True, input_shape=(max_words,1)))
model.add(Dense(784, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.4))
# model.add(LSTM(512, return_sequences=True))
model.add(Dense(2048))
model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=False))
model.add(Dense(2048))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=5000, epochs=15, verbose=1, validation_split=0.1)

model.save('model.h5')

print('saved model!')
