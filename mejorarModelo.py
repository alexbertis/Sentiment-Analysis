import json

import keras
import keras.preprocessing.text as kpt
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")

max_words = 4000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)

with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)


def convert_text_to_index_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

allWordIndices = np.asarray(allWordIndices)

train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
del tokenizer

train_y = keras.utils.to_categorical(train_y, 2)

model = load_model('model2.h5')

model.fit(train_x, train_y, batch_size=5000, epochs=15, verbose=1, validation_split=0.1)

model.save(input('Archivo donde guardar el modelo: '))

print('¡Guardado!')
