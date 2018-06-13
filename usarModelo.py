import json

import keras.preprocessing.text as kpt
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# we're still going to use a Tokenizer here, but we don't need to fit it
tokenizer = Tokenizer(num_words=4000)
# for human-friendly printing
labels = ['negativo', 'positivo']

# read in our saved dictionary
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)


def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' no existe en el corpus de entrenamiento; ignorada." %(word))
    return wordIndices


stemmer = SnowballStemmer('english', True)
stop = set(stopwords.words('english'))

# and create a model from that
model = load_model(input("Archivo del modelo: "))

# okay here's the interactive part
while 1:
    evalSent = input('Introduce una frase (en inglés), o pulsa Enter para salir: ')

    if len(evalSent) == 0:
        break

    evalSent = evalSent.lower().strip()
    for sword in stop:
        evalSent = evalSent.replace(' ' + sword.lower() + ' ', ' ')

    # format your input for the neural net
    evalSentence = ''
    for palabra in str(evalSent).split(' '):
        palabra = stemmer.stem(palabra)
        evalSentence += (palabra + ' ')
        del palabra

    testArr = convert_text_to_index_array(evalSentence)
    inp = tokenizer.sequences_to_matrix([testArr], mode='binary')
    # predict which bucket your input belongs in
    pred = model.predict(inp)
    # and print it for the humons
    print("Sentimiento %s; %f%% de seguridad (%s %f%%)" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100,
                                                       labels[np.argmin(pred)], pred[0][np.argmin(pred)] * 100))
