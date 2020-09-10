# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:36:32 2020

@author: luizgomes
"""
SEED = 1480
from numpy.random import seed
seed(SEED)
from tensorflow.random import set_seed
set_seed(SEED+1)

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras


def vectorize_text(text, size, max, mode='sequence'):

    tokenizer = Tokenizer(
        num_words=size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

    tokenizer.fit_on_texts(text)
    if mode == 'tfidf':
        X = tokenizer.texts_to_matrix(text, mode="tfidf")
        X = X[:, 1:max+1]
    else:
        X = tokenizer.texts_to_sequences(text)
        X = pad_sequences(X, maxlen=max)

    return X


docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better']


def make_model(i_dim, o_dim, i_length, mode='sequence'):
    if (mode == 'sequence'):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(i_dim, o_dim, input_length=i_length))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1, activation='sigmoid'))
    else:
        model = keras.Sequential([
            keras.layers.LSTM(units=128, input_shape=i_length, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(units=128, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2, activation='softmax')
        ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    return model

#docs = ['a b b c c c!', 'a a a b']
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y = pd.get_dummies(labels).values

print(vectorize_text(docs, size=0, max=10, mode='tfidf'))

# pad documents to a max length of 4 words
max_length = 4
vocab_size = 50

# modeling and predicting using sequence
x = vectorize_text(docs, size=vocab_size, max=max_length, mode='sequence')
model = make_model(vocab_size, 8, max_length)
model.fit(x, y.argmax(axis=1), epochs=50, verbose=True)
loss, accuracy = model.evaluate(x, y.argmax(axis=1), verbose=True)
print(x)
print(model.summary())
print('Accuracy %f' % (accuracy * 100))

# modeling and predicting using tdidf
x = vectorize_text(docs, size=vocab_size, max=max_length, mode='tfidf')
x = x[:, :, None]
model = make_model(vocab_size, 8, x.shape[1: ], mode='tfidf')

model.fit(x, y.argmax(axis=1), epochs=50, verbose=True)
loss, accuracy = model.evaluate(x, y.argmax(axis=1), verbose=True)
print(x)
print(model.summary())
print('Accuracy %f' % (accuracy * 100))