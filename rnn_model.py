import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import logging
import os
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class RNN():

    def __init__(self):
        self.logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        response_folder = '~/Documents/Metis/project4'
        response_folder = os.path.expanduser(self.response_folder)

    def readinput(self, filename):
        with open(os.path.join(self.response_folder, filename), 'rb') as f:
            self.filename.split('.')[0] = pickle.load(f)

        return self.filename.split('.')[0]

    def sepExp(self, text):
        regex = re.compile(".*?\((.*?)\)")
        expression = re.findall(regex, text)
        expression = ','.join(expression)
        diag_filt = re.sub("[\(\[].*?[\)\]]", "", text)

        return expression, diag_filt

    def processing(self, text):
        chars = sorted(list(set(text)))
        # print(chars)
        char_to_int = dict((c, i) for i, c in enumerate(chars))

        self.n_chars = len(text)
        self.n_vocab = len(chars)
        # print("Total Characters: ", n_chars)
        # print("Total Vocab: ", n_vocab)
        # print(char_to_int)
        return char_to_int

    def rnnPrep(self, length,character=1):  # main_character = ['Phoebe:', 'Rachel:', 'Ross:', 'Monica:', 'Chandler:', 'Joey:']
        self.seq_length = length
        dataX = []
        dataY = []
        raw_text = self.df_diag[character].diag_filt
        df = self.df_diag[character]['char_to_int']

        for i in range(0, self.n_chars - self.seq_length, 1):
            seq_in = raw_text[i:i + self.seq_length]
            # print(seq_in)
            seq_out = raw_text[i + self.seq_length]
            dataX.append([df[char] for char in seq_in])
            dataY.append(df['char_to_int'][seq_out])

        n_patterns = len(dataX)
        print("Total Patterns: ", n_patterns)
        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, self.seq_length, 1))
        # normalize
        X = X / float(self.n_vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)

        return X,y

    def rnnTrain(self,X,y):
        # define the LSTM model
        model = Sequential()
        model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # define the checkpoint
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        model.fit(X, y, epochs=1, batch_size=128, callbacks=callbacks_list)
        return model

    def writeOutput(self, file):
        with open(os.path.join(self.response_folder, f'{file}.plk'), 'wb') as fo:
            pickle.dump(file, fo)


def main():
    diag = RNN()
    df_diag = diag.readinput('df_diag.pkl')
    df_diag.columns = ['diag']
    print('Done input')

    df_diag['expression'], df_diag['diag_filt'] = zip(*df_diag['diag'].map(diag.sepExp))

    df_diag['char_to_int_diag'] = df_diag['diag_filt'].apply(diag.processing)
    df_diag['char_to_int_expr'] = df_diag['expression'].apply(diag.processing)
    print('Done transformation')

    # train Phoebe
    X_ph,y_ph = diag.rnnPrep(100, character=0)
    model_ph = diag.rnnTrain(X_ph,y_ph)
    print('Done')
    diag.writeOutput(model_ph)

    # train Rachel
    X_ra, y_ra = diag.rnnPrep(100, character=1)
    model_ra = diag.rnnTrain(X_ra, y_ra)
    print('Done')
    diag.writeOutput(model_ra)

    # train Ross
    X_ro, y_ro = diag.rnnPrep(100, character=2)
    model_ro = diag.rnnTrain(X_ro, y_ro)
    print('Done')
    diag.writeOutput(model_ro)

    # train Monica
    X_mo, y_mo = diag.rnnPrep(100, character=3)
    model_mo = diag.rnnTrain(X_mo, y_mo)
    print('Done')
    diag.writeOutput(model_mo)

    # train Chandler
    X_ch, y_ch = diag.rnnPrep(100, character=4)
    model_ch = diag.rnnTrain(X_ch, y_ch)
    print('Done')
    diag.writeOutput(model_ch)

    # train Joey
    X_jo, y_jo = diag.rnnPrep(100, character=5)
    model_jo = diag.rnnTrain(X_jo, y_jo)
    print('Done')
    diag.writeOutput(model_jo)


if __name__ == "__main__":
    main()
