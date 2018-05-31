import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils import multi_gpu_model
import types
import tempfile
import keras.models
import string

import sys
import logging
import os
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class RNN():


    def __init__(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.response_folder = '~/Documents/Metis/project4'
        #self.response_folder = '~/Friends_text_generator'
        self.response_folder = os.path.expanduser(self.response_folder)



    def readinput(self, filename):
        with open(os.path.join(self.response_folder, filename), 'rb') as f:
            df = pickle.load(f)

        return df

    def sepExp(self, text):
        regex = re.compile(".*?\((.*?)\)")
        expression = re.findall(regex, text)
        expression = ','.join(expression)
        diag_filt = re.sub("[\(\[].*?[\)\]]", "", text)

        return expression, diag_filt

    def cleandoc(self,text):
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in text]
        # print(tokens)
        tokens = [word for word in tokens if word.isalpha()]

        return tokens

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

    def rnnPrep(self, length,df_diag,character):  # main_character = ['Phoebe:', 'Rachel:', 'Ross:', 'Monica:', 'Chandler:', 'Joey:']
        self.seq_length = length
        self.dataX = []
        self.dataY = []

        raw_text = df_diag.iloc[character].diag_filt
        print(raw_text)
        df = df_diag.iloc[character]['char_to_int_diag']
        # print(df)
        for i in range(0, self.n_chars - self.seq_length, 1):
            seq_in = raw_text[i:i + self.seq_length]
            # print(seq_in)
            seq_out = raw_text[i + self.seq_length]
            self.dataX.append([df[char] for char in seq_in])
            self.dataY.append(df[seq_out])
        # print(self.dataX)
        n_patterns = len(self.dataX)
        print("Total Patterns: ", n_patterns)
        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(self.dataX, (n_patterns, self.seq_length, 1))
        # print(X)
        # normalize
        X = X / float(self.n_vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(self.dataY)

        return X,y

    def rnnTrain(self,X,y):
        # define the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
        self.model.add(Dropout(0.2))
        # model.add(LSTM)
        self.model.add(Dense(y.shape[1], activation='softmax'))
        # parallel_model = multi_gpu_model(model,gpus=2)
        # parallel_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        # define the checkpoint
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        self.model.fit(X, y, epochs=1, batch_size=128, callbacks=callbacks_list)


    # pick a random seed
    def textgenerate(self,df_diag,character):

        chars = sorted(list(set(df_diag.iloc[character].diag_filt)))

        int_to_char = dict((i, c) for i, c in enumerate(chars))
        # start = numpy.random.randint(0, len(self.dataX) - 1)
        pattern = self.dataX[0]
        # print(pattern)
        print("Seed:")
        print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
        # generate characters
        for i in range(100):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.n_vocab)
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_char[index]
            seq_in = [int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        # print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
        print("\nDone.")


    def make_keras_picklable(self):

        def __getstate__(self):
            model_str = ""
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                keras.models.save_model(self, fd.name, overwrite=True)
                model_str = fd.read()
            d = {'model_str': model_str}
            return d

        def __setstate__(self, state):
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                fd.write(state['model_str'])
                fd.flush()
                model = keras.models.load_model(fd.name)
            self.__dict__ = model.__dict__

        cls = keras.models.Model
        cls.__getstate__ = __getstate__
        cls.__setstate__ = __setstate__

    def writeOutput(self, file,filename):


        # self.make_keras_picklable()

        with open(os.path.join(self.response_folder, filename), 'wb') as fo:
            pickle.dump(file, fo)


    def writemodel(self,filename):
        f = open(os.path.join(self.response_folder, filename), 'wb')
        self.model.save(os.path.join(self.response_folder, filename))


def main():
    diag = RNN()
    df_diag = diag.readinput('df_diag.pkl')
    df_diag.columns = ['diag']
    # print(df_diag)
    print('Done input')

    df_diag['expression'], df_diag['diag_filt'] = zip(*df_diag['diag'].map(diag.sepExp))
    df_diag['diag_filt_clean']  = df_diag['diag_filt'].apply(diag.cleandoc)
    df_diag['char_to_int_diag'] = df_diag['diag_filt'].apply(diag.processing)
    # print(df_diag.char_to_int_diag[0])
    df_diag['char_to_int_expr'] = df_diag['expression'].apply(diag.processing)
    print(df_diag)
    print('Done transformation')

    # train Phoebe
    for i in range(0,6):
        X,y = diag.rnnPrep(100, df_diag,i)
        diag.rnnTrain(X,y)

        diag.writeOutput(diag.dataX,f'datax{i}.pkl')
        diag.writeOutput(diag.dataY,f'datay{i}.pkl')
        diag.writemodel(f'model{i}.h5')
        # diag.writeOutput(model,f"{i}.pkl")

        print('Done trining')
        # model = diag.readinput('Phobe.pkl')
        # print('done loading model')
        diag.textgenerate(df_diag,i)


if __name__ == "__main__":
    main()
