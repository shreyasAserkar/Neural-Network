#!/usr/bin/env python
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import csv
lst = []
# fix random seed for reproducibility
seed = 7
dataType = None
try:
    f = open('transposedDataset.csv', 'w')
    with open('free-zipcode-database.csv') as g:
        gr = csv.reader(g)
        for x in zip(*gr):
            for y in x:
                f.write(y + ',')
            f.write('\n')
        f.close()
    numpy.random.seed(seed)
    # load transposed dataset
    dataset = numpy.loadtxt("transposedDataset.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:-1]
    Y = dataset[:,-1]
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=len(X), init='uniform', activation='relu'))
    model.add(Dense(len(X), init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=10,  verbose=2)
    # calculate predictions
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
except Exception as ex:
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)


