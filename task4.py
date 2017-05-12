import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.contrib.learn as lrn

from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD



def writeOut(model):
    X,Y = getData()
    model.fit(X,Y,batch_size=300, epochs=15 )
    X_df = pd.read_hdf('test.h5')
    X_real = X_df.as_matrix(X_df.columns[0:129])
    Y_real = model.predict(X_real, batch_size=300)
    out = X_df.ix[:,0]
    out.name = 'y'
    out.loc[30000:] = Y_real
    out.to_csv('submission1_py.csv', index=True, header=True, index_label='Id')

def getData():
    df = pd.read_hdf('train_labeled.h5', 'train')
    X_index = df.columns[1:129]
    X = df.as_matrix(X_index)
    Y = df['y'].as_matrix()
    return X,Y

X,Y = getData()
scaler = StandardScaler()
scaler.fit(X)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, random_state=42)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def model():
    model = Sequential()
    model.add(Dense(units=5000, input_dim=128, activation='relu'))
    model.add(Dense(units=2000, activation='tanh'))
    model.add(Dense(units=2000, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    sgd = SGD(lr=0.20, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
'''
model = model()
model.fit(x_train, y_train, epochs=5, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print("Loss and metrics: ")
print(loss_and_metrics)
'''
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

pipe = KerasClassifier(build_fn=model, verbose=1)

batch_size = [300]
epochs = [5]
param_grid = dict(batch_size = batch_size, nb_epoch=epochs)

clf = GridSearchCV(pipe, param_grid, cv=5, verbose=1)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(x_test)
print(classification_report(y_true, y_pred))
print()

writeOut(pipe)

