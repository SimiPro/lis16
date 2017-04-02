import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler,MinMaxScaler, Normalizer
from sklearn.linear_model import Ridge, HuberRegressor, LassoCV, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier

from sklearn.preprocessing import FunctionTransformer

import pandas as pd
import numpy as np


def transform_y(Y_t):
    return Y_t
def retransform_y(Y_p):
    return Y_p


def log_some_cols(X):
    X[:, 1] = np.log10(X[:,1])
    return X

global degree
degree = 3

#create data framce containing your data, column can be accesesed
# by df['column name']
df = pd.read_csv('train.csv')

X_index = df.columns[2:17]

X = df.as_matrix(X_index)
Y = df['y'].as_matrix()



class MyColLogTransformer(TransformerMixin):
    def __init__(self, colsToLog):
        self.columns = colsToLog
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for z in range(self.columns.size):
            i = self.columns[z]
            X[:,i] = np.log(abs(X[:,i]))
        return X

class MyColumnTransformer(TransformerMixin):
    def __init__(self, i):
        self.column = i
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        newX = np.delete(X, self.column, 1)
        return newX

def removeAboveAndBelow(under, over, X_train, Y_train):
    indexes = np.array([])
    for i in range(Y_train.size):
        if Y_train[i] < under or  Y_train[i] > over:
            #print("Y_train[{:d}] |  {:d} > {:f} > {:d}".format(i, under, Y_train[i],  over))
            indexes = np.append(indexes, i)


    newX = np.delete(X_train, indexes, 0)
    newY = np.delete(Y_train, indexes, 0)
   # print("outliners removed: {:d} | new max: {:f} | new min: {:f}".format(indexes.size, max(newY), min(newY)))

    return newX, newY

#alpha = 0.1 10% removed
def removeOutliners(X_trainz , Y_trainz ,Y_predz, alpha=0.1):
    if alpha == 0:
        return X_trainz, Y_trainz # if alpha = 0 no change

    err = np.square(Y_trainz - Y_predz)
    parts = int(err.size*alpha)
    indexToRemove = np.argpartition(err, -parts)[-parts:]
    newX_trainz = np.delete(X_trainz, indexToRemove,0)
    newY_trainz = np.delete(Y_trainz, indexToRemove,0)
    return newX_trainz, newY_trainz

# every index has to be proven twice before we remove it
marked = np.array([]) # for deleted Cols
markedL = np.array([]) # for log cols
def removeAllExcept(array, indexes, marked):
    toDel = np.array([])
    for i in range(array.size):
        if not (array[i] in indexes):
            if i in marked:
                toDel = np.append(toDel, i)
            else:
                marked = np.append(marked, i)
    return np.delete(array, toDel), marked

def fitz(X_train, Y_train, toDel, colsToLog, X_test):
    estimators = []
    estimators.append(('preproc_minmax', MinMaxScaler((-1,1))))
    estimators.append(('polynomic', PolynomialFeatures(degree=degree)))
    estimators.append(('delete_some_cols', MyColumnTransformer(toDel)))
    estimators.append(('log_some_cols', MyColLogTransformer(colsToLog)))
    estimators.append(('ridge', Ridge(alpha=.1)))
    model = Pipeline(estimators)

    #X_train,Y_train = removeAboveAndBelow(-120, 220, X_train, Y_train)
    model.fit(X_train, Y_train)
    Y_pred = retransform_y(model.predict(X_test))

    return Y_pred, model

def writeOut(model):
    X_df =  pd.read_csv('test.csv')
    X_real = X_df.as_matrix(X_df.columns[1:17])
    Y_real = model.predict(X_real)
    out = X_df['Id']
    out = pd.concat([out, pd.DataFrame({'y' : Y_real})], 1)
    out.to_csv('submission1_py.csv', index=False)

def printEachRelation(X_test, Y_test, Y_pred):
    n = Y_pred.size
    resiuds = Y_pred - Y_test
    plt.figure(0)
    plt.scatter(Y_pred, resiuds, c='b')
    plt.hlines(y=0, xmin=-200, xmax=300)
    plt.xlabel('Y Pred')
    plt.ylabel('Residuals')

    for l in X_test.shape[1]:
        plt.figure(l+1)
        plt.scatter(new_Y_pred_mod, X_transformed[:,l])
        plt.xlabel('X:{:d}'.format(l))
        plt.ylabel('Residual')
        plt.hlines(y=0, xmin=-100, xmax=220)

def getColsToDelete():
    return [ 60 ,86 ,307, 323, 409, 474, 534, 577, 595, 637, 654, 778, 779]
# first try alpha = .2[155, 164, 217, 274, 290, 291, 295, 348, 368, 380, 403, 414, 457, 481, 503, 505, 507, 598, 613, 640, 737, 764, 790]

def getColsToLog():
    return []



poly = PolynomialFeatures(degree)
X_trans = poly.fit_transform(X)
m = X_trans.shape[1]
#m = m - len(getColsToDelete()) # remove deleted cools
print('features: {:d}'.format(m))


# we use kfold to remove some features which do badly in every round
# first we have all and then we remove everytime each one which isnt in the selected bad features
allBad = np.array(range(m))

# we use this array to check which features has a log relationship
logs = np.array(range(m))

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    toBeat = 0.

    # set to true if u want to find out which cols to delete
    # if false set the columns in badF to the cols u want to delete
    deleteCols = False

    # set to true to find out which cols to log
    logCols = False

    if not deleteCols:
        badF  = np.array(getColsToDelete())
    else:
        badF = np.array([])

    if not logCols:
        logs = np.array(getColsToLog())
    else:
        logs = np.array([])

    Y_train = transform_y(Y_train)

    for i in range(1,2):#range(m + 2):#range(1,2):

        if i == m + 1 or not deleteCols:
            toDel = badF
        elif i == 0:
            # do nothing on first run just remember value to Beat
            toDel = np.array([])
        else:
            toDel = i-1

        if i == m + 1 or not logCols:
            colsToLog = logs
        elif i == 0:
            colsToLog = np.array([])
        else:
            colsToLog = np.array([i-1])

        # delete column in toDel and fit
        Y_pred_mod, model = fitz(X_train, Y_train, toDel, colsToLog, X_test)

        rsme_model = mean_squared_error(Y_test, Y_pred_mod)**0.5
        if (toBeat == 0):
            toBeat = rsme_model

        #print("current rsme: {:f}".format(rsme_model))
        # if we're better with the deleted column set it on the black list
        if (rsme_model < toBeat and m+1 != i and deleteCols):
            badF = np.append(badF, i-1)
        if (logCols and rsme_model < toBeat and m+1 != i):
            colsToLog = np.append(colsToLog, i-1)

    print("rsme: {:f}".format(rsme_model))
    if deleteCols:
        print("cols deleted: ")
        print(badF)
    if logCols:
        print("cols logarithmed:")
        print(colsToLog)

    # in all bad there just the bad ones left whom who increase our rsme
    if deleteCols:
        allBad, marked  = removeAllExcept(allBad, badF, marked)

    # in all logs remove the ones which were worse twice
    if logCols:
        logs, markedL = removeAllExcept(logs, logCols, markedL)

if deleteCols:
    print('we should just remove:')
    print(allBad)
if logCols:
    print('Cols we should log:')
    print(logs)

Y_pred, model = fitz(X_train, Y_train, allBad, logs, X_test)
writeOut(model)

plt.figure(0)
plt.scatter(Y_pred, Y_pred - Y_test, c ='b')
plt.hlines(y=0, xmin=-200, xmax=300)
plt.xlabel('Y pred')
plt.ylabel('residuals')

print("rsme before remove outliners:{:f}".format(mean_squared_error(Y_test, Y_pred)**0.5) )

X_train, Y_train = removeOutliners(X_train, Y_train, model.predict(X_train), .1)
Y_pred, model = fitz(X_train, Y_train, allBad, logs, X_test)
print("rsme after removal outliners:{:f}".format(mean_squared_error(Y_test, Y_pred)**0.5) )

#plt.figure(1)
plt.scatter(Y_pred, Y_pred - Y_test, c= 'g')
plt.hlines(y=0, xmin=-200, xmax=300)
plt.xlabel('Y pred')
plt.ylabel('residuals')


plt.show()
