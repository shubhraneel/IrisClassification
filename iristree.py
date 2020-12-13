import numpy as np
import pandas as pd
import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt

data = pd.read_csv("./iris.data", names = ['sl', 'sw', 'pl', 'pw', 'cl'])
X = data.loc[:, ['sl', 'sw', 'pl', 'pw']].values.tolist()
Y = data.loc[:, 'cl'].values.tolist()
key = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
value = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


#colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
y = [key[k] for k in Y] 
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, shuffle = True)
clf = SVC(kernel='poly', degree=2)
clf = clf.fit(xTrain, yTrain)



y_pred = clf.predict(xTest)
print("Predicted values: ")
#print([value[k] for k in y_pred])
acc = accuracy_score(yTest, y_pred)
print("Accuracy score: ")
print(acc)
 

