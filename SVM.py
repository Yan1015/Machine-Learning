
# coding: utf-8

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt



df = pd.read_csv("/Users/xuqing/Desktop/Tandon 2018 Spring /ml/project/data.csv")
df = df.iloc[:,1:]



X1 = df.iloc[:-1,[1,3]]



def get_train_test_split(X,y,a):
    test_size = int(len(X)*a)
    X_train,X_test,y_train,y_test = X[0:test_size],X[test_size:len(X)],y[0:test_size],y[test_size:len(X)]
    return X_train,X_test,y_train,y_test

from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
X1 = df.iloc[:-2,[1,3]]
y1 = df.iloc[:-2,11]


X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

X1 = df.iloc[:-2,[1,2,3,4]]
y1 = df.iloc[:-2,11]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

X1 = df.iloc[:-2,[1,2,3,4,5,6]]
y1 = df.iloc[:-2,11]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

X1 = df.iloc[:-2,[7,9]]
y1 = df.iloc[:-2,11]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)


X1 = df.iloc[:-2,[7,8,9,10]]
y1 = df.iloc[:-2,11]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

X1 = df.iloc[:-2,1:11]
y1 = df.iloc[:-2,11]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)


def get_train_test_split(X,y,a):
    test_size = int(len(X)*a)
    X_train,X_test,y_train,y_test = X[0:test_size],X[test_size:len(X)],y[0:test_size],y[test_size:len(X)]
    return X_train,X_test,y_train,y_test

from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
X1 = df.iloc[:-301,[1,3]]
y1 = df.iloc[:-301,16]


X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

X1 = df.iloc[:-301,[1,2,3,4]]
y1 = df.iloc[:-301,16]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

X1 = df.iloc[:-301,[1,2,3,4,5,6]]
y1 = df.iloc[:-301,16]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

X1 = df.iloc[:-301,[7,9]]
y1 = df.iloc[:-301,16]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)


X1 = df.iloc[:-301,[7,8,9,10]]
y1 = df.iloc[:-301,16]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

X1 = df.iloc[:-301,1:11]
y1 = df.iloc[:-301,16]

X_train,X_test,y_train,y_test = get_train_test_split(X1,y1,0.7)

svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train) 
y_pred = svc.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)



# we can see that the 3rd, 5th and 6th are the same
# So we just need to compare the 3 pairs remaining

X1 = df.iloc[:-2,[1,3]]
y1 = df.iloc[:-2,11]
X2 = df.iloc[:-6,[1,3]]
y2 = df.iloc[:-6,12]
X3 = df.iloc[:-11,[1,3]]
y3 = df.iloc[:-11,13]
X4 = df.iloc[:-51,[1,3]]
y4 = df.iloc[:-51,14]
X5 = df.iloc[:-101,[1,3]]
y5 = df.iloc[:-101,15]
X6 = df.iloc[:-301,[1,3]]
y6 = df.iloc[:-301,16]

datas = [[X1,y1],[X2,y2],[X3,y3],[X4,y4],[X5,y5],[X6,y6]]

ret = {}


scores = []
for data in datas:
    X_train,X_test,y_train,y_test = get_train_test_split(data[0],data[1],0.7)
    svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
avg = np.average(scores)

avg



X1 = df.iloc[:-1,[1,2,3,4]]
y1 = df.iloc[:-1,11]
X2 = df.iloc[:-5,[1,2,3,4]]
y2 = df.iloc[:-5,12]
X3 = df.iloc[:-10,[1,2,3,4]]
y3 = df.iloc[:-10,13]
X4 = df.iloc[:-50,[1,2,3,4]]
y4 = df.iloc[:-50,14]
X5 = df.iloc[:-100,[1,2,3,4]]
y5 = df.iloc[:-100,15]
X6 = df.iloc[:-300,[1,2,3,4]]
y6 = df.iloc[:-300,16]

datas = [[X1,y1],[X2,y2],[X3,y3],[X4,y4],[X5,y5],[X6,y6]]

ret = {}


scores = []
for data in datas:
    X_train,X_test,y_train,y_test = get_train_test_split(data[0],data[1],0.7)
    svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
avg = np.average(scores)

avg



X1 = df.iloc[:-1,[1,2,3,4,5,6]]
y1 = df.iloc[:-1,11]
X2 = df.iloc[:-5,[1,2,3,4,5,6]]
y2 = df.iloc[:-5,12]
X3 = df.iloc[:-10,[1,2,3,4,5,6]]
y3 = df.iloc[:-10,13]
X4 = df.iloc[:-50,[1,2,3,4,5,6]]
y4 = df.iloc[:-50,14]
X5 = df.iloc[:-100,[1,2,3,4,5,6]]
y5 = df.iloc[:-100,15]
X6 = df.iloc[:-300,[1,2,3,4,5,6]]
y6 = df.iloc[:-300,16]

datas = [[X1,y1],[X2,y2],[X3,y3],[X4,y4],[X5,y5],[X6,y6]]

ret = {}


scores = []
for data in datas:
    X_train,X_test,y_train,y_test = get_train_test_split(data[0],data[1],0.7)
    svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
avg = np.average(scores)

avg



X1 = df.iloc[:-1,[1,2,3,4,5,6]]
y1 = df.iloc[:-1,11]
X2 = df.iloc[:-5,[1,2,3,4,5,6]]
y2 = df.iloc[:-5,12]
X3 = df.iloc[:-10,[1,2,3,4,5,6]]
y3 = df.iloc[:-10,13]
X4 = df.iloc[:-50,[1,2,3,4,5,6]]
y4 = df.iloc[:-50,14]
X5 = df.iloc[:-100,[1,2,3,4,5,6]]
y5 = df.iloc[:-100,15]
X6 = df.iloc[:-300,[1,2,3,4,5,6]]
y6 = df.iloc[:-300,16]

datas = [[X1,y1],[X2,y2],[X3,y3],[X4,y4],[X5,y5],[X6,y6]]

ret = {}


scores = []
for data in datas:
    X_train,X_test,y_train,y_test = get_train_test_split(data[0],data[1],0.7)
    svc = svm.SVC(kernel = 'linear',C = 1, gamma = 0.2).fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
avg = np.average(scores)

avg



X1 = df.iloc[:-2,[7,9]]
y1 = df.iloc[:-2,11]
X2 = df.iloc[:-6,[7,9]]
y2 = df.iloc[:-6,12]
X3 = df.iloc[:-11,[7,9]]
y3 = df.iloc[:-11,13]
X4 = df.iloc[:-51,[7,9]]
y4 = df.iloc[:-51,14]
X5 = df.iloc[:-101,[7,9]]
y5 = df.iloc[:-101,15]
X6 = df.iloc[:-301,[7,9]]
y6 = df.iloc[:-301,16]

datas = [[X1,y1],[X2,y2],[X3,y3],[X4,y4],[X5,y5],[X6,y6]]

ret = {}


scores = []
for data in datas:
    X_train,X_test,y_train,y_test = get_train_test_split(data[0],data[1],0.7)
    svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
avg = np.average(scores)

avg




# The best features are [1,2,3,4,5,6]
def get_train_test_split(X,y,a):
    test_size = int(len(X)*a)
    X_train,X_test,y_train,y_test = X[0:test_size],X[test_size:len(X)],y[0:test_size],y[test_size:len(X)]
    return X_train,X_test,y_train,y_test
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score

X1 = df.iloc[:-1,[1,2,3,4,5,6]]
y1 = df.iloc[:-1,11]
X2 = df.iloc[:-5,[1,2,3,4,5,6]]
y2 = df.iloc[:-5,12]
X3 = df.iloc[:-10,[1,2,3,4,5,6]]
y3 = df.iloc[:-10,13]
X4 = df.iloc[:-50,[1,2,3,4,5,6]]
y4 = df.iloc[:-50,14]
X5 = df.iloc[:-100,[1,2,3,4,5,6]]
y5 = df.iloc[:-100,15]
X6 = df.iloc[:-300,[1,2,3,4,5,6]]
y6 = df.iloc[:-300,16]

datas = [[X1,y1],[X2,y2],[X3,y3],[X4,y4],[X5,y5],[X6,y6]]

ret = {}



y_preds = []

for data in datas:
    X_train,X_test,y_train,y_test = get_train_test_split(data[0],data[1],0.7)
    svc = svm.SVC(kernel = 'rbf',C = 1, gamma = 0.2).fit(X_train,y_train)
    y_pred = svc.predict(data[0])
    y_preds.append(y_pred)

columns = ['output0.1','output0.5','output1','output5','output10','output30']

df1 = pd.DataFrame(columns = columns)
df1.iloc[:,0] = y_preds[0]

#df1.iloc[:len(y_preds[1]),1] = y_preds[1]

#df1.iloc[:len(y_preds[2]),2] = y_preds[2]

for i in np.arange(1,6):
    df1.iloc[:len(y_preds[i]),i] = y_preds[i]

df1
    
    
writer = pd.ExcelWriter('/Users/xuqing/Desktop/Tandon 2018 Spring /ml/project/output.xlsx')
df1.to_excel(writer,'Sheet1')
writer.save()   



# In[9]:

df1


# In[34]:

columns = ['output0.1','output0.5','output1','output5','output10','output30']

df1 = pd.DataFrame(columns = columns)
df1.iloc[:,0] = y_preds[0]

#df1.iloc[:len(y_preds[1]),1] = y_preds[1]

#df1.iloc[:len(y_preds[2]),2] = y_preds[2]

for i in np.arange(1,6):
    df1.iloc[:len(y_preds[i]),i] = y_preds[i]


writer = pd.ExcelWriter('/Users/xuqing/Desktop/Tandon 2018 Spring /ml/project/output.xlsx')
df1.to_excel(writer,'Sheet1')
writer.save()



