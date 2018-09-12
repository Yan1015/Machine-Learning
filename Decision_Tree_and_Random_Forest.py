#import warnings
#warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display
import math
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
#np.set_printoptions(suppress=True)
#DISPLAY_MAX_ROWS = 20 # number of max rows to print for a DataFrame
#pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

data = pd.read_csv("data.csv")
data.head()

X = data[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y = data['output_0.1']

# 0.1s
data0 = data.iloc[:-1,:]
row = len(data0)
training0 = data0.iloc[:int(row*0.7),:]
testing0 = data0.iloc[int(row*0.7):,:]
X_train0 = training0[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_train0 = training0['output_0.1']
X_test0 = testing0[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_test0 = testing0['output_0.1']

maxdepth = 10
minleaves = 5
maxleaves = 10
clf = tree.DecisionTreeClassifier(max_depth = maxdepth, min_samples_leaf = minleaves, 
                                  max_leaf_nodes = maxleaves, criterion = 'gini')
clf.fit(X_train0, Y_train0)
a = clf.predict(X)

# 0.5s
data1 = data.iloc[:-5,:]
row = len(data1)
training1 = data1.iloc[:int(row*0.7),:]
testing1 = data1.iloc[int(row*0.7):,:]
X_train1 = training1[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_train1 = training1['output_0.5']
X_test1 = testing1[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_test1 = testing1['output_0.5']

maxdepth = 10
minleaves = 5
maxleaves = 10
clf = tree.DecisionTreeClassifier(max_depth = maxdepth, min_samples_leaf = minleaves, 
                                  max_leaf_nodes = maxleaves, criterion = 'gini')
clf.fit(X_train1, Y_train1)
b = clf.predict(X)
clf.score(X_test1, Y_test1)

# 1s
data2 = data.iloc[:-10,:]
row = len(data2)
training2 = data2.iloc[:int(row*0.7),:]
testing2 = data2.iloc[int(row*0.7):,:]
X_train2 = training2[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_train2 = training2['output_1']
X_test2 = testing2[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_test2 = testing2['output_1']

maxdepth = 10
minleaves = 5
maxleaves = 10
clf = tree.DecisionTreeClassifier(max_depth = maxdepth, min_samples_leaf = minleaves, 
                                  max_leaf_nodes = maxleaves, criterion = 'gini')
clf.fit(X_train2, Y_train2)
c = clf.predict(X)
clf.score(X_test2, Y_test2)

# 5s
data3 = data.iloc[:-50,:]
row = len(data3)
training3 = data3.iloc[:int(row*0.7),:]
testing3 = data3.iloc[int(row*0.7):,:]
X_train3 = training3[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_train3 = training3['output_5']
X_test3 = testing3[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_test3 = testing3['output_5']

maxdepth = 10
minleaves = 5
maxleaves = 10
clf = tree.DecisionTreeClassifier(max_depth = maxdepth, min_samples_leaf = minleaves, 
                                  max_leaf_nodes = maxleaves, criterion = 'gini')
clf.fit(X_train3, Y_train3)
d = clf.predict(X)
clf.score(X_test3, Y_test3)

# 10s
data4 = data.iloc[:-100,:]
row = len(data4)
training4 = data4.iloc[:int(row*0.7),:]
testing4 = data4.iloc[int(row*0.7):,:]
X_train4 = training4[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_train4 = training4['output_10']
X_test4 = testing4[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_test4 = testing4['output_10']

maxdepth = 10
minleaves = 5
maxleaves = 10
clf = tree.DecisionTreeClassifier(max_depth = maxdepth, min_samples_leaf = minleaves, 
                                  max_leaf_nodes = maxleaves, criterion = 'gini')
clf.fit(X_train4, Y_train4)
e = clf.predict(X)
clf.score(X_test4, Y_test4)

# 30s
data5 = data.iloc[:-300,:]
row = len(data5)
training5 = data5.iloc[:int(row*0.7),:]
testing5 = data5.iloc[int(row*0.7):,:]
X_train5 = training5[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_train5 = training5['output_0.5']
X_test5 = testing5[['best_ask_price','best_ask_size','best_bid_price','best_bid_size','ask_total_size','bid_total_size',
          'ask_weighted_average_price','bid_weighted_average_price', 'mid_weighted_average_price']]
Y_test5 = testing5['output_0.5']

maxdepth = 10
minleaves = 5
maxleaves = 10
clf = tree.DecisionTreeClassifier(max_depth = maxdepth, min_samples_leaf = minleaves, 
                                  max_leaf_nodes = maxleaves, criterion = 'gini')
clf.fit(X_train5, Y_train5)
f = clf.predict(X)
clf.score(X_test5, Y_test5)

table = pd.DataFrame()
table["output0.1"] = a
table["output0.5"] = b
table["output1"] = c
table["output5"] = d
table["output10"] = e
table["output30"] = f
writer = pd.ExcelWriter('DecisionTree.xlsx', engine='xlsxwriter')
table.to_excel(writer,'Decision Tree')
writer.save()

from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import graphviz

def clfTreeVisual(clf_model):   
    dot_data = StringIO()
    tree.export_graphviz(clf_model, out_file = dot_data, feature_names = X.columns,
                    class_names = ['-1','0','1'], filled = True, rounded = True, special_characters = True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    display(Image(graph.create_png()))

clfTreeVisual(clf)

from sklearn.ensemble import RandomForestClassifier
num_trees = 30
max_features = 9
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
model.fit(X_train0, Y_train0)
a = model.predict(X)
model.score(X_test0, Y_test0)

model.fit(X_train1, Y_train1)
b = model.predict(X)
model.score(X_test1, Y_test1)

model.fit(X_train2, Y_train2)
c = model.predict(X)
model.score(X_test2, Y_test2)

model.fit(X_train3, Y_train3)
d = model.predict(X)
model.score(X_test3, Y_test3)

model.fit(X_train4, Y_train4)
e = model.predict(X)
model.score(X_test4, Y_test4)

model.fit(X_train5, Y_train5)
f = model.predict(X)
model.score(X_test5, Y_test5)

table = pd.DataFrame()
table["output0.1"] = a
table["output0.5"] = b
table["output1"] = c
table["output5"] = d
table["output10"] = e
table["output30"] = f
writer = pd.ExcelWriter('RandomForest.xlsx', engine='xlsxwriter')
table.to_excel(writer,'Random Forest')
writer.save()
