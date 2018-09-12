# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:07:12 2018

@author: 2012
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

length = 59368
test_size = 0.3
test_length = int(length * test_size)
time = [0.1, 0.5, 1, 5, 10, 30]

df1 = pd.read_csv('data.csv')
data = df1.tail(test_length + 1).head(test_length - 299) 

method = ['SVM', 'RF', 'DT', 'DL', 'LOG', 'SGD', 'LIN']
name = ['SVM', 'Random Forest', 'Decision Tree', 'Neural Network',
        'Logistic Regression', 'Stochastic Gradient Descent', 'Linear Regression']

def result_plot(method_name, ni, n_predictions = 5000):
    path = method_name
    df = pd.read_excel(path +'/output.xlsx').tail(test_length).head(test_length - 299) 
    actual = data.iloc[:,13:19]
    n, m = df.shape
    
    error = df - actual
    num_succ = (error == 0).astype(int).sum(axis=0) / n
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.yaxis.grid(True, color = 'gainsboro',linestyle = '--', zorder=0)
    ax.bar(np.arange(m), num_succ.values, width=0.3, align='center', color='darkblue', zorder=3)
    vals = ax.get_yticks()
    ax.xaxis.set_ticks(np.arange(m))
    ax.xaxis.set_ticklabels(time)
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
    title = "Success after " + str(n_predictions) + " predictions (" + name[ni] + ")"
    ax.set_title(title, fontdict = {'fontsize':22})
    
    ret_second = []
    for i in range(m):
        j = int(time[i] / time[0])
        temp = j + n_predictions
        sign = df.iloc[0:n_predictions, i]
        signal = pd.DataFrame(sign.values, columns = ['signal'])
        price = df1.tail(test_length + 1)[0:temp][['best_mid_price']]
        ret = pd.DataFrame(price['best_mid_price'].diff(j), columns = ['best_mid_price'])
        price = price[0:n_predictions]
        ret = ret[np.isfinite(ret['best_mid_price'])]
        signal.index, ret.index = price.index, price.index
        ret = ret.div(price, axis='best_mid_price')
        ret['best_mid_price'] = ret['best_mid_price'] * signal['signal']
        ret['best_mid_price'] = ret['best_mid_price'] + 1
        compound = ret['best_mid_price'].astype(object).product()
        ret_second.append(float(compound ** (8 * 3600 / n_predictions) - 1))
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.yaxis.grid(True, color = 'gainsboro',linestyle = '--', zorder=0)
    ax.xaxis.grid(True, color = 'gainsboro',linestyle = '--', zorder=0)
    ax.plot(np.arange(m), ret_second, color='darkblue', zorder=3)
    vals = ax.get_yticks()
    ax.xaxis.set_ticks(np.arange(m))
    ax.xaxis.set_ticklabels(time)
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
    title = "Return after " + str(n_predictions) + " predictions (" + name[ni] + ")"
    ax.set_title(title, fontdict = {'fontsize':22})
    return ret_second

res = []
for i in method:
    temp = result_plot(i, ni = method.index(i))
    temp = [x - 5e-3 for x in temp]
    res.append(temp)
colors = ['navy', 'dodgerblue', 'steelblue', 'lightseagreen', 'mediumseagreen',
          'g', 'yellowgreen']
m, width = 6, 0.12
x = np.arange(m) - len(method) / 2 *width     
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
ax.yaxis.grid(True, color = 'gainsboro',linestyle = '--', zorder=0)

for i in range(len(res)):
    ax.bar(x + i*width, res[i], width=0.1, align='center',
           color=colors[i], zorder=3)
vals = ax.get_yticks()
ax.xaxis.set_ticks(np.arange(m))
ax.xaxis.set_ticklabels(time)
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
ax.legend(name)
ax.set_title("Summary of the methods", fontdict = {'fontsize':22})

    
    
