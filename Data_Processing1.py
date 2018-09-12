import pandas as pd
from numpy import NaN
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

df1 = pd.read_csv("C:/Users/zy108/Desktop/9733_machine learning/project/data1.csv")

date_num_list = list(set(list(df1['date_num'])))
date_num_list.sort()
print(date_num_list)
min_date_num = date_num_list[0]
max_date_num = date_num_list[-1]

pseudo_index_list = list(range(int(min_date_num * 10), int(max_date_num * 10 + 1), 1))
index_list = []
for item in pseudo_index_list:
    index_list.append(float(item) / 10)

print(len(index_list))
sum_date = len(index_list)
columns = ['date_num', 'date', 'best_ask_price', 'best_ask_size', 'best_bid_price', 'best_bid_size', 'ask_total_size', 'bid_total_size',
           'ask_weighted_average_price', 'bid_weighted_average_price', 'best_mid_price', 'mid_weighted_average_price']
df = pd.DataFrame(np.arange(0, sum_date * 12).reshape(sum_date, 12), index = index_list, columns= columns)
for item in columns:
    df[item] = 0

df1.index = df1['date_num']
for i in date_num_list:
    print(i)
    df2 = df1[df1['date_num'] == i]
    date = list(df2['date'])[0]
    best_bid_list = list(df2['best_bid_price'])
    bid_size_list = list(df2['best_bid_size'])
    best_bid_price = max(best_bid_list)
    best_bid_size = 0
    for j in range(len(best_bid_list)):
        if best_bid_list[j] == best_bid_price:
            best_bid_size = best_bid_size + bid_size_list[j]

    best_ask_list = list(df2['best_ask_price'])
    ask_size_list = list(df2['best_ask_size'])
    best_ask_price = max(best_ask_list)
    best_ask_size = 0
    for j in range(len(best_ask_list)):
        if best_ask_list[j] == best_ask_price:
            best_ask_size = best_ask_size + ask_size_list[j]

    ask_total_size_list = list(df2['ask_total_size'])
    bid_total_size_list = list(df2['bid_total_size'])
    ask_total_size = sum(ask_total_size_list)
    bid_total_size = sum(bid_total_size_list)

    num_sum = 0
    ask_weighted_list = list(df2['ask_weighted_average_price'])
    for j in range(len(ask_weighted_list)):
        num_sum = num_sum + ask_weighted_list[j] * ask_total_size_list[j]

    ask_weighted_average_price = float(num_sum) / ask_total_size

    num_sum = 0
    bid_weighted_list = list(df2['bid_weighted_average_price'])
    for j in range(len(bid_weighted_list)):
        num_sum = num_sum + bid_weighted_list[j] * bid_total_size_list[j]

    bid_weighted_average_price = float(num_sum) / bid_total_size

    best_mid_price = np.mean([best_ask_price, best_bid_price])

    mid_weighted_average_price = np.mean([ask_weighted_average_price, bid_weighted_average_price])
    df.loc[i, 'date_num'] = i
    df.loc[i, 'date'] = date
    df.loc[i, 'best_ask_price'] = best_ask_price
    df.loc[i, 'best_ask_size'] = best_ask_size
    df.loc[i, 'best_bid_price'] = best_bid_price
    df.loc[i, 'best_bid_size'] = best_bid_size
    df.loc[i, 'ask_total_size'] = ask_total_size
    df.loc[i, 'bid_total_size'] = bid_total_size
    df.loc[i, 'ask_weighted_average_price'] = ask_weighted_average_price
    df.loc[i, 'bid_weighted_average_price'] = bid_weighted_average_price
    df.loc[i, 'best_mid_price'] = best_mid_price
    df.loc[i, 'mid_weighted_average_price'] = mid_weighted_average_price

df.to_csv("C:/Users/zy108/Desktop/9733_machine learning/project/data2.csv", index=False)

for i in index_list[::-1]:
    flag = df.loc[i, 'date_num']
    if flag == 0:
        df.loc[i, 'date_num'] = i
        timeStamp = i
        d = datetime.datetime.utcfromtimestamp(timeStamp)
        date = str(d.strftime("%Y-%m-%d %H:%M:%S.%f"))
        df.loc[i, 'date'] = date
        df.loc[i, 'best_ask_price'] = best_ask_price
        df.loc[i, 'best_ask_size'] = best_ask_size
        df.loc[i, 'best_bid_price'] = best_bid_price
        df.loc[i, 'best_bid_size'] = best_bid_size
        df.loc[i, 'ask_total_size'] = ask_total_size
        df.loc[i, 'bid_total_size'] = bid_total_size
        df.loc[i, 'ask_weighted_average_price'] = ask_weighted_average_price
        df.loc[i, 'bid_weighted_average_price'] = bid_weighted_average_price
        df.loc[i, 'best_mid_price'] = best_mid_price
        df.loc[i, 'mid_weighted_average_price'] = mid_weighted_average_price
    else:
        best_ask_price = df.loc[i, 'best_ask_price']
        best_ask_size = df.loc[i, 'best_ask_size']
        best_bid_price = df.loc[i, 'best_bid_price']
        best_bid_size = df.loc[i, 'best_bid_size']
        ask_total_size = df.loc[i, 'ask_total_size']
        bid_total_size = df.loc[i, 'bid_total_size']
        ask_weighted_average_price = df.loc[i, 'ask_weighted_average_price']
        bid_weighted_average_price = df.loc[i, 'bid_weighted_average_price']
        best_mid_price = df.loc[i, 'best_mid_price']
        mid_weighted_average_price = df.loc[i, 'mid_weighted_average_price']


df['0.1'] = df['best_mid_price'].shift(-1)
df['0.5'] = df['best_mid_price'].shift(-5)
df['1'] = df['best_mid_price'].shift(-10)
df['5'] = df['best_mid_price'].shift(-50)
df['10'] = df['best_mid_price'].shift(-100)
df['30'] = df['best_mid_price'].shift(-300)

df['output_0.1'] = df.apply(lambda x: 1 if x['0.1'] > x['best_mid_price'] else -1 if x['0.1'] < x['best_mid_price'] else 0, axis=1)
df['output_0.5'] = df.apply(lambda x: 1 if x['0.5'] > x['best_mid_price'] else -1 if x['0.5'] < x['best_mid_price'] else 0, axis=1)
df['output_1'] = df.apply(lambda x: 1 if x['1'] > x['best_mid_price'] else -1 if x['1'] < x['best_mid_price'] else 0, axis=1)
df['output_5'] = df.apply(lambda x: 1 if x['5'] > x['best_mid_price'] else -1 if x['5'] < x['best_mid_price'] else 0, axis=1)
df['output_10'] = df.apply(lambda x: 1 if x['10'] > x['best_mid_price'] else -1 if x['10'] < x['best_mid_price'] else 0, axis=1)
df['output_30'] = df.apply(lambda x: 1 if x['30'] > x['best_mid_price'] else -1 if x['30'] < x['best_mid_price'] else 0, axis=1)

df = df.loc[:, columns + ['output_0.1', 'output_0.5', 'output_1', 'output_5', 'output_10', 'output_30']]
df.to_csv("C:/Users/zy108/Desktop/9733_machine learning/project/data.csv", index=False)

print(df)