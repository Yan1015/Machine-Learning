import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime


def split(item1, item2):
    size1_list = []
    size2_list = []
    price1_list = []
    price2_list = []
    item1 = item1.split(' ')
    item2 = item2.split(' ')
    timeStamp = round(float(item1[0]) / 1000000, 1)
    date_num = timeStamp
    d = datetime.datetime.utcfromtimestamp(timeStamp)
    date = str(d.strftime("%Y-%m-%d %H:%M:%S.%f"))

    for i in range(1, int(len(item1) / 2)):
        price1_list.append(int(item1[2 * i]))
        size1_list.append(int(item1[2 * i + 1]))


    for i in range(1, int(len(item2) / 2)):
        price2_list.append(int(item2[2 * i]))
        size2_list.append(int(item2[2 * i + 1]))


    best_ask_price = price2_list[0]
    best_ask_size = size2_list[0]
    best_bid_price = price1_list[0]
    best_bid_size = size1_list[0]
    ask_total_size = sum(size2_list)
    bid_total_size = sum(size1_list)

    total_sum = 0
    for j in range(len(price2_list)):
        total_sum = total_sum + price2_list[j] * size2_list[j]
    ask_weighted_average_price = total_sum / ask_total_size

    total_sum = 0
    for j in range(len(price1_list)):
        total_sum = total_sum + price1_list[j] * size1_list[j]
    bid_weighted_average_price = total_sum / bid_total_size

    best_mid_price = np.mean([best_ask_price, best_bid_price])

    mid_weighted_average_price = np.mean([ask_weighted_average_price, bid_weighted_average_price])

    return [date_num, date, best_ask_price, best_ask_size, best_bid_price, best_bid_size, ask_total_size, bid_total_size,
           ask_weighted_average_price, bid_weighted_average_price, best_mid_price, mid_weighted_average_price]


df1 = pd.read_csv("C:/Users/zy108/Downloads/data_project2.csv", header=None)
columns = ['date_num', 'date', 'best_ask_price', 'best_ask_size', 'best_bid_price', 'best_bid_size', 'ask_total_size', 'bid_total_size',
           'ask_weighted_average_price', 'bid_weighted_average_price', 'best_mid_price', 'mid_weighted_average_price']
date_num = int(len(df1[0]) / 2)

df = pd.DataFrame(np.arange(0, date_num * 12).reshape(date_num, 12), index = range(date_num), columns= columns)

for i in range(date_num):
    item1 = df1[0][2 * i]
    item2 = df1[0][2 * i + 1]

    item_list = split(item1, item2)

    for j in range(len(item_list)):
        df.loc[i, columns[j]] = item_list[j]

df.to_csv("C:/Users/zy108/Desktop/9733_machine learning/project/data1.csv", index=False)
print(df)




