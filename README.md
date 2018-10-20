# Machine-Learning
## Introduction
This is the final project of Machine Learning Course
The project aims to predict the funture price of one security based on the historical intraday data with different kinds of machine learning models.
## Data Processing
The raw data is in the data_proejct2.zip. Each row in the excel include the time, bid or ask, price and volume. And I used this original data to create features for machine learning models. 

The minimum time interval of the time series data is 0.1 second which means from 1/21/2018  10:20:02 PM to 1/21/2018  11:58:59 PM, every 0.1 second we have data. Therefore, the original data might have multiple data in the same 0.1-second, we combined them together in the same 0.1-second. What’s more, the original data might have missing data in the some 0.1-second times, we used the future nearest data as the data in that 0.1-second.
Therefore, There are ten features in total which are best_ask_price, best_bid_price, mid_price, best_ask_size(size of best ask price), best_bid_size(size of best bid price), total_ask_size(sum of all the ask size), total_bid_size(sum of all the bid size), ask_weighted_average_price(weighted average price of ask calculated by sum(ask price * ask size) / total_ask_size), bid_weighted_average_price(weighted average price of bid calculated by sum(bid price * bid size) / total_bid_size), mid_weighted_average_price. These things are included in data.csv
## Linear Regression
Linear Model performs poorly in this project. Classifier instead of Regression should be more adaptable in the prediction of simple 'up or down' problem.
## Logistics Regression
Logistic Regression is supposed to perform better than Linear Regression. I got a relatively better prediction using Logistic Regression.
However, Regression is not well adaptable to the prediction of 'up or down' problem. Also, due to the large size of train data, logistic regression is computational intensive. The number of iterations should be reduced.
## Stochastic Gradient Descent
SGD Method can apply both SVM and logistic regression to calculate the loss function. Linear SVM uses I2 regularization, while logistic regression uses I1 regularization.
Linear SVM performs better than logistic regression.
SGDclassifier can be regarded as a linear classification method.
## Support Vector Machine
Model parameter tuning: For this set of data, kernel seems to have a dominant influence on the prediction accuracy. We chose the kernel 'rbf', which is the only kernel actually works in an acceptable amount of running time. For kernel 'linear', 'sigmoid' and 'poly', it is very slow in terms of running time. As for the 'C' and 'gamma', different parameters tend to give the same results. The parameters I chose are {'kernel':'rbf', 'C'= 1, 'gamma'= 0.2) 
## Decision Tree
Pruning the trees: In order to avoid overfitting and underfitting, after trying several values for parameters, I chose max depth to be 10, min leaves to be 5 and max leaves to be 10.
## Random Forest
Parameters: The values of parameters I chose for random forest model are: number of trees to be 30, and maximum features to be 9.
## Neural Network
Parameters: I calibrate model parameters, such as hidden layers’ numbers, solver and activation functions, to realize better prediction. In conclusion, I choose relu activation function, adam solver and two hidden layers as 12 and 8.


