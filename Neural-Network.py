import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import confusion_matrix

data = pd.read_csv('/Users/CherieZ/Desktop/data.csv')
data.head()

interval = [0.1,0.5,1,5,10,30]
pred_results = []
for i in range(6):
    num = int(interval[i]*10)
    
    X = np.array(data[['best_ask_size','best_bid_size','ask_total_size','bid_total_size','ask_weighted_average_price',
                       'bid_weighted_average_price','best_ask_price','best_bid_price']])
    Y = np.array(data['output_%s'%interval[i]])
    sample = len(X)
    train_num = int(sample*0.7)
    X = X[:-num]
    Y = Y[:-num]
    X_train = X[:train_num]
    Y_train = Y[:train_num]
    X_test = X[train_num:]
    Y_test = Y[train_num:]
    XOR_MLP = MLP(activation='relu', alpha=1e-05, batch_size='auto', early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(12,8,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    XOR_MLP.fit(X_train,Y_train)
    Y_pred = XOR_MLP.predict(X_test)
    pred_results.append(Y_pred)
    
    print(pred_results)
    predictions = pd.DataFrame(pred_results).T
    predictions.columns = ['output%s'%i for i in interval]
    predictions.to_csv('predictions1.csv', encoding='utf-8', index=False)
