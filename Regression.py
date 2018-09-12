
# coding: utf-8




import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso





data=pd.read_csv('data.csv')
del data['mid_weighted_average_price']





def LinearModel(model,data,interval):#using LinearRegression and LogisticRegression
    clf=model
    train=data.iloc[0:int(data.shape[0]*0.7)]#split data to 70% train and 30% test
    test=data.iloc[int((data.shape[0])*0.7+1):-1]
    test.index=range(int(data.shape[0]*0.3-1))
    y_train=train['best_mid_price'].iloc[interval:-1]#set 'best_mid_price' after a certain interval as the target values
    y_test=test['best_mid_price'].iloc[interval:-1]
    clf.fit(train.iloc[0:-interval-1,2:10],y_train)
    predict1=clf.predict(train.iloc[0:-interval-1,2:10])#for calculating the bias
    predict2=clf.predict(test.iloc[0:-interval-1,2:10])#predict the target values in test data
    

    xx=np.zeros(y_train.shape[0])#collecting the bias for each record
    for i in range(y_train.shape[0]):
        xx[i]=predict1[i]-y_train.iloc[i]

    


    xx2=np.zeros(y_test.shape[0])
    yy=np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):#determine the prediction
        xx2[i]=predict2[i]-y_test.iloc[i]
        if xx2[i]<-0.01:
            yy[i]=-1
        elif xx2[i]>0.01:
            yy[i]=1
        else:
            yy[i]=0
    s=0
    index=int(data.shape[0]*0.7+1)
    for i in range(len(yy)):
        if int(yy[i])==data['output_30'].iloc[index+i]:
            s+=1
    
    return yy,s/len(yy)
    





def LinearClassifier(data,interval,column):# using SGDClassifier
    clf=SGDClassifier()
    train=data.iloc[0:int(data.shape[0]*0.7)]
    test=data.iloc[int((data.shape[0])*0.7+1):-1]
    test.index=range(int(data.shape[0]*0.3-1))
    y_train=train.iloc[interval:-1,10+column]
    y_test=test.iloc[interval:-1,10+column]
    clf.fit(train.iloc[0:-interval-1,2:10],y_train)
    predict=clf.predict(test.iloc[0:-interval-1,2:10])
    score=clf.score(test.iloc[0:-interval-1,2:10],y_test)
    return predict,score
    




def ret_cal(seq,data,interval):
    ret=0
    for i in range(seq.shape[0]):
        if seq[i]==-1:
            ret+=(data.iloc[i]-data.iloc[i+interval])/data.iloc[i]
        elif seq[i]==1:
            ret+=(data.iloc[i+interval]-data.iloc[i])/data.iloc[i]
    return ret



output=pd.DataFrame([])
score=np.zeros(18)
s=0
for i in ['1','5','10','50','100','300']:
    yy,score[s]=LinearModel(LinearRegression(),data,int(i))#yy is an array of output, score is the accuracy of prediction
    yy=yy[0:17505]
    output[str(s)]=yy
    yy,score[s+1]=LinearModel(LogisticRegression(max_iter=10),data,int(i))#max_iter cannot be large or it will cost minutes to fit
    yy=yy[0:17505]
    output[str(s+1)]=yy
    yy,score[s+2]=LinearClassifier(data,int(i),int(s/3+1))
    yy=yy[0:17505]
    output[str(s+2)]=yy
    s=s+3
print(score)
output.columns=['lin_0.1','log_0.1','sgd_0.1','lin_0.5','log_0.5','sgd_0.5',
               'lin_1','log_1','sgd_1','lin_5','log_5','sgd_5',
                'lin_10','log_10','sgd_10','lin_30','log_30','sgd_30'
               ]


ret_cal(output.iloc[:,0],data["best_mid_price"].iloc[41559:-1],300)


t=0
for i in ['1','5','10','50','100','300','1','5','10','50','100','300','1','5','10','50','100','300']:
    ret[t]=ret_cal(output.iloc[:,t],data["best_mid_price"].iloc[41559:-1],int(i))
    t+=1




output.loc['return']=ret
output.loc['score']=score




output.to_csv('C:/Users/Hao Li/Desktop/2018 spring/9733/project/lin_log_sgd.csv')

