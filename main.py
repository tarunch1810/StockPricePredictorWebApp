# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 23:57:33 2021

@author: TARUN
"""
import streamlit as st
import nsepy
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

st.title("Stock Price Predictor")
st.markdown("<hr/>",unsafe_allow_html=True)
sname=st.sidebar.text_input('Enter the stock symbol')
st.write (sname)
dur= st.sidebar.slider('No. of Days of historical data to be used', min_value=1, max_value=700, value=300, step=1)
algo_name=st.sidebar.selectbox("Select Algorithm",("Random Forest","Linear Regression","SVM","ARIMA"))


def gethdata(stockname,d):
    today=datetime.date.today()
    duration=d
    duration2=10
    start=today+datetime.timedelta(-duration)
    end1=today
    stockdata=nsepy.get_history(symbol=stockname,start=start,end=end1)
    return stockdata
rf=gethdata(sname,dur)
st.write(rf)
st.set_option('deprecation.showPyplotGlobalUse', False)
gv=st.multiselect("Select the feature to be represented in graph",("Prev Close","High","Deliverable Volume",'VWAP','Low','Trades','Open','Trades','Turnover'),["High"])
if not gv:
        st.error("Please select at least one feature.")
a=[]
i=0
t=rf.index.values
for dt in t:
    a.append(dt.strftime("%m/%d/%Y"))
fig, ax = plt.subplots(figsize=(40, 20))
xd = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in a]
ax = plt.axes()
ax.set_facecolor("#1b1e38")
ax.plot(xd, rf[gv],'-o',label=gv)
plt.xlabel('Dates')
ax.grid()
plt.legend()
st.pyplot()

params=dict()
  
       
     



# PREDICTION
if algo_name=="Random Forest":
    n_estimator= st.sidebar.slider("n_estimators",min_value=1,max_value=100,step=1)
    params[n_estimator]=n_estimator
    ndh=rf[['High']]
    v=ndh.values
    v=np.append(v, [0])
    v=np.delete(v,0)
    ndh2=rf[['Low']]
    v2=ndh2.values
    v2=np.append(v2, [0])
    v2=np.delete(v2,0)
    rf['ndhigh']=v
    rf['ndlow']=v2 
    rf.reset_index(inplace = True)
    fs=st.multiselect("Select the features to be used as input",("Prev Close","High","Last","Close","Deliverable Volume",'Volume','%Deliverble','VWAP','Low','Open','Trades','Turnover'),['High'])
    X= rf[fs]   
    Y=rf['ndhigh']
    X2= rf[fs]  
    Y2=rf['ndlow']
    Xtestn=X.iloc[-1:]
    Xtestnl=X2.iloc[-1:]
    X.drop(index=X.index[-1], axis=0, inplace=True)
    Y.drop(index=Y.index[-1], axis=0, inplace=True)
    X2.drop(index=X2.index[-1], axis=0, inplace=True)
    Y2.drop(index=Y2.index[-1], axis=0, inplace=True)
    tst_sze=st.slider("Testing data size",min_value=0.1,max_value=1.0,step=0.1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = tst_sze, random_state=1234)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size =tst_sze, random_state=1234)
    regressor = RandomForestRegressor(n_estimators= params[n_estimator])
    regressor2 = RandomForestRegressor(n_estimators= params[n_estimator])
    regressor.fit(X_train,Y_train)
    regressor2.fit(X2_train,Y2_train)
    test_data_prediction = regressor.predict(X_test)
    test_data_prediction2 = regressor2.predict(X2_test)
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    error_score2 = metrics.r2_score(Y2_test, test_data_prediction2)
    st.write("R squared error for HIGH : ", error_score)
    st.write("R squared error for LOW : ", error_score2)
    
    # Plotting prediction_graph
    Y_test = list(Y_test)
    Y2_test = list(Y2_test)
    plt.subplots(figsize=(20, 10))   
    ax = plt.axes()
    ax.set_facecolor("#1b1e38")
    
    plt.plot(Y_test,'-o', color='blue', label = 'Actual high Value')
    plt.plot(test_data_prediction,'-o', color='red', label='Predicted high Value')
    plt.plot(Y2_test,'-x', color='green', label = 'Actual low Value')
    plt.plot(test_data_prediction2,'-x', color='orange', label='Predicted low Value')
    plt.title('Actual Price vs Predicted Price')
    plt.xlabel('Number of values')
    
    
    plt.ylabel('HIGH & LOW Price')
    ax.grid()
    plt.legend()
    st.pyplot()
    test_data_predictionmon = regressor.predict(Xtestn)
    test_data_predictionlow = regressor2.predict(Xtestnl)
    st.write("Tomorrow's PREDICTED HIGH : ",test_data_predictionmon[0])
    st.write("Tomorrow's PREDICTED LOW : ",test_data_predictionlow[0])

elif algo_name=="Linear Regression":
    today=datetime.date.today()    
    duration2= st.sidebar.slider('No. of Days of testing data to be used:', min_value=1, max_value=700, value=50, step=1)
    start=today+datetime.timedelta(-dur)
    end1=today+datetime.timedelta(-duration2)
    lstockdata=nsepy.get_history(symbol=sname,start=start,end=end1)  
    lrf=lstockdata
    ndh=lrf[['High']]
    v=ndh.values
    v=np.append(v, [0])
    v=np.delete(v,0)
    ndh2=lrf[['Low']]
    v2=ndh2.values
    v2=np.append(v2, [0])
    v2=np.delete(v2,0)
    lrf['ndhigh']=v
    lrf['ndlow']=v2 
    lrf.reset_index(inplace = True)
    fs=st.multiselect("Select the features to be used as input",("Prev Close","High","Last","Close","Deliverable Volume",'Volume','%Deliverble','VWAP','Low','Open','Trades','Turnover'),['High'])
    X= lrf[fs]   
    Y=lrf['ndhigh']
    X2= lrf[fs]  
    Y2=lrf['ndlow']
    Xtestn=X.iloc[-1:]
    Xtestnl=X2.iloc[-1:]
    X.drop(index=X.index[-1], axis=0, inplace=True)
    Y.drop(index=Y.index[-1], axis=0, inplace=True)
    X2.drop(index=X2.index[-1], axis=0, inplace=True)
    Y2.drop(index=Y2.index[-1], axis=0, inplace=True)
    X_train=X
    Y_train=Y
    X2_train=X2
    Y2_train=Y2    
    lr = LinearRegression()
    z=lr.fit(X_train,Y_train)
    lr2 = LinearRegression()
    z2=lr2.fit(X2_train,Y2_train)
    today=datetime.date.today()
       
    start=today+datetime.timedelta(-duration2)    
    stockdatalt=nsepy.get_history(symbol=sname,start=start,end=today)
    df1=stockdatalt
    test=df1[fs]    
    ndhi=df1[['High']]
    v11=ndhi.values
    v11=np.append(v11, [0])
    v11=np.delete(v11,0)
    ndhi2=df1[['Low']]
    v22=ndhi2.values
    v22=np.append(v22, [0])
    v22=np.delete(v22,0)
    test['ND high']=v11
    test['ND LOW']=v22
    df1.reset_index(inplace = True)
    testxh= df1[fs]
    lphigh=z.predict(testxh)
    testxl= df1[fs]
    lplow=z2.predict(testxl)
    Xtestn=X.iloc[-1:]
    test['high prediction']=lphigh
    test['low prediction']=lplow
    #prediction df
    Xtesthl=test.iloc[-1:] 
    test.drop(index=test.index[-1],axis=0, inplace=True)
    error_score = metrics.r2_score(test['ND high'],  test['high prediction'])
    error_score2 = metrics.r2_score( test['ND LOW'], test['low prediction'])
    st.write("R squared error for HIGH : ", error_score)
    st.write("R squared error for LOW : ", error_score2)
    # plotting data
    a=[]
    i=0
    t=test.index.values
    for dt in t:
        a.append(dt.strftime("%m/%d/%Y"))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = plt.axes()
    ax.set_facecolor("#1b1e38")
    xd = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in a]
    ax.plot(xd, test['ND high'],'-o',label = "Actual next day High", color = 'red')
    ax.plot(xd, test['high prediction'],'-o', label = "predicted high by the model", color = 'orange')
    ax.plot(xd, test['ND LOW'],'-o',label = "Actual next day low", color = 'yellow')
    ax.plot(xd, test['low prediction'],'-o', label = "predicted low by the model", color = 'blue')
    plt.title('Actual Price vs Predicted Price')
    ax.grid()
    plt.legend()
    st.pyplot()
    st.write("Tomorrow's PREDICTED HIGH : ",Xtesthl['high prediction'].reset_index(drop=True).to_string(index=False))
    st.write("Tomorrow's PREDICTED LOW : ",Xtesthl['low prediction'].reset_index(drop=True).to_string(index=False))
else:
    st.error("ALGORITHM not added")