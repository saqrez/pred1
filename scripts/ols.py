# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:24:29 2020

@author: saquib
"""

import sys, os
import numpy as np
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import timeit
import datetime
from statsmodels.formula.api import ols
from scipy import stats

from pandasql import sqldf

def daytype__Wednesday(date_fld):
    date = pd.to_datetime(date_fld)
    if date.weekday() == 2 :
        return 1
    else:
        return 0


def daytype__Week_End(date_fld):
    date = pd.to_datetime(date_fld)
    if date.weekday() in [5,6] :
        return 1
    else:
        return 0
    
    
def splevent(ds):
    date = pd.to_datetime(ds)
    if date in ((datetime.datetime(2020,1,22),datetime.datetime(2020,1,23),datetime.datetime(2020,1,24),datetime.datetime(2020,1,25),datetime.datetime(2020,1,26))) :
        return 1
    else:
        return 0  
    
def eltag (elasticity):
    if elasticity>0:
        return 'inelastic'
    elif -1<elasticity<0:
        return 'low elastic'
    elif -2<elasticity<-1:
        return 'moderate elastic'
    elif elasticity<-2:
        return 'high elastic'
    else:
        return 'unexplained'

def confid (disc_t):
    if np.abs(disc_t)>=1.65:
        return 'High'
    else:
        return 'Low'
 
###  Train Data Preparation for entire period ##########
train=pd.DataFrame()
start_date = datetime.date(2019, 1, 1)
end_date   = datetime.date(2020, 1, 15)
train['date'] = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days)+1)]
train['date']=pd.to_datetime(train['date'])
#train['month']=train['date'].dt.month
#train['month'] = train['month'].astype('category')

##################  Creating Furure Dataframe  ############################
futuredf=pd.DataFrame()
start_date = datetime.date(2019, 1, 1)
end_date   = datetime.date(2020, 2, 29)
futuredf['date'] = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days)+1)]
futuredf['date']=pd.to_datetime(futuredf['date'])
futuredf['wednesday'] = futuredf['date'].apply(daytype__Wednesday)
futuredf['weekend'] = futuredf['date'].apply(daytype__Week_End) 
futuredf['month']=futuredf['date'].dt.month
futuredf['month'] = futuredf['month'].astype('category')
    


##########################  ols prediction at format level ######################333
def ols():
     
    
     print("i was clicked")
     df2 = data_ols[data_ols['product_desc']==product.value]
     df3 = df2[df2['city']==city.value]

     data_ols['wednesday'] = data_ols['date_fld'].apply(daytype__Wednesday)
     data_ols['weekend'] = data_ols['date_fld'].apply(daytype__Week_End)
     data_ols['tot_qty'] = pd.to_numeric(data_ols['tot_qty'])
     data_ols['discount'] = pd.to_numeric(data_ols['discount'])
     data_ols['net_sales'] = pd.to_numeric(data_ols['net_sales'])
     data_ols['disc_pct']=data_ols['discount']/(data_ols['net_sales']+data_ols['discount'])
     data_ols['net_price']=(data_ols['net_sales'])/(data_ols['tot_qty'])
     data_ols['y']=data_ols['tot_qty']
     data_ols['ds']=data_ols['date_fld']
     data_ols['month']=data_ols['date_fld'].dt.month
     data_ols['month'] = data_ols['month'].astype('category')
    # mcal=pd.read_excel('mcal.xlsx')
    # promo=mcal[['promo']]
    # data_ols['holiday'] = np.where(data_ols['date_key'].isin(promo['promo']), 1, 0)
     data_ols=data_ols[(data_ols['date_key']>=20190101) & (data_ols['date_key']<20200115)]
     data_ols=data_ols[data_ols['y']>0]
     data_ols=data_ols[data_ols['net_sales']>0]
     data_ols['mop']=pd.to_numeric(data_ols['mop'])
     data_ols['zone_qty']=pd.to_numeric(data_ols['zone_qty'])
     data_ols=data_ols[data_ols['mop']>0]
     data_ols=data_ols[data_ols['discount']>=0]
    
           
     test=data_ols
    
    ############Removing outliers ###############
    # test = test[np.isfinite(test['net_price'])]         
    # test['z'] = np.abs(stats.zscore(test['tot_qty']))      
    # test=test[test['z']<3 ]
    #################################################
    
     mop=np.float(max(test['mop'].tail(28)))
     ftfl= np.mean(test['footfall'].tail(14))
     mcqty= np.mean(test['zone_qty'].tail(14))
    
     mindt=min(test['ds'])
     mxdt=max(test['ds'])
     count_days = len(test['ds'].unique())
     traindata=pd.merge(train,test, how="left",left_on="date",right_on="ds")
    #        traindata['disc_pct']=traindata['disc_pct'].fillna(0)
     traindata['disc_pct']=traindata['disc_pct'].fillna(method="ffill")
     traindata['disc_pct']=traindata['disc_pct'].fillna(method="bfill")
     traindata['y']=traindata['y'].fillna(0)
       
     traindata=traindata.fillna(method="ffill")
     traindata=traindata.fillna(method="bfill")
    # traindata['y_lag1y']=traindata['y'].shift(364)
     traindata['y_lag2w']=traindata['y'].shift(28)
    # traindata=traindata.fillna(method="ffill")
    # traindata=traindata.fillna(method="bfill")
     traindata['y_avg_4w']=traindata['y'].rolling(28).mean()
     traindata=traindata.dropna()
     traindata['ds']=traindata['date']
      
     traindata2=traindata[(traindata['date']>= pd.datetime(2019,1,1))]
    
     traindata2=traindata2[(traindata2['ds']>=mindt)]
     traindata2['wednesday'] = traindata2['date_fld'].apply(daytype__Wednesday)
     traindata2['weekend'] = traindata2['date_fld'].apply(daytype__Week_End)
      
     fit = ols('y ~ C(month)+disc_pct+y_lag2w+y_avg_4w+footfall', traindata2).fit()     
    
     for pct in (0.0,0.05):
         fdf=pd.DataFrame()
         futuredf['ds']=futuredf['date']
         future = pd.merge(futuredf,traindata[['ds','y','zone_qty']],on='ds', how='left')
            # future['y_lag1y']=future['y'].shift(364)
         future['y']=future['y'].fillna(0)
         future['y_lag2w']=future['y'].shift(14)
         future['y_avg_4w']=future['y'].rolling(28).mean()
         future['mc_avg_qty']=mcqty
         future=future.fillna(method="ffill")
         future=future.drop(['y'],axis=1)
         future=future.dropna()
         future['holiday']=future['ds'].apply(splevent)
            # future=future[(future['date']>=datetime.datetime(2020,2,1))]
    #                        futuredf['net_price']=(mop-(mop*pct))
         future['mop']=mop-(mop*pct)
         future['disc_pct']=pct
         future['footfall']=ftfl
            
    #                    X=future[['ds','mop','disc_pct','wednesday','weekend','footfall','y_lag1y','y_lag2w','month']]
    
    #                    X=future[['ds','disc_pct','footfall','y_lag2w','y_lag1y','month','mc_qty']]
            
         X=future[['ds','disc_pct','footfall','y_lag2w','y_avg_4w','month','zone_qty','mop','wednesday','weekend','holiday']]
    #                        Xnew=np.column_stack((X['wednesday'],X['weekend'],X['disc_pct']))
    #                        Xnew = sm.add_constant(X)
         fdf['pred'+str(pct)]=fit.predict(X)
    #                    futuredf.set_index('date', inplace=True)
         frcs_df2=pd.concat((frcs_df2,fdf), axis=1)
         print ("forecast Data is ready")        

 