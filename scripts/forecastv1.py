# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:25:30 2020

@author: saquib
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:51:35 2020

@author: saquib
"""
import pandas as pd
from bokeh.plotting import figure, show,output_file,curdoc
from bokeh.models import ColumnDataSource,CDSView,DataTable,TableColumn,NumberFormatter,Legend
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel,HTMLTemplateFormatter,DatePicker
from bokeh.palettes import Spectral5,Category10,Viridis256
from bokeh.themes import built_in_themes
from bokeh.models.tools import HoverTool
from collections import OrderedDict
from bokeh.models import LinearAxis, Range1d
from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider, 
								  Tabs, CheckboxButtonGroup, 
								  TableColumn, DataTable, Select,Button,MultiSelect)
from bokeh.models import DaysTicker, DatetimeTickFormatter

# from scripts.ols import ols

import sys, os
from os.path import dirname, join
import numpy as np
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import timeit
import datetime
from datetime import date
import calendar
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
    
 
###  Train Data Preparation for entire period ##########
train=pd.DataFrame()
start_date = datetime.date(2019, 1, 1)
end_date   = datetime.date(2020, 2, 15)
train['date'] = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days)+1)]
train['date']=pd.to_datetime(train['date'])
#train['month']=train['date'].dt.month
#train['month'] = train['month'].astype('category')

##################  Creating Furure Dataframe  ############################
futuredf=pd.DataFrame()
start_date = datetime.date(2019, 1, 1)
end_date   = datetime.date(2020, 3, 31)
futuredf['date'] = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days)+1)]
futuredf['date']=pd.to_datetime(futuredf['date'])
futuredf['wednesday'] = futuredf['date'].apply(daytype__Wednesday)
futuredf['weekend'] = futuredf['date'].apply(daytype__Week_End) 
# futuredf['month']=futuredf['date'].dt.month
# futuredf['month'] = futuredf['month'].astype('category')
    

# #######################################################################################
# # data_ols=pd.read_pickle("./data/data.pkl")
# data_ols = pd.read_pickle(join(dirname(__file__), 'data', 'data.pkl')).dropna()
# data_ols['date_fld']=pd.to_datetime(data_ols['date_fld'])
# data_ols['tot_qty'] = pd.to_numeric(data_ols['tot_qty'])
# data_ols['discount'] = pd.to_numeric(data_ols['discount'])
# data_ols['net_sales'] = pd.to_numeric(data_ols['net_sales'])
# data_ols['disc_pct']=data_ols['discount']/(data_ols['net_sales']+data_ols['discount'])
# data_ols['tot_qty'] = pd.to_numeric(data_ols['tot_qty'])
# data_ols=data_ols[data_ols['city']=='KOCHI']
def forecast(data):
    

    
    data_ols=data
    
    
    prod = []
    prod.append('Select')
    
    prod.extend(data_ols['product_desc'].unique().tolist())
    product = Select(title="Product",  options=sorted(prod), value="Select",background='grey')
    
    city = []
    city.append('Select')
    
    city.extend(data_ols['city'].unique().tolist())
    city = Select(title="City",  options=sorted(city), value="Select",background = "grey")
    
    action=['None','Forecast']
    act=Select(title="Action",options=action, value="Action",background='grey')
    
    pred=["0","5","10","Predict"]
    prd=Select(title="Predict",options=pred, value="Predict",background='grey')
    
    # prodl=['All','None']
    # prod.sort()
    # prod_sel = CheckboxGroup(labels=prodl,active = [0, 1])
    
    dt_pck=DatePicker(title='Select start of training date',min_date=date(2017,1,1),max_date=date.today())
    
    
    # source = ColumnDataSource(data_ols)
    
    tt_categ=[
    ('Date', '@date_fld{%F}'),
    ('Net Sales', '@net_sales'),
    ('Discount', '@discount'),
    
    ]
    tt_prod =[('Date', '@x{%F}'),('Product', '@label'),
          ('Quantity', '$y')]
    						  
    
    f1=figure(x_axis_type='datetime', title="Forecast",tooltips=tt_categ,plot_width=900, plot_height=400)
    # p = figure(x_axis_type='datetime',title="Sales Trend",plot_width=900, plot_height=400)	
    # data_table = DataTable(width = 280, height = 400, editable = True)
    bt = Button(label='Forecast')
    # layout = column(row([product,city,act], width=400), p)
    
    
    # layout = row(column(row([product,city],width=400),p),column(column([bt],width=300),f1))
    # layout = column(column(row([product,city],width=400),p),column(column([act],width=250),f1))
    
    # curdoc().add_root(layout)
    # curdoc().add_root(bt)
    
    
    template="""
    <div style="background:<%= 
    (function colorfromint(){
        if(pred0 >0){
            return("grey")}
        else{return("grey")}
        }()) %>; 
    color: black"> 
    <%= value %></div>
    """
    
    tabfrmt =  HTMLTemplateFormatter(template=template)
    
    
    
    def forecast(attr, old, new):
             
    	
    
        f2 = figure(x_axis_type='datetime',title="Forecast",tooltips=tt_categ,plot_width=1200, plot_height=370)
        f2.xaxis.formatter.days = '%m/%d/%Y'
        f2.xaxis.axis_label = 'Date'
        if prd.value=="0":
            y="pred0.0"
        elif prd.value=="5":
            y="pred0.05"
        elif prd.value=="10":
            y="pred0.1"
        else:
            y="pred0.15"
                
            
        
        # layout.children[1] = column(row([act], width=400), f2)
        
        #quoted it 17 june while developing tbas
        # layout.children[0] = column(row([product,city,act], width=400), f2)
        
        if product.value!= "Select":
            if city.value!= "Select":
                if act.value!="None":
                    
                    ols_act(data_ols)
                    fd['year']=fd['date'].dt.strftime('%Y')
                    fd['month']=fd['date'].dt.month   
                    grp1=ps.sqldf("""select year,month,sum(y) as actual, sum([pred0.0]) as pred0, 
                                  sum([pred0.05]) as pred5,sum([pred0.1]) as pred10,sum([pred0.15]) as pred15,
                                  sum([pred0.2]) as pred20,sum([pred0.25]) as pred25,
                                  sum([pred0.3]) as pred30,sum([pred0.35]) as pred35,
                                  sum([pred0.4]) as pred40 from fd group by 1,2;""")
                    # calendar.month_name(grp1['month'][1])
                    grp1['pred0']=round(grp1['pred0'],0)
                    grp1['pred5']=round(grp1['pred5'],0)
                    grp1['pred10']=round(grp1['pred10'],0)
                    grp1['pred15']=round(grp1['pred15'],0)
                    grp1['pred20']=round(grp1['pred20'],0)
                    grp1['pred25']=round(grp1['pred25'],0)
                    grp1['pred30']=round(grp1['pred30'],0)
                    grp1['pred35']=round(grp1['pred35'],0)
                    grp1['pred40']=round(grp1['pred40'],0)
                    
                    grp2=grp1[-6:]
                    tabsource=ColumnDataSource(grp2)
                    columns = [
                        TableColumn(field="year", title="Year"),
                        TableColumn(field="month", title="Month"),
                        TableColumn(field="actual", title="Actual"),
                        # TableColumn(field="pred0", title="0%",formatter=NumberFormatter(format='0,0')),
                        TableColumn(field="pred0", title="0%"),
                        TableColumn(field="pred5", title="5%"),
                        TableColumn(field="pred10", title="10%"),
                        TableColumn(field="pred15", title="15%"),
                        TableColumn(field="pred20", title="20%"),
                        TableColumn(field="pred25", title="25%"),
                        TableColumn(field="pred30", title="30%"),
                        TableColumn(field="pred35", title="35%"),
                        TableColumn(field="pred40", title="40%")
                            ]
                    data_table = DataTable(source=tabsource, columns=columns, width=1200, height=250,
                                           index_position=None,editable=True,fit_columns=True,background='black')
                    #quoted below line for tab
                    layout.children[1] = column(column(f2,data_table))
                    newSource2 =  ColumnDataSource(fd)
                    plotf=f2.line(x='date_fld', y='y', source=newSource2, line_width=3,line_color='red')
                    plotf=f2.line(x='date_fld', y=y, source=newSource2, line_width=3,line_color='green')
                    f2.add_tools(HoverTool(renderers=[plotf], tooltips=tt_categ,
                                        formatters = {"@date_fld": "datetime","@pred0.0":"numeral"},mode='vline'))
               
            
    
    
    
    def ols_act (data_ols):
        
        global fd
        frcs_df2=pd.DataFrame()
        print('i was clicked')
        df2 = data_ols[data_ols['product_desc']==product.value]
        df3 = df2[df2['city']==city.value]
        data_ols=df3
        
        data_ols['wednesday'] = data_ols['date_fld'].apply(daytype__Wednesday)
        data_ols['weekend'] = data_ols['date_fld'].apply(daytype__Week_End)
        data_ols['tot_qty'] = pd.to_numeric(data_ols['tot_qty'])
        data_ols['discount'] = pd.to_numeric(data_ols['discount'])
        data_ols['net_sales'] = pd.to_numeric(data_ols['net_sales'])
        data_ols['disc_pct']=data_ols['discount']/(data_ols['net_sales']+data_ols['discount'])
        data_ols['net_price']=(data_ols['net_sales'])/(data_ols['tot_qty'])
        data_ols['y']=data_ols['tot_qty']
        data_ols['ds']=data_ols['date_fld']
        data_ols['year']=data_ols['date_fld'].dt.strftime('%Y')
        data_ols['month']=data_ols['date_fld'].dt.month
        data_ols['month'] = data_ols['month'].astype('category')
        data_ols['day']=data_ols['date_fld'].dt.strftime('%A')
        data_ols['day'] = data_ols['day'].astype('category')
        # mcal=pd.read_excel('mcal.xlsx')
        # promo=mcal[['promo']]
        # data_ols['holiday'] = np.where(data_ols['date_key'].isin(promo['promo']), 1, 0)
        data_ols=data_ols[(data_ols['date_key']>=20190101) & (data_ols['date_key']<20200215)]
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
        
        traindata['y_lag2w']=traindata['y'].shift(28)
        # traindata=traindata.fillna(method="ffill")
        # traindata=traindata.fillna(method="bfill")
        traindata['y_avg_4w']=traindata['y'].rolling(28).mean()
        traindata=traindata.dropna()
        traindata['ds']=traindata['date']
         
        traindata2=traindata[(traindata['date']>= pd.datetime(2019,1,1))]
           
        traindata2=traindata2[(traindata2['ds']>=mindt)]
        # traindata2['wednesday'] = traindata2['date_fld'].apply(daytype__Wednesday)
        # traindata2['weekend'] = traindata2['date_fld'].apply(daytype__Week_End)
         
        fit = ols('y ~ C(month)+C(day)+disc_pct+y_avg_4w+footfall+zone_qty', traindata2).fit()     
           
        for pct in (0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4):
            fdf=pd.DataFrame()
            futuredf['ds']=futuredf['date']
            future = pd.merge(futuredf,traindata[['ds','y','zone_qty','month','day','year']],on='ds', how='left')
                # future['y_lag1y']=future['y'].shift(364)
            future['y']=future['y'].fillna(0)
            future['y_lag2w']=future['y'].shift(14)
            future['y_avg_4w']=future['y'].rolling(28).mean()
            future['mc_avg_qty']=mcqty
            future=future.fillna(method="ffill")
            # future=future.drop(['y'],axis=1)
            future=future.dropna()
            future['holiday']=future['ds'].apply(splevent)
                # future=future[(future['date']>=datetime.datetime(2020,2,1))]
        #                        futuredf['net_price']=(mop-(mop*pct))
            future['mop']=mop-(mop*pct)
            future['disc_pct']=pct
            future['footfall']=ftfl
            X=future[['ds','disc_pct','footfall','y_lag2w','y_avg_4w','month','zone_qty','mop','wednesday','weekend','holiday','day']]
        #                        Xnew=np.column_stack((X['wednesday'],X['weekend'],X['disc_pct']))
        #                        Xnew = sm.add_constant(X)
            fdf['pred'+str(pct)]=fit.predict(X)
        #                    futuredf.set_index('date', inplace=True)
            frcs_df2=pd.concat((frcs_df2,fdf), axis=1)
            fd=pd.DataFrame(pd.concat((future,frcs_df2),axis=1))
            fd['date_fld']=fd['date']
        return fd
            # print(fd)
        # print ("forecast Data is ready") 
        # plot1=p2.line(x='date_fld', y='tot_qty', source=frcs_df2, line_width=3)
    
          	
    product.on_change('value', forecast)
    city.on_change('value', forecast)
    act.on_change('value', forecast)
    prd.on_change('value', forecast)
    
    
    
    
    # bt = Button(label='Click me')
    
    
    
    # bt.on_event(ButtonClick, callback)
    # bt.on_click(ols)
    # layout = column(column(row([city,p],width=400),f1))
    layout = column(row(product,row([city,act,prd], width=400)),f1)
    # layout= column(row([city,p2], width=400),row(prod_sel,f2))
    
    # curdoc().add_root(layout)
    curdoc().title = "data"
    curdoc().theme = 'dark_minimal'
    
    tab = Panel(child=layout, title = 'Forecast')
    return tab










