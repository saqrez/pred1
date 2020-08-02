# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:51:35 2020

@author: saquib
"""
import pandas as pd
from bokeh.plotting import figure, show,output_file
from bokeh.models import ColumnDataSource,CDSView,DataTable,TableColumn,NumberFormatter,Legend
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel,HTMLTemplateFormatter,DatePicker
from bokeh.models import Select, Button
from bokeh.palettes import Spectral5,Category10,Viridis256
from bokeh.plotting import curdoc
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

from scripts.forecast import forecast
from scripts.category import categ

# from flask import Flask, render_template, request
# app = Flask(__name__)



def daytype__Wednesday(date_fld):
    date = pd.to_datetime(date_fld)
    if date.weekday() == 2 :
        return 1
    else:
        return 0


def daytype__Week_End(date_fld):
    date = pd.to_datetime(date_fld)
    if date.weekday() in [6] :
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

# @app.route('/')
# def start():
        
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
    

# mcl=pd.read_excel("./bok2/data/events.xlsx")
data_ols = pd.read_pickle(join(dirname(__file__), 'data', 'data.pkl')).dropna()
mcal = pd.read_csv(join(dirname(__file__), 'data', 'mcal.csv'))
data_ols['date_fld']=pd.to_datetime(data_ols['date_fld'])
data_ols['tot_qty'] = pd.to_numeric(data_ols['tot_qty'])
data_ols['discount'] = pd.to_numeric(data_ols['discount'])
data_ols['net_sales'] = pd.to_numeric(data_ols['net_sales'])
data_ols['disc_pct']=data_ols['discount']/(data_ols['net_sales']+data_ols['discount'])
# data_ols['tot_qty'] = pd.to_numeric(data_ols['tot_qty'])
# data_ols=data_ols[data_ols['city']=='KOCHI']

tab1 = categ(data_ols)
tab2 = forecast(data_ols,mcal)
# layout= column(row([city,p2], width=400),row(prod_sel,f2))
tabs = Tabs(tabs = [tab1, tab2])

curdoc().add_root(tabs)
# curdoc().title = "data"
curdoc().theme = 'dark_minimal'
    # return render_template('index.html') 




# if __name__ == '__main__':
 	# app.run(port=5000, debug=True)





