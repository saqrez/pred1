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
    
 
# ###  Train Data Preparation for entire period ##########
# train=pd.DataFrame()
# start_date = datetime.date(2019, 1, 1)
# end_date   = datetime.date(2020, 2, 15)
# train['date'] = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days)+1)]
# train['date']=pd.to_datetime(train['date'])
# #train['month']=train['date'].dt.month
# #train['month'] = train['month'].astype('category')

# ##################  Creating Furure Dataframe  ############################
# futuredf=pd.DataFrame()
# start_date = datetime.date(2019, 1, 1)
# end_date   = datetime.date(2020, 3, 31)
# futuredf['date'] = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days)+1)]
# futuredf['date']=pd.to_datetime(futuredf['date'])
# futuredf['wednesday'] = futuredf['date'].apply(daytype__Wednesday)
# futuredf['weekend'] = futuredf['date'].apply(daytype__Week_End) 
# # futuredf['month']=futuredf['date'].dt.month
# # futuredf['month'] = futuredf['month'].astype('category')
    

# #######################################################################################
# # data_ols=pd.read_pickle("./data/data.pkl")
# data_ols = pd.read_pickle(join(dirname(__file__), 'data', 'data.pkl')).dropna()
# data_ols['date_fld']=pd.to_datetime(data_ols['date_fld'])
# data_ols['tot_qty'] = pd.to_numeric(data_ols['tot_qty'])
# data_ols['discount'] = pd.to_numeric(data_ols['discount'])
# data_ols['net_sales'] = pd.to_numeric(data_ols['net_sales'])
# data_ols['disc_pct']=data_ols['discount']/(data_ols['net_sales']+data_ols['discount'])
# # data_ols['tot_qty'] = pd.to_numeric(data_ols['tot_qty'])
# # data_ols=data_ols[data_ols['city']=='KOCHI']
def categ(data):
    
    
    
    
    ############################  DATA PULL FROM POSTGRES   #############################################
    
    data_ols=data
    prod = []
    # prod.append('All')
    
    prod.extend(data_ols['product_desc'].unique().tolist())
    product = Select(title="Product",  options=sorted(prod), value="Product")
    
    city = []
    city.append('City')
    
    city.extend(data_ols['city'].unique().tolist())
    city = Select(title="City",  options=sorted(city), value="City",background = "grey")
    
    action=['None','Forecast']
    act=Select(title="Action",options=action, value="Action")
    
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
    						  
    
    f1=figure(x_axis_type='datetime', title="Products Quantity",plot_width=1000, plot_height=250)
    p = figure(x_axis_type='datetime',title="Sales Trend",plot_width=1000, plot_height=250)	
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
    
    
    def city_categ(attr, old, new):
    	
        p2 = figure(x_axis_type='datetime',title="Category Sales Trend for " + city.value ,tooltips=tt_categ,plot_width=1000, plot_height=250)
        # p2.extra_y_ranges['disc'] = Range1d(start=0, end=1)
        # p2.add_layout(LinearAxis(y_range_name='discount', axis_label='Discount'), 'right')
        p2.xaxis.formatter.days = '%m/%d/%Y'
        p2.xaxis.axis_label = 'Date'
        p2.yaxis.axis_label = 'Sales Value'
        # p2.add_layout(LinearAxis(y_range_name="Footfall"), 'right')
        # layout.children[0] = column(city, width=400), p2,f2)
        
        f2 = figure(x_axis_type='datetime',title="Unit Sold",tooltips=tt_prod,plot_width=1000, plot_height=250)
        f2.xaxis.formatter.days = '%m/%d/%Y'
        f2.xaxis.axis_label = 'Date'
        f2.yaxis.axis_label = 'Sales Volume'
        
        # layout.children[0] = column(row([city,p2], width=400), f2)
        # layout.children[0] = column(row([city], width=400), f2)
        # layout.children[1] = column(city, width=400), f2)
        # layout.children[1] = column(row([act], width=400), f2)
        # layout.children[0] = column(row([product,city,act], width=400), f2)
        # layout2 = row(column(row([product,city],width=400),p2),column(column([bt],width=300),f1))
        
        if city.value=="City":
            df1 = data_ols.copy()
            df1=df1.sort_values(by=['date_fld','product_desc'])
            gby = df1.groupby(['date_fld'])['net_sales','discount'].sum().reset_index()
           
            # sales_city=ps.sqldf("""select date_fld, sum(net_sales) as sales, sum(discount) as disc from data_n group by 1;""")
            # qty=df1['tot_qty']
            newSource =  ColumnDataSource(gby)
    #         prod=list(set(gby['product_desc'].unique()))
    #         prod.sort()
    #         prod_sel = CheckboxGroup(labels=prod, 
    # 									   active = [0, 1])
    #         layout.children[0] = column(row([city], width=400),p2)
            # layout.children[0] = column(row([city], width=400),p2,row(f2,prod_sel))
        # if product.value!= "All":
        #     df2 = data_ols[data_ols['product_desc']==product.value]
        #     df2=df2.sort_values(by=['date_fld','product_desc'])
        #     # qty=df2['tot_qty']
        #     newSource =  ColumnDataSource(df2)
        #     act.value="None"
        		
        if city.value!= "City":
            global prolist
            global citydata
            global int_prod
            global prod_sel
            global src
            df3 = data_ols[data_ols['city']==city.value]
            df3=df3.sort_values(by=['date_fld','product_desc'])
            gbx = df3.groupby(['date_fld','product_desc'])
            citydata = df3.groupby(['date_fld','product_desc'])['net_sales','tot_qty','discount'].sum().reset_index()
            newSource =  ColumnDataSource(citydata)
            prod=list(set(citydata['product_desc'].unique()))
            prodlist=", ".join(map(str, prod))
            prod.sort()
            # prod_sel = CheckboxGroup(labels=prod,active = [0, 1],height=300)
            prolist=MultiSelect(title="Products",options=prod,height=240,width=300,background = "grey")
            # prod_list = [prod_sel.labels[i] for i in prolist.value]
            # prod_list = [prod_list for i in prolist.value]
     					 # i in prod_sel.active]
            # wb=column([prod_sel],height=300)
            xs = []
            ys = []
            colors = []
            labels = []
            # for i, prd in enumerate(prod_list):
            for i, prd in enumerate(prolist.value):
                lbl=prd
            # for i, prd in enumerate(prod_list):
                subset = citydata[citydata['product_desc'] == prd]
                x = subset['date_fld']
                y= subset['tot_qty']
                xs.append(list(x))
                ys.append(list(y))
            
             			# Append the colors and label
                colors.append(Viridis256[i])
                labels.append(lbl)
                
            src = ColumnDataSource(data={'x': xs, 'y': ys,'color': colors, 'label': labels})
            # return print(y)
            # src.data.update(new_src.data)
            # src.data.update(newsrc.data)
            f2.multi_line('x', 'y', color = 'color',  line_width = 3, source = src)
            # return print (labels)
            # prod_sel.on_change('active', plotm)
            prolist.on_change('value',plotm)
    
        # plotf=f2.line(x='date_fld', y='pred0.0', source=newSource2, line_width=3,line_color='red')
        
        ####quoting below 2 lines 17 june
        layout.children[0] = row(p2,city)
        layout.children[1] = row(f2,prolist) 
            
        plot1=p2.line(x='date_fld', y='net_sales', source=newSource, line_width=3,line_color='white')
        # plot1=p2.line(x='date_fld', y='discount', source=newSource,line_width=3,line_color='yellow')
        # plot1=p2.vbar(x='date_fld', top=qty, width=0.3)
        # plot2=p2.circle(x='date_fld', y='disc_pct', source=newSource,fill_color="red",size=10,y_range_name='disc')
        p2.add_tools(HoverTool(renderers=[plot1], tooltips=tt_categ,
                               formatters = {"@date_fld": "datetime","@discount":"numeral","@net_sales":"numeral"},mode='vline'))
    
        
    
    def plotm(attr, old, new):
        # global newsrc
        # f3 = figure(x_axis_type='datetime',title="Unit Sold",tooltips=tt_categ,plot_width=900, plot_height=250)
        # f3.xaxis.formatter.days = '%m/%d/%Y'
        # f3.xaxis.axis_label = 'Date'
        # f3.yaxis.axis_label = 'Sales Volume'
        
        xs = []
        ys= []
        colors = []
        labels = []
        
        # prod_to_plot = [prod_sel.labels[i] for i in	prod_sel.active]
        # prod_to_plot = [prolist.value[i] for i in	prolist.value]
        # return print(prod_to_plot)
        # prod_list = [prolist.value[i] for i in prolist.value]
        # for prd in prolist.value:
            # return print (i,prd)
            # return print(prod_list)
        for i, prd in enumerate(prolist.value):
            lbl=prd
            subset = citydata[citydata['product_desc'] == prd]
            # subset['date_fld'].apply(lambda d: d.strftime('%m/%d/%y'))
            # return print(subset.dtypes)
            x = subset['date_fld']
            x=pd.to_datetime(x,unit='F')
            y= subset['tot_qty']
            xs.append(list(x))
            ys.append(list(y))
            colors.append(Spectral5[i])
            labels.append(lbl)
        newsrc = ColumnDataSource(data={'x': xs, 'y': ys,'color': colors, 'label': labels})
    
    
        
        src.data.update(newsrc.data)
        f2 = figure(x_axis_type='datetime',title="Unit Sold",tooltips=tt_prod,plot_width=1000, plot_height=250)
        f2.xaxis.ticker = DaysTicker(days = [1, 5, 10, 15, 20, 25, 30])
        f2.xaxis.axis_label = 'Date'
        f2.yaxis.axis_label = 'Sales Volume'
        plot2=f2.multi_line('x', 'y', color = 'color',
    					 line_width = 3,
    					 source = src)
        
        
        
        f2.add_tools(HoverTool(renderers=[plot2], tooltips=tt_prod,
                               formatters = {"@x": "datetime","@y":"numeral"},mode='vline'))
    
        # f2.add_layout(f2.legend[0], 'below')
      
        
    
          	
    # product.on_change('value', update_plot)
    city.on_change('value', city_categ)
    # act.on_change('value', forecast)
    
    
    
    
    # bt = Button(label='Click me')
    
    
    
    # bt.on_event(ButtonClick, callback)
    # bt.on_click(ols)
    # layout = column(column(row([city,p],width=400),f1))
    layout = column(row(p,row([city], width=250)),f1)
    # layout= column(row([city,p2], width=400),row(prod_sel,f2))
    
    # curdoc().add_root(layout)
    curdoc().title = "data"
    curdoc().theme = 'dark_minimal'
    
    tab = Panel(child=layout, title = 'Category')
    return tab










