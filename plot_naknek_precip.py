#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot cutthroat trout DNA concentrations
"""

# importing modules, the beginning of all python code
import os
import sys
import matplotlib
matplotlib.use('macosx')
ep = os.path.abspath('/Users/elizabethbrasseale/Projects/Upwelling/code')
if ep not in sys.path:
    sys.path.append(ep)
import efun
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
import string
import numpy as np
from scipy.stats import linregress

# some helpful plotting commands
plt.close('all')
tab10 = plt.get_cmap('tab10',10)
rainbow = plt.get_cmap('rainbow')

# load in data
home = '/Users/elizabethbrasseale/Projects/EOFO/'
data_fn = home+'data/acrc_USW00025503_annual_precip_1744066974405.csv'

df = pd.read_csv(data_fn,sep=',',engine='python',skiprows=4,skipfooter=1)
df['Year'] = [int(year) for year in df.Year]
df = df.drop(df[df.Year<1947].index)

# generate figure and axis handles
fw,fh = efun.gen_plot_props()
fig = plt.figure(figsize=(12,6))

ax = fig.gca()
ax.plot(df['Year'],df['Annual Total Precipitation (in)'],marker='o',color='k',linestyle='dashed',lw=1.0,markersize=10)
slope, intercept, r_value, p_value, std_err = linregress(df['Year'].values,df['Annual Total Precipitation (in)'].values)
yy= intercept+slope*df['Year'].values
ax.plot(df['Year'],yy,color='purple',linestyle='solid',marker='none',lw=3)


ax.grid()
ax.set_xlabel('Year')
ax.set_ylabel('Precipitation (in)')
ax.text(0.1,0.9, 'King Salmon Airport\nAnnual Total Precipitation (in)\n1947-2022',transform=ax.transAxes,va='top')
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')


# show plot
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
