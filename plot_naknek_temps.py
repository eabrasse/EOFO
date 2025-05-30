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
data_fn = home+'data/PAKN.csv'

df = pd.read_csv(data_fn,sep=',',engine='python')
df['datetime'] = pd.to_datetime(df.valid, format='%Y-%m-%d %H:%M')
df['year'] = [dt.year for dt in df.datetime]
# df = df.drop(df[df.year<1970].index)
df = df.drop(df[df.year==2025].index)

# generate figure and axis handles
fw,fh = efun.gen_plot_props()
fig = plt.figure(figsize=(12,6))

ax = fig.gca()
gg = df.groupby('year')['tmpf'].mean()
ax.plot(gg.index,gg.values,marker='o',color='k',linestyle='dashed',lw=1.0,markersize=10)
slope, intercept, r_value, p_value, std_err = linregress(gg.index,gg.values)
yy= intercept+slope*gg.index
ax.plot(gg.index,yy,color='purple',linestyle='solid',marker='none',lw=3)


ax.grid()
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (F)')
ax.text(0.1,0.9, 'King Salmon Airport\nAnnual mean air temperature',transform=ax.transAxes)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')


# show plot
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
