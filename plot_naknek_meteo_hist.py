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
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
import string
import numpy as np
from matplotlib.gridspec import GridSpec

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
df['month'] = [dt.month for dt in df.datetime]
df['day'] = [dt.day for dt in df.datetime]
df['hour'] = [dt.hour for dt in df.datetime]
df['minute'] = [dt.minute for dt in df.datetime]
df['datetime2000'] = pd.to_datetime(dict(year=2000,month=df.month,day=df.day,hour=df.hour,minute=df.minute))

# df.set_index('datetime')

# dfH = df.set_index('datetime').resample('1H')
years = df.year.unique()
nyears = len(years)
yeargroupsize = 10
nyeargroups = int(np.floor(nyears/yeargroupsize))
rainbow = plt.get_cmap('rainbow',nyeargroups)

summer_months = [6,7,8,9,10,11]
winter_months = [1,2,3,4,5,12]

# si = 'm'
# rainbow = plt.get_cmap('rainbow',nfiles)
# # data_fn_list = home+'46041c2003.txt'

# generate figure and axis handles
# fig,axs = plt.subplots(nrows=3, ncols=1,figsize=(8,6))
fig = plt.figure(figsize=(12,6))
# rows=3
# cols=int(np.ceil(nyeargroups/rows))
# gs = GridSpec(rows,cols)
ax = fig.gca()

count=0
for i in range(nyeargroups):
    # irow = int(np.floor(i/rows))
    # icol = int(i-irow*cols)
    # ax = fig.add_subplot(gs[irow,icol])
    yeargroup = years[i*yeargroupsize:(i+1)*yeargroupsize]
    
    gg = df[df['year'].isin(yeargroup)&df['month'].isin(winter_months)][["sknt"]]
    ax.hist(gg['sknt'],bins=np.arange(0,40,2.5),density=True,edgecolor=rainbow(i),histtype='step',label='{}-{}'.format(yeargroup[0],yeargroup[-1]))
    # resample = gg.set_index('datetime').resample(si)
    # interpolated = resample.interpolate(method='linear')
    # interpolated = resample.mean()
    # ggg=interpolated.groupby('datetime2000').mean()
    # wsknt = gg.sknt.resample('H').interpolate()
    # drs = gg.datetime2000.resample('H').interpolate()
    # ax.plot(ggg.index,ggg.sknt,color=rainbow(count),lw=0.7,label='{}-{}'.format(yeargroup[0],yeargroup[-1]))
    # ax.text(0.9,0.9,'{}-{}'.format(yeargroup[0],yeargroup[-1]),transform=ax.transAxes,ha='right')
    
    
    count+=1
# ax.plot(df.datetime,df.sknt)

# count=0
# for fn in f_list:
#     data_fn = home+fn
#
#     #first four years formatted differently from other years
#     if fn in f_list[:4]:
#         df = pd.read_csv(data_fn,sep='\s+',engine='python')
#         # convert date from "321" to March 2021
#         # df['date'] = pd.to_datetime(dict(year=df.YYYY, month=df.MM, day=df.DD, hour=df.hh, minute=df.mm))
#         df['date'] = pd.to_datetime(dict(year=2000, month=df.MM, day=df.DD, hour=df.hh, minute=df.mm))
#         # df['minuteofyear'] = 60*(24 * (df.date.dt.dayofyear - 1) + df.date.dt.hour)+df.date.dt.minute
#
#         df['year'] = df['YYYY']
#
#         mask = (df.DIR<998)&(df.SPD<98)
#
#         #convert from degrees CCW from north to radians
#         df['DIR_rad'] = (90-df['DIR'])*np.pi/180
#
#         #convert from kts to meters per second
#         df['SPD_met'] = df['SPD']/1.94348
#
#         #convert to n/s wind velocity
#         df['VEL_ns'] = df['SPD_met']*np.sin(df['DIR_rad'])
#
#     # from 2007 forward
#     else:
#         df = pd.read_csv(data_fn,sep='\s+',engine='python',skiprows=[1])
#         # df['date'] = pd.to_datetime(dict(year=df['#YY'], month=df.MM, day=df.DD, hour=df.hh, minute=df.mm))
#         df['date'] = pd.to_datetime(dict(year=2000, month=df.MM, day=df.DD, hour=df.hh, minute=df.mm))
#         # df['minuteofyear'] = 60*(24 * (df.date.dt.dayofyear - 1) + df.date.dt.hour)+df.date.dt.minute
#         df['year'] = df['#YY']
#
#         mask = (df.WDIR<998)&(df.WSPD<98)
#
#         #convert from degrees CCW from north to radians
#         df['WDIR_rad'] = (90-df['WDIR'])*np.pi/180
#
#         #convert to n/s wind velocity
#         df['VEL_ns'] = df['WSPD']*np.sin(df['WDIR_rad'])
#
#     ax.scatter(df['date'][mask],df['VEL_ns'][mask],color=rainbow(count),label='{}'.format(df['year'][df['year'].size-1]),s=2)
#
#     if count==0:
#         ax.set_ylabel('meters per second')
#         ax.text(0.1,0.9,'N/S Wind velocity',transform=ax.transAxes)
#
#     count+=1


# ax.grid()
# ax.set_xlabel('Date')
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
# ax.set_ylim(0,30)
# ax.set_ylabel('Wind speed (knots)')
ax.text(0.8,0.9, 'King Salmon Airport\nWINTER',transform=ax.transAxes,ha='right')
# # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
ax.legend()
ax.set_xlabel('wind speed (kts)')
ax.set_ylabel('normalized hist density')

# show plot
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
