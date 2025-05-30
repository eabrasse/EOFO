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
df = df.drop(df[df.year<1970].index)
df['month'] = [dt.month for dt in df.datetime]
df['day'] = [dt.day for dt in df.datetime]
df['hour'] = [dt.hour for dt in df.datetime]
df['minute'] = [dt.minute for dt in df.datetime]
df['datetime2000'] = pd.to_datetime(dict(year=2000,month=df.month,day=df.day,hour=df.hour,minute=df.minute))
df['rad'] = (270-df['drct'][:])*np.pi/180

df['wind_EW'] = df.apply(lambda row: row['sknt']*np.cos(row['rad']),axis=1)

# df.set_index('datetime')

# dfH = df.set_index('datetime').resample('1H')
years = df.year.unique()
nyears = len(years)
yeargroupsize = 5
nyeargroups = int(np.floor(nyears/yeargroupsize))
rainbow = plt.get_cmap('rainbow',nyeargroups)


si = '7d'
# rainbow = plt.get_cmap('rainbow',nfiles)
# # data_fn_list = home+'46041c2003.txt'

# generate figure and axis handles
# fig,axs = plt.subplots(nrows=3, ncols=1,figsize=(8,6))
fig = plt.figure(figsize=(12,6))
# ax_mean = fig.add_subplot(1,3,1)
# ax_05 = fig.add_subplot(1,3,2)
# ax_95 = fig.add_subplot(1,3,3)
ax = fig.gca()

count=0
for i in range(nyeargroups):
    yeargroup = years[i*yeargroupsize:(i+1)*yeargroupsize]
    gg = df[df['year'].isin(yeargroup)][["datetime","datetime2000", "wind_EW"]]
    daily_75 = gg.groupby([gg['datetime2000'].dt.date]).quantile(0.75)
    # resample = gg.set_index('datetime').resample(si)
    # # interpolated = resample.interpolate(method='linear')
    # interpolated = resample.mean()
    # ggg=interpolated.groupby('datetime2000').mean()
    # wsknt = gg.sknt.resample('H').interpolate()
    # drs = gg.datetime2000.resample('H').interpolate()
    ax.plot(daily_75.index,daily_75.wind_EW,color=rainbow(count),lw=0.7,label='{}-{}'.format(yeargroup[0],yeargroup[-1]))
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


ax.grid()
ax.set_xlabel('Date')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
ax.set_ylim(-10,20)
ax.set_ylabel('E-W Wind speed (knots)')
ax.text(0.1,0.9, 'King Salmon Airport\n75th percentile for date in yeargroup',transform=ax.transAxes)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
ax.legend(ncols=3,loc='upper right')

# show plot
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
