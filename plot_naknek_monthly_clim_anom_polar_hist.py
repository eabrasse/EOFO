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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import calendar
import cmocean as cmo
import pickle

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
df['rad'] = df['drct'][:]*np.pi/180
# df.set_index('datetime')

# ignore years before 1970 because measurements looked different
df = df.drop(df[df.year<1970].index)

# dfH = df.set_index('datetime').resample('1H')
years = df.year.unique()
nyears = len(years)
yeargroupsize = 1
nyeargroups = int(np.floor(nyears/yeargroupsize))
rainbow = plt.get_cmap('rainbow',nyeargroups)

summer_months = [6,7,8,9,10,11]
winter_months = [1,2,3,4,5,12]

# si = 'm'
# rainbow = plt.get_cmap('rainbow',nfiles)
# # data_fn_list = home+'46041c2003.txt'



climfn = home+'data/kingsalmon_monthly_wind_climatology_2dpolarhist_after1970.p'
D = pickle.load(open(climfn,'rb'))


rbinedges= D['rbinedges']
thetabinedges = D['thetabinedges']
theta, R = np.meshgrid(thetabinedges, rbinedges)
ncolors = int(np.ceil(0.12/0.01))

count=0
for yr in range(nyeargroups):
    # generate figure and axis handles
    # fig,axs = plt.subplots(nrows=3, ncols=1,figsize=(8,6))
    fig = plt.figure(figsize=(10,8))
    rows=3
    cols=int(np.ceil(12/rows))
    gs = GridSpec(rows,cols)
    # ax = fig.gca()
    # yr=0
    for mo in range(12):
        irow = int(np.floor(mo/cols))
        icol = int(mo-irow*cols)
        ax = fig.add_subplot(gs[irow,icol],polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        # yeargroup = years[yr*yeargroupsize:(yr+1)*yeargroupsize]
        yeargroup = [years[yr]]
        month=mo+1
    
        # #make cmap
        # mycol = [p for p in rainbow(i)]
        # colors = [(mycol[0], mycol[1], mycol[2],c) for c in np.linspace(0.2,1,100)]
        # alpharainbow = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colors, N=ncolors)
    
        gg = df[(df['year'].isin(yeargroup))&(df['month']==month)][["sknt","rad"]]
        # gg = df[df['year'].isin(yeargroup)&df['month'].isin(winter_months)][["sknt"],["rad"]]
        # ax.scatter(gg['rad'],gg['sknt'],marker='o',s=2.0,color=rainbow(i),alpha=0.5)
        hist, _, _ = np.histogram2d(gg['rad'],gg['sknt'], bins=(thetabinedges, rbinedges),density=True)
    
        dkey = calendar.month_name[mo+1]+ '_hist'
        hist_diff = hist-D[dkey]
    
        hist_diff[hist_diff==0] = np.nan
        hist_diffm = np.ma.array(hist_diff,mask=np.isnan(hist_diff))
    

    
        p=ax.pcolormesh(theta, R, hist_diffm.T, cmap=cmo.cm.balance,vmin=-0.01,vmax = 0.01,zorder=5)
    
        if mo==0:
            cbaxes = inset_axes(ax, width="4%", height="60%", loc='lower right',bbox_transform=ax.transAxes,bbox_to_anchor=(0.1,0.,1,1))
            cb = fig.colorbar(p, cax=cbaxes, orientation='vertical')

        ax.text(1.3,1.1,calendar.month_name[mo+1],transform=ax.transAxes,ha='right',color='k',fontweight='bold')
        ax.set_rmax(50)
    
    
        count+=1



    # ax.text(0.8,0.9, 'King Salmon Airport\nWINTER',transform=ax.transAxes,ha='right')
    # plt.suptitle('King Salmon Airport\n{}-{}\nMonthly deviance from climatology (from 1970)'.format(yeargroup[0],yeargroup[-1]))
    plt.suptitle('King Salmon Airport\n{}\nMonthly deviance from climatology (from 1970)'.format(yeargroup[0]))
    # # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
    # ax.legend()
    # ax.set_xlabel('wind speed (kts)')
    # ax.set_ylabel('normalized hist density')

    # show plot
    plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)
    # outfn = home+'plots/Naknek wind/decadal wind anomalies/kingsalmon_{}-{}_after1970.png'.format(yeargroup[0],yeargroup[-1])
    outfn = home+'plots/Naknek wind/decadal wind anomalies/kingsalmon_after1970_{}.png'.format(yeargroup[0])
    plt.savefig(outfn)
    plt.close()
    
