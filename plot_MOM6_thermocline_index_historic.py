# Python code to plot surface, bottom, and water column mean temperature movie
# for bering sea from 2020-2024
# EAB 1/31/2025

import os
import sys
ep = os.path.abspath('/Users/elizabethbrasseale/Projects/eDNA/code/')
if ep not in sys.path:
    sys.path.append(ep)
import efun
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import cmocean as cmo
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
from scipy.stats import linregress
import geopandas
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
import numpy.ma as ma
import calendar
import pickle

#close all currently open figures
plt.close('all')

# load in data and grid
home = '/Users/elizabethbrasseale/projects/EOFO/'

data_fn = home+'data/MOM6-NEP_EBSmean_thermocline_JulAugSep_1993-2024.csv'
df = pd.read_csv(data_fn,parse_dates = ['time'])

Month = False

# #add M2 to ax1
m2_fn = home+'data/M2_composite_1m.daily.2024.nc'
ds = xr.load_dataset(m2_fn)
temp = ds['temperature'].values[:]
z = -ds['depth'].values[:]
dt_array = pd.to_datetime(ds['time'].values[:])
nz = z.shape[0]
mask = [dt.month in [7,8,9] for dt in dt_array]
mask_tile = np.tile(mask,(nz,1)).T
temp_ma = ma.masked_where(~mask_tile,temp)
dtdz = np.diff(temp_ma,axis=1) # implied /1m because the cells are all 1m
mld_ind = np.argmax(np.abs(dtdz),axis=1)
mld = np.array([0.5*(z[mi]+z[mi+1]) for mi in mld_ind])
mld_ma = ma.masked_where(~np.array(mask),mld)

if ~Month:
    df['year'] = df.time.dt.year
    yearsmod = list(set(df.year))
    thermocline_year_avg = -df.groupby('year')['thermocline'].mean()
    
    tc_std = np.std(thermocline_year_avg)
    tc_mean = np.mean(thermocline_year_avg)
    tc_plus = tc_mean+tc_std
    tc_minus = tc_mean-tc_std
    # thermocline_demeaned = thermocline_year_avg-tcmean
    
    yearsM2 = list(set([dt.year for dt in dt_array]))
    summer_mld_year_avg = np.zeros(len(yearsM2))
    for yr in range(len(yearsM2)):
        year = yearsM2[yr]
        year_mask = np.array([dt.year!=year for dt in dt_array])
        mld_year_ma= ma.masked_where(year_mask,mld_ma)
        summer_mld_year_avg[yr] = np.mean(mld_year_ma)

    ymmod = [yr in yearsM2 for yr in yearsmod] # year mask model
    ymM2 = [yr in yearsmod for yr in yearsM2] # year mask M2





if Month:
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.gca()
    rainbow12 = plt.get_cmap('rainbow',12)
    for mo in [7,8,9]:
        inds=np.argwhere(df['time'].dt.month.values==mo)
        ax1.plot(df['time'].values[inds[:,0]],df['thermocline'][inds[:,0]],marker='o',linestyle='solid',mfc=rainbow12(mo-1),mec='none',color=rainbow12(mo-1),markersize=8,lw=2)
        ax1.text(0.93,0.55-0.05*mo,calendar.month_abbr[mo],transform=ax1.transAxes,ha='right',color=rainbow12(mo-1),fontweight='bold',fontsize=12)
    # rainbow365 = plt.get_cmap('rainbow',365)
    yearday = np.array([dt.timetuple().tm_yday for dt in dt_array])
    # ax1.scatter(dt_array,-mld_ma,c=yearday,cmap='rainbow',vmin=0,vmax=365,s=5)
    ax1.text(0.1,0.9,'EBS mean thermocline depth',transform=ax1.transAxes)
else:
    fig = plt.figure(figsize=(8,6))
    # ax1 = fig.add_subplot(1,2,1)
    ax1 = fig.gca()
    ax1.plot(yearsmod,thermocline_year_avg,marker='None',linestyle='solid',lw=1.5,color='k')
    ax1.grid(False)
    ax1.axhline(y=tc_plus,color='gray',linestyle='solid',lw=0.5)
    ax1.axhline(y=tc_minus,color='gray',linestyle='solid',lw=0.5)
    ax1.axhline(y=tc_mean,color='gray',linestyle='dashed',lw=0.5)
    ax1.fill_between(yearsmod,tc_plus,thermocline_year_avg,color='red',alpha=0.4,where=thermocline_year_avg>tc_plus,interpolate=True)
    ax1.fill_between(yearsmod,thermocline_year_avg,tc_minus,color='blue',alpha=0.4,where=thermocline_year_avg<tc_minus,interpolate=True)
    # ax1.plot(yearsM2,summer_mld_year_avg,marker='None',linestyle='dashed',markersize=8,lw=1.,color='m')
    # ax1.text(0.95,0.8,'MOM6',transform=ax1.transAxes,color='k',ha='right')
    # ax1.text(0.95,0.75,'M2',transform=ax1.transAxes,color='m',ha='right')
    ax1.text(0.1,0.9,'Jul-Aug-Sep average EBS mean thermocline depth trend',transform=ax1.transAxes)
    
    # ax2 = fig.add_subplot(1,2,2)
    # ax2.plot(-summer_mld_year_avg[ymM2],thermocline_year_avg[ymmod],linestyle='none',marker='o',mfc='k',mec='none')
    # ax2.set_xlabel('M2')
    # ax2.set_ylabel('MOM6')
    # ax2.axline((0, 0), slope=1,color='gray',alpha=0.5,linestyle='dashed')
    # ax2.set_aspect(1)
    # ax2.set_xlim(12,30)
    # ax2.set_ylim(12,30)

ax1.set_ylabel('(m)')
ax1.set_xlabel('Date time')


fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

