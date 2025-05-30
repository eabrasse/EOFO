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
import netCDF4 as nc
import pandas as pd
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import pytz
import cmocean as cmo
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import string
import matplotlib.path as mpath
import cartopy.mpl.patch as cpatch
import matplotlib.patches as patches
import cartopy.mpl.ticker as ctk
import xarray as xr
from scipy.stats import linregress
import calendar

def willmott(m,o):
    """
    Calculates willmott skill score between two vectors, a model and a set of observations
    """
    # if len(m)!=len(o):
    #     error('Vectors must be the same length!');
    # end

    MSE = np.nanmean((m - o)**2)
    denom1 = abs(m - np.nanmean(o))
    denom2 = abs(o - np.nanmean(o))
    denom = np.nanmean((denom1 + denom2)**2)
    
    if denom==0:
        WS = 0
    else:
        WS = 1 - MSE/denom
    
    return WS

#close all currently open figures
plt.close('all')

# import useful labeling string
atoz = string.ascii_lowercase

# custom 2-color colormap
landcol = matplotlib.colors.to_rgba('lightgray')
seacol = (1.0,1.0,1.0,0)
cmap_mask = matplotlib.colors.ListedColormap([landcol,seacol])

# load in data and grid
home = '/Users/elizabethbrasseale/projects/EOFO/'

mo=4
year=2009

fname_mom6 = home+'data/no3.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'
ds = xr.load_dataset(fname_mom6)
ds = ds.sel(time=f'{year:}-{mo:02}',lon=slice(180,195))
ds = ds.sel(lat=60,method='nearest')
vmin=0
vmax=30
scale = 1.027*10**6

fname=home+'data/ACOD_NUT_v1.0.csv'
df = pd.read_csv(fname,skiprows=[1])
df['DATETIME'] = pd.to_datetime(df.TIME)

gg = df[(df.DATETIME.dt.year==2009)&(df.DATETIME.dt.month==mo)]
gg = gg[(gg.LATITUDE<60.)&(gg.LATITUDE>59.8)]
gg['NO3'] = gg['NO3'].astype(float)

fig = plt.figure(figsize=(11,6))

# ax0 = fig.add_subplot(1,2,1)
ax0 = fig.gca()
lon,zz = np.meshgrid(ds.lon,-ds.z_l)
p=ax0.pcolormesh(lon,zz,scale*ds.no3[0,:,:],shading='nearest',cmap=cmo.cm.dense,vmin=vmin,vmax=vmax)
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Depth (z)')
cbaxes = inset_axes(ax0, width="4%", height="40%", loc='center right',bbox_transform=ax0.transAxes,bbox_to_anchor=(-0.15,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_xlabel(r'NO3 $\mu$mole $\mathrm{L}^{-1}$')
ax0.set_ylim([-250,20])
ax0.set_xlim([180,193])

# ax1 = fig.add_subplot(1,2,2,projection='3d')
# ax1.scatter(gg.LATITUDE,gg.LONGITUDE,-gg.PRESSURE,c=gg.NO3,cmap=cmo.cm.dense,vmin=vmin,vmax=vmax)
# ax1.set_xlabel('Longitude')
# ax1.set_ylabel('Latitude')
# ax1.set_zlabel('Depth (dbar)')
ax0.scatter(gg.LONGITUDE,-gg.PRESSURE,c=gg.NO3,cmap=cmo.cm.dense,vmin=vmin,vmax=vmax,edgecolors='white')

ci=5
lvl = np.arange(vmin,vmax+ci,ci)
ax0.contour(lon,zz,scale*ds.no3[0,:,:],levels=lvl,colors='cyan')
ax0.tricontour(gg.LONGITUDE,-gg.PRESSURE,gg.NO3,levels=lvl,colors='magenta')

ax0.text(0.6,0.4,'ACOD NO3 (circles, magenta contours)\nMOM6 NEP NO3 (color fill, cyan contours)\n{} {}\nLatitude 60 N'.format(calendar.month_abbr[mo],year),transform=ax0.transAxes,ha='center',va='center')


fig.subplots_adjust(left=0.1,right=0.9)
fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

fig2 = plt.figure(figsize=(11,6))
# same but with contours
ax2 = fig2.add_subplot(1,2,1)
p=ax2.contourf(lon,zz,scale*ds.no3[0,:,:],levels=lvl,cmap=cmo.cm.dense,vmin=vmin,vmax=vmax)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Depth (z)')
cbaxes = inset_axes(ax2, width="4%", height="40%", loc='center right',bbox_transform=ax2.transAxes,bbox_to_anchor=(-0.15,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_xlabel(r'NO3 $\mu$mole $\mathrm{L}^{-1}$')
ax2.set_ylim([-250,20])
ax2.set_xlim([180,193])

ax3 = fig2.add_subplot(1,2,2)
p=ax3.tricontourf(gg.LONGITUDE,-gg.PRESSURE,gg.NO3,levels=lvl,cmap=cmo.cm.dense,vmin=vmin,vmax=vmax)
ax3.scatter(gg.LONGITUDE,-gg.PRESSURE,c=gg.NO3,cmap=cmo.cm.dense,vmin=vmin,vmax=vmax,edgecolors='white')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Depth (z)')
ax3.set_ylim([-250,20])
ax3.set_xlim([180,193])

fig2.subplots_adjust(left=0.1,right=0.9)
fig2.tight_layout()
plt.show(block=False)
plt.pause(0.1)