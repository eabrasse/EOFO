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

# tested using actual density vs approximate density to see if
# the density approximation impacted the bias
# But it did not. The approximation of 1027 kg/m3 only
# varies by O(1) kg/m3, which is +- O(0.1)%

momvar = 'thetao'
units = '°C'
vmin=0
vmax=12
fname = home+'data/thetao.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'
ds = xr.load_dataset(fname)

# ds_mom6_subset = ds_mom6.sel(lat=slice(54,58),lon=190.5,time='2004-07',method='nearest').load()
ds = ds.sel(time='2010-08',lat=slice(50,70),lon=slice(185,200))
ds_ll = ds.sel(z_l=0,method='nearest')
ds_lz = ds.sel(lon=190,method='nearest')
ds_lz = ds_lz.sel(lat=slice(55,60),z_l=slice(0,250))


fig = plt.figure(figsize=(10,6))

ax0 = fig.add_subplot(1,2,1)
lon,lat = np.meshgrid(ds_ll.lon,ds_ll.lat)
p=ax0.pcolormesh(lon,lat,ds_ll.thetao[0,:,:],shading='nearest',cmap='magma',vmin=vmin,vmax=vmax)
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
cbaxes = inset_axes(ax0, width="40%", height="4%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='horizontal')
cbaxes.set_xlabel(units)
ax0.plot([190,190],[ds_lz.lat.min(),ds_lz.lat.max()],marker='none',linestyle='dashed',color='cyan')
efun.dar(ax0)
ax0.text(0.9,0.1,'SST\nAug 2010',transform=ax0.transAxes,ha='right',va='bottom')

ax1 = fig.add_subplot(1,2,2)
latz, z_l = np.meshgrid(ds_lz.lat,ds_lz.z_l)
ax1.pcolormesh(latz,-z_l,ds_lz.thetao[0,:,:],shading='nearest',cmap='magma',vmin=vmin,vmax=vmax)
ax1.set_xlabel('Latitude')
ax1.set_ylabel('z (m)')
ax1.set_ylim([-250,0])
ax1.text(0.9,0.1,'Temperature transect',transform=ax1.transAxes,ha='right',va='bottom')

fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)