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
import numpy.ma as ma
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
import geopandas
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import split
from shapely.affinity import translate

#close all currently open figures
plt.close('all')

# import useful labeling string
atoz = string.ascii_lowercase

# custom 2-color colormap
landcol = matplotlib.colors.to_rgba('lightgray')
seacol = (1.0,1.0,1.0,0)
cmap_mask = matplotlib.colors.ListedColormap([landcol,seacol])

tab10 = plt.get_cmap('tab10',10)
Paired = plt.get_cmap('Paired',12)

# load in data and grid
home = '/Users/elizabethbrasseale/projects/EOFO/'

m2_fn = home+'data/M2_composite_1m.daily.2024.nc'
ds = nc.Dataset(m2_fn)
temp = ds['temperature'][:]
z = -ds['depth'][:]
ot = ds['time'][:]

dt_list = []
for ott in ot:
    dtt = datetime(1900,1,1)+timedelta(days=int(ott))
    dt_list.append(dtt)
dt_array = np.array(dt_list)


nz = z.shape[0]
mask = [dt.month in [7,8,9] for dt in dt_list]
mask_tile = np.tile(mask,(nz,1)).T
temp_ma = ma.masked_where(~mask_tile,temp)

tt, zz = np.meshgrid(dt_array,z)



# # initialize plot
fw,fh = efun.gen_plot_props()
fig = plt.figure(figsize=(fw*3,fh))
# ax0 = fig.add_subplot(1,2,1)
ax0 = fig.gca()

p=ax0.pcolormesh(tt,zz,temp_ma.T,shading='nearest',cmap='magma',vmin=0,vmax=10)
cbaxes = inset_axes(ax0, width="4%", height="40%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_ylabel('°C')
ax0.set_xlabel('Date')
ax0.set_ylabel('Depth')

dtdz = np.diff(temp_ma,axis=1) # implied /1m because the cells are all 1m
mld_ind = np.argmax(np.abs(dtdz),axis=1)
mld = np.array([0.5*(z[mi]+z[mi+1]) for mi in mld_ind])
mld_ma = ma.masked_where(~np.array(mask),mld)
ax0.plot(dt_array,mld_ma,linestyle='dashed',lw=1,marker='none',color='cyan')

fig.subplots_adjust(left=0.1,right=0.85,bottom=0.15)
plt.show(block=False)

#now calculate mean summer MLD by year
fig2 = plt.figure(figsize=(fw*2,fh))
ax1 = fig2.gca()

years = list(set([dt.year for dt in dt_list]))
summer_mld_year_avg = np.zeros(len(years))

for yr in range(len(years)):
    year = years[yr]
    year_mask = np.array([dt.year!=year for dt in dt_list])
    mld_year_ma= ma.masked_where(year_mask,mld_ma)
    summer_mld_year_avg[yr] = np.mean(mld_year_ma)
    
ax1.plot(years,summer_mld_year_avg,marker='o',linestyle='solid',color='k',lw=1)
mean_mean_MLD = np.mean(summer_mld_year_avg)
ax1.axhline(y=mean_mean_MLD,linestyle='dashed',color='cyan',lw=1)
ax1.set_xlabel('Year')
ax1.set_ylabel('Avg summer MLD (m)')
ax1.text(0.2,0.9,'M2 mooring',transform=ax1.transAxes)
fig2.subplots_adjust(left=0.1,right=0.9,bottom=0.15)
plt.show(block=False)
plt.pause(0.1)

# ax0.text(0.1,1.03,'Mixed Layer Depth (MLD_restrat)',transform=ax0.transAxes,ha='left',fontsize=12,zorder=600,va='bottom')
# ax0.text(0.05,0.13,pd.to_datetime(Sep_regrid.time[t].values).strftime("%b %Y"), transform=ax0.transAxes, ha='left', zorder=205, fontsize=8, color='k', fontweight='bold', bbox=dict(facecolor='white',edgecolor='None'))
# ax0.text(0.05,0.13,f"Jul–Sep {int(Sep_regrid_masked.year.values):}", transform=ax0.transAxes, ha='left', zorder=205, fontsize=8, color='k', fontweight='bold', bbox=dict(facecolor='white',edgecolor='None'))
# ax0.text(0.05,0.05,'Bathy contours at 50, 100, 200, 1000, 3500 m', transform=ax0.transAxes, ha='left', zorder=205, fontsize=8, color='k', bbox=dict(facecolor='white',edgecolor='None'))



