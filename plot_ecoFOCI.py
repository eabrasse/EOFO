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

data_fn = home+'data/EBO-AK_v0.97_NUT-CDF_compiled_on_06-18-24.nc'
grid_fn = home+'data/Bering10K_extended_grid.nc'
ds = nc.Dataset(data_fn)
dsg = nc.Dataset(grid_fn)
n = len(ds['NO3'])

data_dict = {}
for key in ds.variables.keys():
    if len(ds[key])==n:
        data_dict[key] = ds[key][:]

data_dict['datetime'] = [datetime(1900,1,1)+timedelta(days=tt) for tt in ds['TIME'][:]]

df = pd.DataFrame.from_dict(data_dict,orient='columns')
df.set_index(df.datetime)

# df['year'] = [dt.year for dt in df.datetime]
# df['month'] = [dt.month for dt in df.datetime]
df['yearmonth'] = ['{}/{}'.format(dt.month,dt.year) for dt in df.datetime]

# years = df.year.unique()
# months = df.month.unique()
yearmonths = df.yearmonth.unique()

# # all will use same cmap and vlims, but I'm going to add these to the indiviual dicts anyway
# # for future generalizeability
# vmin = np.nanpercentile(temp,3,method='nearest')
# vmax = np.nanpercentile(temp,97,method='nearest')
# for var in var_list:
#     var['vmin'] = vmin
#     var['vmax'] = vmax
#     var['norm'] = matplotlib.colors.Normalize(vmin=var['vmin'],vmax=var['vmax'])
#     var['cmap'] = cmo.cm.thermal
#     var['units'] = r'$^{\circ}$C'

# alias some grid variables for legibility
lonr = dsg['lon_rho'][:]
latr = dsg['lat_rho'][:]
h = dsg['h'][:]
maskr = dsg['mask_rho'][:]

year = 2004
month = 8
# yearmonth = '{}/{}'.format(month,year)

ymsamplecount = []
# for yearmonth in yearmonths:
gg = df[(df.yearmonth=='{}/{}'.format(month,year))&(df.PRESSURE<20)]
# ymsamplecount.append(len(gg))
# if yearmonth == '{}/{}'.format(month,year):
fw, fh = efun.gen_plot_props()
fig = plt.figure(figsize=(12,8))
# ax0 = fig.add_subplot(1,2,1)
ax0=fig.gca()

# add coastline
ax0.pcolormesh(lonr,latr,maskr,cmap=cmap_mask,shading='nearest',zorder=5)
ax0.contour(lonr,latr,h,levels=np.arange(0,1000,50),colors='k',linewidths=0.5,zorder=6,linestyles=':')

# add sample locations
p=ax0.scatter(gg.LONGITUDE,gg.LATITUDE,s=20,c=gg.NO3,cmap=cmo.cm.dense,edgecolors='k',linewidths=0.5)
cbaxes = inset_axes(ax0, width="4%", height="40%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_ylabel('NO3 $\mu$mole L-1')
ax0.text(0.9,0.1,'Sample locations\n{}/{}'.format(month,year),transform=ax0.transAxes,ha='right',va='bottom')
ax0.text(0.1,0.9, 'EcoFOCI samples\nNO3 in surface 20m',transform=ax0.transAxes,ha='left',va='top')
ax0.axis([180,210,50,65])
efun.dar(ax0)
# look at depth distribution just for this month
# ax1 = fig.add_subplot(1,2,2)
# ax1.hist(gg.PRESSURE,bins=np.arange(0,150,10))
# ax1.set_xlabel('Sample depth (dbar)')
#

# fig2 = plt.figure(figsize=(12,6))
# ax2 = fig2.gca()
# ax2.bar(yearmonths,ymsamplecount)
# ax2.tick_params(axis='x', labelrotation=90)
# ax2.text(0.95,0.9,'Number of samples in ecoFOCI data compendium\nby month', transform=ax2.transAxes,ha='right',va='top')
# fig2.tight_layout()

# fig.show(block=False)
fig.subplots_adjust(left=0.1,right=0.85,)
plt.show(block=False)
plt.pause(0.1)
# plt.close()
