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

# opendap_url = 'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/raw/r20241015/no3os.nep.full.hcast.monthly.raw.r20241015.199301-201912.nc'
opendap_url = 'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/raw/r20241015/MLD_003.nep.full.hcast.monthly.raw.r20241015.199301-201912.nc'

ds_mom6 = xr.open_dataset(opendap_url)
ds_mom6_subset = ds_mom6.sel(time='2004-08',yh=slice(45,70)).load()


fw, fh = efun.gen_plot_props()
fig = plt.figure(figsize=(7.5,5))
ax0 = fig.gca()

# p=ax0.pcolormesh(ds_mom6_subset.xh,ds_mom6_subset.yh,ds_mom6_subset.no3os[0,:,:],cmap=cmo.cm.dense,shading='nearest')
p=ax0.pcolormesh(ds_mom6_subset.xh,ds_mom6_subset.yh,ds_mom6_subset.MLD_003[0,:,:],cmap=cmo.cm.dense,shading='nearest')
cbaxes = inset_axes(ax0, width="4%", height="40%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
# cbaxes.set_ylabel('NO3')
cbaxes.set_ylabel('(m)')
efun.dar(ax0)
ax0.set_xlabel('nominal longitude (?)')
ax0.set_ylabel('nominal latitude')
# ax0.text(0.98,0.1,'MOM6 surface NO3\n08-2004',transform=ax0.transAxes,ha='right',va='bottom')
ax0.text(0.98,0.1,'MOM6 MLD\n08-2004',transform=ax0.transAxes,ha='right',va='bottom')

fig.subplots_adjust(left=0.1,right=0.8,bottom=0.05,top=0.98)
plt.show(block=False)
plt.pause(0.1)

