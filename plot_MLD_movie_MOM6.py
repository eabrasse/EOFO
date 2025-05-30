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

opendap_url= 'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/regrid/r20241015/MLD_003.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'


ds_mom6 = xr.open_dataset(opendap_url)
ds_mom6_subset = ds_mom6.sel(lat=slice(50,70)).load()


for t in range(len(ds_mom6.time)):
    # t=100 #while testing

    # initialize plot
    fw,fh = efun.gen_plot_props()
    fig = plt.figure(figsize=(10,7))
    # gs = GridSpec(1,len(var_list))
    ax = fig.gca()


    # now add heatmap
    p=ax.pcolormesh(ds_mom6_subset.lon,ds_mom6_subset.lat,ds_mom6_subset.MLD_003[t,:,:],shading='nearest',cmap=cmo.cm.deep,vmin=0,vmax=100)

    cbaxes = inset_axes(ax, width="4%", height="40%", loc='upper right',bbox_transform=ax.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
    cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
    cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
    cbaxes.set_ylabel('m')

    ax.set_xlabel('Longitude')
    ax.set_xlim(180,210)
    ax.set_ylabel('Latitude')
    efun.dar(ax)
    ax.text(0.1,1.05,'Mixed Layer Depth',transform=ax.transAxes,ha='left',fontsize=12,zorder=600,va='bottom')


    ax.text(0.05,0.1,pd.to_datetime(ds_mom6_subset.time[t].values).strftime("%b %Y"), transform=ax.transAxes, ha='left', zorder=55, fontsize=8, color='k', fontweight='bold', bbox=dict(facecolor='white',edgecolor='None'))


    # now tidy up and show the plot
    fig.subplots_adjust(right=0.9,left=0.08,bottom=0.15,top = 0.9)
    # fig.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)
    outfn = home+f'plots/MOM6 MLD/figure_{t:0>4}.png'
    plt.savefig(outfn)
    plt.close()
#
# # close netCDF datasets
# ds_mom6_subset.close()
