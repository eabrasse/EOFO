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
import matplotlib.dates as mdates
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()
# import pyreadr

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


fname=home+'data/Fur seal georeferenced dives_examples.csv'
# readRDS = robjects.r['readRDS']
# furseal632 = readRDS(fname)
# furseal632 = pandas2ri.rpy2py_dataframe(furseal632)
# result = pyreadr.read_r(fname)
# furseal632 = result[None]
df = pd.read_csv(fname)
df['dt'] = pd.to_datetime(df['gmt.y'])
dbids = df.dbid.unique()
dbid = dbids[0]
df0 = df[df.dbid==dbid] #look at the first NFS. No need to hard code what its ID is.



fw, fh = efun.gen_plot_props()
fig = plt.figure(figsize=(7.5,6))
ax0 = fig.gca()

tripno=1
gg = df0[(df0.tripno==tripno)&(df0.Maxdepth>4)&(df0.DiveNumber<1543)]

p=ax0.scatter(gg.dt,-gg.CorrectedDepth,c=gg.etemp,cmap='magma')
cbaxes = inset_axes(ax0, width="4%", height="40%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_ylabel(r'$^{\circ}$C')
ax0.set_xlabel('Time')
ax0.set_ylabel('Depth')
ax0.text(0.98,0.1,f'Northern Fur Seal {dbid:}\nTrip no. {tripno:}',transform=ax0.transAxes,ha='right',va='bottom')
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
plt.setp( ax0.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')

fig.subplots_adjust(left=0.11,right=0.85,bottom=0.15,top=0.98)
# fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

