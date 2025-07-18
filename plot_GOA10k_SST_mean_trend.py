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
import geopandas
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import split
from shapely.affinity import translate
import numpy.ma as ma
from scipy.stats import linregress
import matplotlib.dates as mdates

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

datafn = home+'data/nep_wb_ssp585_Monthly_GOA_Mean_SST.csv'
df = pd.read_csv(datafn)
df['Datetime'] =  pd.to_datetime(df.Datetime)

# # initialize plot
fw,fh = efun.gen_plot_props()
fig = plt.figure(figsize=(fw*2.5,fh))
# ax0 = fig.add_subplot(1,2,1)
ax0 = fig.gca()

ax0.plot(df['Datetime'],df['Monthly GOA mean SST'],linestyle='solid',color=tab10(0))
ts = df.Datetime.values.astype(np.int64)//10**9
slope, intercept, r_value, p_value, std_err=linregress(ts,df['Monthly GOA mean SST'].values)
yy = intercept+slope*ts
ax0.plot(df['Datetime'],yy,color='k',linestyle='dashed')
ax0.set_xlabel('Year')
ax0.set_ylabel('Monthly mean GOA SST')
ax0.xaxis.set_major_locator(mdates.YearLocator(5))
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp( ax0.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
ax0.text(0.1,0.9,'SST Average for Gulf of Alaska by Month (SSP 585)',transform=ax0.transAxes)

fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

