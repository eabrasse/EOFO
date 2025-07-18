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
import pickle

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

gridfn = home+'data/roms_grd_nep.nc'
dg = xr.load_dataset(gridfn)

datafn = home+'data/h126wb_2015-2100_85dd.p'
D = pickle.load(open(datafn,'rb'))

sf_path = home+'data/GOA shapefile/iho/iho.shp'
goa_geom = geopandas.read_file(sf_path)
poly = goa_geom.geometry[0]
# moved_ebs_geom = shift_geom(180,ebs_geom)
# polygons = list(goa_geom.geometry[0].geoms)
# newpolygon = []
# for poly in polygons:
lons, lats = poly.exterior.coords.xy
newlons = [lon%360 for lon in lons]
newpoly = Polygon(zip(newlons,lats))
# newpolygons.append(newpoly)
# MPoly = MultiPolygon(newpolygons)

t=0
lon = dg['lon_rho'][:]
lat = dg['lat_rho'][:]
h = dg['h'][:]
a = np.array([Point(x, y) for x, y in zip(lon.values.flatten(), lat.values.flatten())], dtype=object)
goa_mask = np.array([newpoly.contains(point) for point in a])
goa_mask = goa_mask.reshape(lon.shape)
# h_mask = h<200
goa_mask_bc = np.broadcast_to(goa_mask,D['dd85'].shape)

dd85_ma = ma.masked_where((~goa_mask_bc)&np.isnan(D['dd85']),D['dd85'][:])

dd85 = D['dd85'][:]
years = D['years'][:]

yrbar = np.mean(years)
yrMinusyrbar = years-yrbar
deno = np.sum(yrMinusyrbar**2)
xbar = np.nanmean(dd85_ma,axis=0)
xMinusxbar = dd85_ma-xbar
numer = np.zeros(dd85_ma.shape)
for i in range(len(years)):
    numer[i,:,:] = yrMinusyrbar[i]*xMinusxbar[i,:,:]
numer = np.nansum(numer,axis=0)
dd85_slope = numer/deno
dd85_inter = xbar-dd85_slope*yrbar

slope_ma = ma.masked_where(~goa_mask,dd85_slope)

notnancount = np.sum(~np.isnan(dd85[:,:,:]),axis=0)

# # initialize plot
fw,fh = efun.gen_plot_props()
fig = plt.figure(figsize=(fw*2.5,fh))
# ax0 = fig.add_subplot(1,2,1)
ax0 = fig.gca()

p=ax0.pcolormesh(lon,lat,slope_ma,shading='nearest',cmap=cmo.cm.balance_r,vmin=-0.5,vmax=0.5)
ax0.pcolormesh(lon,lat,dg.mask_rho,shading='nearest',cmap=cmap_mask,zorder=100)

cbaxes = inset_axes(ax0, width="4%", height="40%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_ylabel(r'$\Delta$ deg day/yr')
ax0.set_xlabel('Longitude')
ax0.set_xlim(195,225)
ax0.set_ylim(54,62)
ax0.set_ylabel('Latitude')
efun.dar(ax0)
ax0.text(0.9,.1,'SSP 126 (GFDL)\n'+r'$\Delta$ 85 degree day per year'+'\nfrom 2015 to 2100',transform=ax0.transAxes,fontsize=12,zorder=600,va='bottom',ha='right')

fig.subplots_adjust(left=0.1,right=0.85,bottom=0.15,top=0.95)

plt.show(block=False)
plt.pause(0.1)

