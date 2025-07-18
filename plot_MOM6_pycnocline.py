# Python code to plot surface, bottom, and water column mean density movie
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
import geopandas
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import split
from shapely.affinity import translate
import numpy.ma as ma

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


fname = home+'data/rhopot0.nep.full.hcast.monthly.regrid.r20250509.199301-202412.nc'
ds = xr.load_dataset(fname)

# ds_mom6_subset = ds_mom6.sel(lat=slice(54,58),lon=190.5,time='2004-07',method='nearest').load()
ds = ds.sel(time='2010-08',lat=slice(50,70),lon=slice(175,205),z_l=slice(0,60))
nt,nz,ny,nx = ds['rhopot0'].shape

#masks 
sf_path = home+'data/ecoregions/ecoregions.shp'
ebs_geom = geopandas.read_file(sf_path)
# moved_ebs_geom = shift_geom(180,ebs_geom)
polygons = list(ebs_geom.geometry[0].geoms)
newpolygons = []
for poly in polygons:
    lons, lats = poly.exterior.coords.xy
    newlons = [lon%360 for lon in lons]
    newpoly = Polygon(zip(newlons,lats))
    newpolygons.append(newpoly)
MPoly = MultiPolygon(newpolygons)

lon,lat = np.meshgrid(ds.lon,ds.lat)
a = np.array([Point(x, y) for x, y in zip(lon.flatten(), lat.flatten())], dtype=object)
ebs_mask = np.array([MPoly.contains(point) for point in a])
ebs_mask = ebs_mask.reshape(ny,nx)
# is pycnocline present? y/n
pycno_pres = np.nanmax(np.abs(ds.rhopot0[0,:,:,:]-ds.rhopot0[0,0,:,:]),axis=0)>0.5

z_l = ds['z_l'].values[:]
dz = np.diff(z_l)
drho = np.diff(ds.rhopot0[0,:,:,:],axis=0)
drhodz = drho/np.tile(np.reshape(dz,(dz.shape[0],1,1)),(ny,nx))
pycno_ind = np.argmax(np.abs(drhodz),axis=0) #note - 'all nan slices encountered'
pycno_ind = pycno_ind.reshape(ny*nx)
rhopot0 = ds['rhopot0'].values.reshape(nz,ny*nx)

#pycnocline depth
pcd = np.array([0.5*(z_l[pycno_ind[i]]+z_l[pycno_ind[i]+1]) for i in range(ny*nx)])
pcd_nan = [np.isnan(rhopot0[pycno_ind[i]+1,i]) for i in range(ny*nx)]
pcd_nan = np.array(pcd_nan).reshape(ny,nx)
pcd  = pcd.reshape(ny,nx)
pcd_ma = ma.masked_where((pcd_nan)|(~ebs_mask)|(~pycno_pres),pcd)

fig = plt.figure(figsize=(12,6))

ax0 = fig.add_subplot(1,2,1)
p=ax0.pcolormesh(lon,lat,pcd_ma,shading='nearest',cmap=cmo.cm.deep,vmin=0,vmax=25)
ax0.pcolormesh(lon,lat,~np.isnan(ds['rhopot0'][0,0,:,:].values),shading='nearest',cmap=cmap_mask,zorder=1000)
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
cbaxes = inset_axes(ax0, width="40%", height="4%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='horizontal')
cbaxes.set_xlabel('m')
efun.dar(ax0)
ax0.text(0.9,0.05,'Pycnocline depth\nAug 2010',transform=ax0.transAxes,ha='right',va='bottom')
ax0.plot([177,200],[57.5,57.5],linestyle='solid',color='cyan')

yind = np.argmin(np.abs(ds.lat.values-57.5))

ax1 = fig.add_subplot(1,2,2)
lonz, z_ll = np.meshgrid(ds.lon,ds.z_l)
ax1.pcolormesh(lonz,-z_ll,ds.rhopot0[0,:,yind,:],shading='nearest',cmap=cmo.cm.matter,vmin=1023,vmax=1027)
ax1.plot(ds.lon,-pcd_ma[yind,:],linestyle='solid',marker='none',color='cyan')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('z (m)')
ax1.set_ylim([-50,0])
ax1.text(0.9,0.1,'Density transect',transform=ax1.transAxes,ha='right',va='bottom')
ax1.set_xlim(177,200)

fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)