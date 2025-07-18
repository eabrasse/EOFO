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

momvar = 'thetao'
fname = home+'data/thetao.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'
ds = xr.load_dataset(fname)

# ds_mom6_subset = ds_mom6.sel(lat=slice(54,58),lon=190.5,time='2004-07',method='nearest').load()
ds = ds.sel(time='2010-08',lat=slice(50,70),lon=slice(175,205),z_l=slice(0,60))
nt,nz,ny,nx = ds['thetao'].shape
# ds_ll = ds.sel(z_l=0,method='nearest')
# ds_lz = ds.sel(lon=190,method='nearest')
# ds_lz = ds_lz.sel(lat=slice(55,60),z_l=slice(0,250))

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


tbar = np.mean(thetao[0,:,:,:],axis=0)




# is thermocline present? y/n
thermo_pres = np.nanmax(np.abs(ds.thetao[0,:,:,:]-ds.thetao[0,0,:,:]),axis=0)>0.5

z_l = ds['z_l'].values[:]
dz = np.diff(z_l)
dtemp = np.diff(ds.thetao[0,:,:,:],axis=0)
dtempdz = dtemp/np.tile(np.reshape(dz,(dz.shape[0],1,1)),(ny,nx))
thermo_ind = np.argmax(np.abs(dtempdz),axis=0) #note - 'all nan slices encountered'
thermo_ind = thermo_ind.reshape(ny*nx)
thetao = ds['thetao'].values.reshape(nz,ny*nx)

#thermocline depth
tcd = np.array([0.5*(z_l[thermo_ind[i]]+z_l[thermo_ind[i]+1]) for i in range(ny*nx)])
tcd_nan = [np.isnan(thetao[thermo_ind[i]+1,i]) for i in range(ny*nx)]
tcd_nan = np.array(tcd_nan).reshape(ny,nx)
tcd  = tcd.reshape(ny,nx)
tcd_ma = ma.masked_where((tcd_nan)|(~ebs_mask)|(~thermo_pres),tcd)

fig = plt.figure(figsize=(12,6))

ax0 = fig.add_subplot(1,2,1)
p=ax0.pcolormesh(lon,lat,tcd_ma,shading='nearest',cmap=cmo.cm.deep,vmin=0,vmax=25)
ax0.pcolormesh(lon,lat,~np.isnan(ds['thetao'][0,0,:,:].values),shading='nearest',cmap=cmap_mask,zorder=1000)
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
cbaxes = inset_axes(ax0, width="40%", height="4%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='horizontal')
cbaxes.set_xlabel('m')
efun.dar(ax0)
ax0.text(0.9,0.05,'Thermocline depth\nAug 2010',transform=ax0.transAxes,ha='right',va='bottom')
ax0.plot([177,200],[57.5,57.5],linestyle='solid',color='cyan')

yind = np.argmin(np.abs(ds.lat.values-57.5))

ax1 = fig.add_subplot(1,2,2)
lonz, z_ll = np.meshgrid(ds.lon,ds.z_l)
ax1.pcolormesh(lonz,-z_ll,ds.thetao[0,:,yind,:],shading='nearest',cmap='magma',vmin=0,vmax=12)
ax1.plot(ds.lon,-tcd_ma[yind,:],linestyle='solid',marker='none',color='cyan')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('z (m)')
ax1.set_ylim([-50,0])
ax1.text(0.9,0.1,'Temperature transect',transform=ax1.transAxes,ha='right',va='bottom')
ax1.set_xlim(177,200)

fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)