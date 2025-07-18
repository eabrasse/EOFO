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
import pandas as pd
from datetime import datetime, timedelta
import pytz
import cmocean as cmo
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
from scipy.stats import linregress
import geopandas
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
import numpy.ma as ma
import calendar
import pickle

#close all currently open figures
plt.close('all')

# import useful labeling string
# atoz = string.ascii_lowercase

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
ds_static = xr.open_dataset(
    "http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/raw/r20241015/ocean_static.nc"
)

# ds_mom6_subset = ds_mom6.sel(lat=slice(54,58),lon=190.5,time='2004-07',method='nearest').load()
ds = ds.sel(lat=slice(50,70),lon=slice(175,205),z_l=slice(0,60))
# ds['year'] = ds.time.dt.year
ds = ds.where((ds.time.dt.month>6)&(ds.time.dt.month<10),drop=True)[["thetao","time","lon","lat","z_l"]]
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
# is thermocline present? y/n
thermo_pres = np.nanmax(np.abs(ds.thetao[0,:,:,:]-ds.thetao[0,0,:,:]),axis=0)>0.5

z_l = ds['z_l'].values[:]
dz = np.diff(z_l)

thermocline = np.zeros((nt))
tcd_ma_mat = np.zeros((nt,ny,nx))
for t in range(nt):

    dtemp = np.diff(ds.thetao[t,:,:,:],axis=0)
    dtempdz = dtemp/np.tile(np.reshape(dz,(dz.shape[0],1,1)),(ny,nx))
    thermo_ind = np.argmax(np.abs(dtempdz),axis=0) #note - 'all nan slices encountered'
    thermo_ind = thermo_ind.reshape(ny*nx)
    thetao = ds['thetao'].values[t,:,:,:].reshape(nz,ny*nx)

    #thermocline depth
    tcd = np.array([0.5*(z_l[thermo_ind[i]]+z_l[thermo_ind[i]+1]) for i in range(ny*nx)])
    # tcd = [z_l[thermo_ind[i]] for i in range(ny*nx)]
    tcd_nan = [np.isnan(thetao[thermo_ind[i]+1,i]) for i in range(ny*nx)]
    tcd_nan = np.array(tcd_nan).reshape(ny,nx)
    tcd  = np.array(tcd).reshape(ny,nx)
    tcd_ma = ma.masked_where((tcd_nan)|(~ebs_mask)|(~thermo_pres),tcd)
    tcd_ma_mat[t,:,:] = tcd_ma
    
    thermocline[t] = tcd_ma.mean()

# calculate correlation map
tcd_ma_array = tcd_ma_mat.reshape(nt,ny*nx)
tcd_ma_corr = np.array([ma.corrcoef(tcd_ma_array[:,i],thermocline[:])[0,1]**2 for i in range(ny*nx)]).reshape(ny,nx)
tcd_ma_corr = ma.masked_where((tcd_nan)|(~ebs_mask)|(~thermo_pres),tcd_ma_corr)

fig = plt.figure(figsize=(8,6))

ax0 = fig.gca()
p=ax0.pcolormesh(lon,lat,tcd_ma_corr,shading='nearest',cmap=cmo.cm.speed,vmin=0.2,vmax=1)
ax0.pcolormesh(lon,lat,~np.isnan(ds['thetao'][0,0,:,:].values),shading='nearest',cmap=cmap_mask,zorder=1000)
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
cbaxes = inset_axes(ax0, width="40%", height="4%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='horizontal')
cbaxes.set_xlabel(r'$R^{2}$')
efun.dar(ax0)
ax0.text(0.05,0.92,'Correlation (R2) with\nEBS-mean summertime thermocline depth',transform=ax0.transAxes,ha='left',va='top',zorder=1001,bbox=dict(facecolor='white',edgecolor='None'))
ax0.contour(ds_static.geolon,ds_static.geolat,ds_static.deptho,levels=[50,100,200,1000,3500],linewidths=0.5,linestyles='dashed',colors='k',zorder=200)

lonM2 = 195.94
latM2 = 56.87

lonind = np.argmin(np.abs(ds.lon.values-lonM2))
latind = np.argmin(np.abs(ds.lat.values-latM2))

ax0.plot(lonM2,latM2,marker='d',mfc='cyan',mec='k',markersize=10)
ax0.text(lonM2+0.3,latM2+0.,'M2',color='k',ha='left',va='center')
ax0.text(0.9,0.05,'R2 at M2 is {:.2}'.format(tcd_ma_corr[latind,lonind]),transform=ax0.transAxes,ha='right',va='bottom',zorder=1002,bbox=dict(facecolor='white',edgecolor='None'))
ax0.text(0.05,0.05,'Bathy contours at 50, 100, 200, 1000, 3500 m', transform=ax0.transAxes, ha='left', zorder=205, fontsize=8, color='k', bbox=dict(facecolor='white',edgecolor='None'))

fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

ds.close()