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
fname = home+'data/thetao.nep.full.hcast.monthly.regrid.r20250509.199301-202412.nc'
ds = xr.load_dataset(fname)

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
    
    thermocline[t] = tcd_ma.mean()

fig = plt.figure(figsize=(12,6))

ax0 = fig.add_subplot(1,2,1)
p=ax0.pcolormesh(lon,lat,tcd_ma,shading='nearest',cmap=cmo.cm.deep,vmin=0,vmax=30)
ax0.pcolormesh(lon,lat,~np.isnan(ds['thetao'][0,0,:,:].values),shading='nearest',cmap=cmap_mask,zorder=1000)
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
cbaxes = inset_axes(ax0, width="40%", height="4%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='horizontal')
cbaxes.set_xlabel('m')
efun.dar(ax0)
ax0.text(0.9,0.05,'Thermocline depth\n'+pd.to_datetime(ds['time'].values[t]).strftime('%b %Y'),transform=ax0.transAxes,ha='right',va='bottom')
ax0.plot([177,200],[57.5,57.5],linestyle='solid',color='cyan')


ax1 = fig.add_subplot(1,2,2)
rainbow = plt.get_cmap('rainbow',12)
for mo in [7,8,9]:
    inds=np.argwhere(ds.time.dt.month.values==mo)
    ax1.plot(ds['time'].values[inds],thermocline[inds],marker='o',linestyle='solid',mfc=rainbow(mo),mec='none',color=rainbow(mo))
    ax1.text(0.9,1.3-0.05*mo,calendar.month_abbr[mo],transform=ax1.transAxes,ha='right',color=rainbow(mo),fontweight='bold')
    
ax1.set_ylabel('(m)')
ax1.set_xlabel('Date time')
ax1.text(0.1,0.9,'EBS mean thermocline depth',transform=ax1.transAxes)

fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

save = True
if save:
    D = {}
    D['time'] = pd.to_datetime(ds['time'].values)
    D['thermocline'] = thermocline
    df = pd.DataFrame.from_dict(D)

    outfn = home+'data/MOM6-NEP_EBSmean_thermocline_JulAugSep_1993-2024.csv'
    df.to_csv(outfn)