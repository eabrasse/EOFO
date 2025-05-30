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
from scipy.stats import linregress
import calendar
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

tripno=2
gg = df0[(df0.tripno==tripno)&(df0.Maxdepth>4)]
gg['mu.x.new'] = gg.apply(lambda row: row['mu.x']%360, axis=1)

mo=9
mo_str = calendar.month_abbr[mo]

fname_mom6 = home+'data/thetao.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'
ds = xr.load_dataset(fname_mom6)
ds = ds.sel(time=f'2010-{mo:02}',lat=slice(52.5,62.5),lon=slice(185,195))
# ds = ds.sel(z_l=0,method='nearest')


# Generate lat bins and lon bins
lonvec = ds.lon.values
# I checked - the diff is constant
dlon = lonvec[1]-lonvec[0]
lonedges = [lon-0.5*dlon for lon in lonvec]
lonedges.append(lonvec[-1]+0.5*dlon)
lonedges = np.array(lonedges)

latvec = ds.lat.values
dlat = latvec[1]-latvec[0]
latedges = [lat-0.5*dlat for lat in latvec]
latedges.append(latvec[-1]+0.5*dlat)
latedges = np.array(latedges)

zvec = ds.z_l.values
dz = np.diff(zvec)
zedges = [zvec[i+1]+0.5*dz[i] for i in range(len(zvec)-1)]
zedges.insert(0,float(0))
zedges = np.array(zedges)

gg['lon_ind'] = np.digitize(gg['mu.x.new'],lonedges)
gg['lat_ind'] = np.digitize(gg['mu.y'],latedges)
gg['z_ind'] = np.digitize(gg['CorrectedDepth'],zedges)
# go ahead and drop samples that are outside the region of interest
gg=gg.drop(gg[gg.lat_ind>=len(latvec)].index)
gg=gg.drop(gg[gg.lon_ind>=len(lonvec)].index)
gg=gg.drop(gg[gg.z_ind>=len(zvec)].index)
gg['mom6'] = gg.apply(lambda row: float(ds['thetao'].isel(time=0,z_l=row.z_ind,lat=row.lat_ind,lon=row.lon_ind)),axis=1)

vmin=0
vmax=10

#one figure to plot SST & seal track
fw, fh = efun.gen_plot_props()
fig1 = plt.figure(figsize=(fw*4,fh))

ax = fig1.add_subplot(1,4,1)
lon,lat = np.meshgrid(ds.lon,ds.lat)
p=ax.pcolormesh(lon,lat,ds.thetao[0,0,:,:],shading='nearest',cmap='magma',vmin=vmin,vmax=vmax)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
cbaxes = inset_axes(ax, width="5%", height="50%", loc='upper right',bbox_transform=ax.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig1.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_xlabel(r'$^{\circ}$C')
ax.plot(gg['mu.x.new'],gg['mu.y'],marker='none',linestyle='solid',color='cyan')
efun.dar(ax)
ax.text(0.9,0.1,f'SST\n{mo_str} 2010',transform=ax.transAxes,ha='right',va='bottom')
ax.plot(ds.lon[32].values,ds.lat[42].values,marker='x',linestyle='none',color='k',markersize=12)


ax0 = fig1.add_subplot(1,4,2)
p=ax0.scatter(gg.dt,-gg.CorrectedDepth,c=gg.etemp,cmap='magma',vmin=vmin,vmax=vmax,s=5)
# cbaxes = inset_axes(ax0, width="4%", height="40%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
# cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
# cb = fig1.colorbar(p, cax=cbaxes,orientation='vertical')
# cbaxes.set_ylabel(r'$^{\circ}$C')
ax0.set_xlabel('Time')
ax0.set_ylabel('Depth')
ax0.text(0.98,0.05,f'Northern Fur Seal {dbid:}\nTrip no. {tripno:}',transform=ax0.transAxes,ha='right',va='bottom',fontsize=10)
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
plt.setp( ax0.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
ax0.set_ylim([-143,3])

ax1 = fig1.add_subplot(1,4,3)
p=ax1.scatter(gg.dt,-gg.CorrectedDepth,c=gg.mom6,cmap='magma',vmin=vmin,vmax=vmax,s=5)
ax1.set_xlabel('Time')
# ax1.set_ylabel('Depth')
ax1.text(0.98,0.05,f'MOM6 {mo_str} 2010\nTemps along NFS path',transform=ax1.transAxes,ha='right',va='bottom',fontsize=10)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
plt.setp( ax1.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
ax1.set_ylim([-143,3])

ax2 = fig1.add_subplot(1,4,4)
ax2.scatter(gg.etemp,gg.mom6,s=5)
ax2.plot([0,10],[0,10],linestyle='dashed',color='cyan')
ax2.set_aspect(1)
ax2.set_xlim([-0.3,10.3])
ax2.set_ylim([-0.3,10.3])
ax2.set_xlabel('Fur Seal temperature')
ax2.set_ylabel('MOM6 temperature')


mask = ~np.isnan(gg.mom6)&~np.isnan(gg.etemp)
bias = np.mean(gg.mom6[mask]-gg.etemp[mask])
rmse = np.sqrt(np.mean((gg.mom6[mask]-gg.etemp[mask])**2))
slope, intercept, r, p, se = linregress(gg.mom6[mask], gg.etemp[mask])
rsquared = r**2
units = 'deg C'
fname = home+'plots/NFS{:}_trip{:}_vs_MOM6_{:02}-{:}_temp_stats.txt'.format(dbid,tripno,ds['time'].dt.month.values[0],ds['time'].dt.year.values[0])
f = open(fname, 'w')
f.write('Northern Fur Seal {:} Trip no. {:} compared with MOM6 NEP hindcast monthly averaged temperatures for {:02}-{:}\n'.format(dbid,tripno,ds['time'].dt.month.values[0],ds['time'].dt.year.values[0]))
f.write('Model bias = {:.2} {}\n'.format(bias,units))
f.write('RMSE = {:.2} {}\n'.format(rmse,units))
f.write('R2 = {:.2} {}\n'.format(rsquared,units))
f.close()


fig1.subplots_adjust(left=0.08,right=0.92,top=0.95,bottom=0.2,wspace=0.6)
# fig1.tight_layout()
plt.show(block=False)
plt.pause(0.1)

