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

def willmott(m,o):
    """
    Calculates willmott skill score between two vectors, a model and a set of observations
    """
    # if len(m)!=len(o):
    #     error('Vectors must be the same length!');
    # end

    MSE = np.nanmean((m - o)**2)
    denom1 = abs(m - np.nanmean(o))
    denom2 = abs(o - np.nanmean(o))
    denom = np.nanmean((denom1 + denom2)**2)
    
    if denom==0:
        WS = 0
    else:
        WS = 1 - MSE/denom
    
    return WS

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
dense=False
if dense:
    # add mom6 data
    dense_opendap_url = 'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/regrid/r20241015/rhoinsitu.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'
    ds_dense = xr.open_dataset(dense_opendap_url)

efvar = 'NO3'
momvar = 'sfc_no3'
units = 'umole L-1'
vmin=0
vmax=30
opendap_url= 'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/regrid/r20241015/'+momvar+'.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'
ds_mom6 = xr.open_dataset(opendap_url)

if dense:
    scale = (10**6)/1000 # 10^6 umoles in a mole times (1/1000) m3 per L
else:
    scale = 1.027*10**6


ds_mom6_subset0 = ds_mom6.sel(lat=slice(50,70)).load()
# ds_dense_subset0 = ds_dense.sel(lat=slice(50,60),z_l=2.5).load()

# Generate lat bins and lon bins
lonvec = ds_mom6_subset0.lon.values
# I checked - the diff is constant
dlon = lonvec[1]-lonvec[0]
lonedges = [lon-0.5*dlon for lon in lonvec]
lonedges.append(lonvec[-1]+0.5*dlon)

latvec = ds_mom6_subset0.lat.values
dlat = latvec[1]-latvec[0]
latedges = [lat-0.5*dlat for lat in latvec]
latedges.append(latvec[-1]+0.5*dlat)

timemax = pd.to_datetime(ds_mom6_subset0.time.values.max())

data_fn = home+'data/EBO-AK_v0.97_NUT-CDF_compiled_on_06-18-24.nc'
grid_fn = home+'data/Bering10K_extended_grid.nc'
ds = nc.Dataset(data_fn)
dsg = nc.Dataset(grid_fn)
lonr= dsg['lon_rho'][:]
latr = dsg['lat_rho'][:]
h=dsg['h'][:]
n = len(ds[efvar])

data_dict = {}
for key in ds.variables.keys():
    if len(ds[key])==n:
        data_dict[key] = ds[key][:]

data_dict['datetime'] = [datetime(1900,1,1)+timedelta(days=tt) for tt in ds['TIME'][:]]

df = pd.DataFrame.from_dict(data_dict,orient='columns')
df.set_index(df.datetime)
df = df.drop(df[df.datetime>timemax].index)

df['lon_ind'] = np.digitize(df.LONGITUDE,lonedges)
df['lat_ind'] = np.digitize(df.LATITUDE,latedges)
# go ahead and drop samples that are outside the region of interest
df=df.drop(df[df.lat_ind>=len(latvec)].index)
df=df.drop(df[df.lon_ind>=len(lonvec)].index)

# df['year'] = [dt.year for dt in df.datetime]
# df['month'] = [dt.month for dt in df.datetime]
df['yearmonth'] = ['{}/{}'.format(dt.year,dt.month) for dt in df.datetime]
df['mom6'] = np.zeros(len(df))*np.nan

# years = df.year.unique()
# months = df.month.unique()
yearmonths = df.yearmonth.unique()
yearmonths.sort()


ymsamplecount = []
for yearmonth in yearmonths:
    month = yearmonth.split('/')[1]
    year = yearmonth.split('/')[0]
    # gg = df[(df.yearmonth=='{}/{}'.format(year,month))&(df.PRESSURE<20)]
    gg = df[(df.yearmonth==yearmonth)&(df.PRESSURE<20)]
    # ymsamplecount.append(len(gg))
    # if yearmonth == '{}/{}'.format(month,year):
    fw, fh = efun.gen_plot_props()
    fig = plt.figure(figsize=(14,6))
    gs = GridSpec(1,3)
    ax0 = fig.add_subplot(gs[:2])
    # ax0=fig.gca()

    # add coastline
    # ax0.pcolormesh(lonr,latr,maskr,cmap=cmap_mask,shading='nearest',zorder=5)
    ds_mom6_subset=ds_mom6_subset0.sel(time='{}-{:0>2}'.format(year,month))
    if dense:
        ds_dense_subset=ds_dense.sel(lat=slice(50,70),z_l=2.5,time='{}-{:0>2}'.format(year,month))
        varm = scale*ds_dense_subset.rhoinsitu.values[0,:,:]*ds_mom6_subset[momvar].values[0,:,:]
    else:
        varm = scale*ds_mom6_subset[momvar].values[0,:,:]
    
    ax0.pcolormesh(ds_mom6_subset.lon,ds_mom6_subset.lat,varm,shading='nearest',cmap=cmo.cm.dense,vmin=vmin,vmax=vmax)
    ax0.contour(lonr,latr,h,levels=np.arange(0,1000,50),colors='k',linewidths=0.5,zorder=6,linestyles=':')

    # add sample locations
    p=ax0.scatter(gg.LONGITUDE,gg.LATITUDE,s=20,c=gg[efvar],cmap=cmo.cm.dense,edgecolors='k',linewidths=0.5,vmin=vmin,vmax=vmax)
    cbaxes = inset_axes(ax0, width="40%", height="4%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0,-0.05,1,1))
    cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
    cb = fig.colorbar(p, cax=cbaxes,orientation='horizontal')
    cbaxes.set_xlabel(efvar + ' ' + units)
    ax0.text(0.9,0.1,'Sample locations\n{}/{}'.format(month,year),transform=ax0.transAxes,ha='right',va='bottom')
    ax0.text(0.1,0.9, 'EcoFOCI samples\n{} in surface 20m'.format(efvar),transform=ax0.transAxes,ha='left',va='top')
    ax0.axis([180,210,50,65])
    efun.dar(ax0)
    ax0.set_xlabel('Longitude (E)')
    ax0.set_ylabel('Latitude (N)')

    # extract MOM6 value at sample location and compare quantitatively
    ax1 = fig.add_subplot(gs[2])

    gg['mom6'] = gg.apply(lambda row: varm[row.lat_ind,row.lon_ind],axis=1)
    df.loc[df.index.isin(gg.index),'mom6'] = gg['mom6'][:]

    ax1.scatter(gg[efvar],gg.mom6,c='k',s=8,zorder=15)
    ax1.set_xlabel('EcoFOCI '+efvar+' upper 20m')
    ax1.set_ylabel('MOM6 surf '+efvar)
    ax1.plot([0,vmax],[0,vmax],marker=None,linestyle='dashed',color='gray',lw=0.5,zorder=5)
    ax1.set_xlim([0,vmax])
    ax1.set_ylim([0,vmax])
    ax1.set_aspect(1)

    # fig.show(block=False)
    fig.subplots_adjust(left=0.1,right=0.98,wspace=0.4)
    # plt.show(block=False)
    # plt.pause(0.1)
    outfn = home+'plots/ecoFOCI vs MOM6/Surf '+efvar+'/Surf_'+efvar+'_{:}_{:0>2}.png'.format(year,month)
    plt.savefig(outfn)
    plt.close()
ds_mom6_subset0.close()

#calculate some statistics and put 'em in a file
bias = np.mean(df.mom6-df[efvar])
rmse = np.sqrt(np.mean((df.mom6-df[efvar])**2))
# rsquared = np.corrcoef(df[efvar],df.mom6)[0,1]**2
mask = ~np.isnan(df[efvar])&~np.isnan(df.mom6)
slope, intercept, r, p, se = linregress(df[efvar][mask], df.mom6[mask])
rsquared = r**2
wss = willmott(df.mom6,df[efvar])
fname = home+'plots/ecoFOCI vs MOM6/Surf '+efvar+'/'+efvar+'_stats.txt'
f = open(fname, 'w')
f.write('ecoFOCI nutrient data compendium vs MOM6 Northeast Pacific grid hindcast\n')
f.write(efvar+' upper 20m\n')
f.write('Model bias = {:.2} {}\n'.format(bias,units))
f.write('RMSE = {:.2} {}\n'.format(rmse,units))
f.write('R2 = {:.2} {}\n'.format(rsquared,units))
f.write('WSS = {:.2} {}'.format(wss,units))
f.close()

fig = plt.figure(figsize=(5,5))
ax = fig.gca()
ax.scatter(df[efvar],df.mom6,c='k',s=8,zorder=15,alpha=0.5)
ax.set_xlabel('EcoFOCI {} upper 20m'.format(efvar))
ax.set_ylabel('MOM6 surf {}'.format(efvar))
ax.plot([-1,vmax],[-1,vmax],marker=None,linestyle='dashed',color='magenta',lw=2,zorder=20)
ax.set_aspect(1)
ax.set_xlim([-1,vmax])
ax.set_ylim([-1,vmax])

fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)