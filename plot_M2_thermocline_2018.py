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

# load in data and grid
home = '/Users/elizabethbrasseale/projects/EOFO/'

month_list = [7,8,9]

# fname = home+'data/thetao.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'
# ds0 = xr.load_dataset(fname)
# lonM2 = 195.94
# latM2 = 56.87
# ds0 = ds0.sel(lat=latM2,lon=lonM2,method='nearest').load()
#
# # ds0 = ds0.where((ds0.time.dt.month>6)&(ds0.time.dt.month<10),drop=True)[["thetao","time","lon","lat","z_l"]]
# ds0 = ds0.where((ds0.z_l<60), drop=True)[["thetao","time","lon","lat","z_l"]]
# ds0['year'] = ds0.time.dt.year
# ds0['month'] = ds0.time.dt.month
# nt,nz = ds0['thetao'].shape
# modmonthmask = [mo in [7,8,9] for mo in ds0.month.values]
# modmonthmask_tile = np.tile(modmonthmask,(nz,1)).T

# z_l = ds0['z_l'].values[:]
# dz = np.diff(z_l)

# dtemp = np.diff(ds0.thetao.values,axis=1)
# dtempdz = dtemp/np.tile(dz,(nt,1))
# thermo_ind = np.argmax(np.abs(dtempdz),axis=1)
# thermocline = np.array([0.5*(z_l[thermo_ind[t]]+z_l[thermo_ind[t]+1]) for t in range(nt)])
# thermocline_ma = ma.masked_where(~np.array(modmonthmask),thermocline)
# thetao = ds0['thetao'].values
# thetao_ma = ma.masked_where(~np.array(modmonthmask_tile),thetao) # for plotting

#add M2 to ax1
m2_fn = home+'data/M2_composite_1m.daily.2024.nc'
ds1 = xr.load_dataset(m2_fn)
ds1 = ds1.sel(time=slice('2018-07-01','2018-09-30')).load()
temp = ds1['temperature'].values[:]
z = -ds1['depth'].values[:]
dt_array = pd.to_datetime(ds1['time'].values[:])
nz = z.shape[0]
# mask = [dt.month in [7,8,9] for dt in dt_array]
# mask_tile = np.tile(mask,(nz,1)).T
# temp_ma = ma.masked_where(~mask_tile,temp)
dtdz = np.diff(temp,axis=1) # implied /1m because the cells are all 1m
mld_ind = np.argmax(np.abs(dtdz),axis=1)
mld = np.array([0.5*(z[mi]+z[mi+1]) for mi in mld_ind])
# mld_ma = ma.masked_where(~np.array(mask),mld)


# yearsmod = list(set(ds0.year.values))
# D = {'thermocline_ma':thermocline_ma,'year':ds0.year.values}
# df = pd.DataFrame.from_dict(D)
# thermocline_year_avg = df.groupby('year')['thermocline_ma'].mean()

# yearsM2 = list(set([dt.year for dt in dt_array]))
# summer_mld_year_avg = np.zeros(len(yearsM2))
# for yr in range(len(yearsM2)):
#     year = yearsM2[yr]
#     year_mask = np.array([dt.year!=year for dt in dt_array])
#     mld_year_ma= ma.masked_where(year_mask,mld_ma)
#     summer_mld_year_avg[yr] = np.mean(mld_year_ma)

# ymmod = [yr in yearsM2 for yr in yearsmod] # year mask model
# ymM2 = [yr in yearsmod for yr in yearsM2] # year mask M2



fig0 = plt.figure(figsize=(12,6))

# ax00 = fig0.add_subplot(2,1,1)
# tz, z_ll = np.meshgrid(ds0.time.values,-ds0.z_l)
# p=ax00.pcolormesh(tz,z_ll,thetao_ma.T,shading='nearest',cmap='magma',vmin=0,vmax=12)
# ax00.plot(ds0.time.values,-thermocline_ma,linestyle='dashed',lw=1,marker='none',color='cyan')
#
# cbaxes = inset_axes(ax00, width="40%", height="4%", loc='upper right',bbox_transform=ax00.transAxes,bbox_to_anchor=(0,-0.05,1,1))
# cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
# cb = fig0.colorbar(p, cax=cbaxes,orientation='horizontal')
# cbaxes.set_xlabel('°C')
# ax00.text(0.9,0.05,'MOM6 Temperature @ M2',transform=ax00.transAxes,ha='right',va='bottom')

# ax01 = fig0.add_subplot(2,1,2)
ax01 = fig0.gca()
z = -ds1['depth']
tt, zz = np.meshgrid(dt_array,z)
ax01.pcolormesh(tt,zz,temp.T,shading='nearest',cmap='magma',vmin=0,vmax=12)
ax01.plot(dt_array,mld,linestyle='dashed',lw=1,marker='none',color='cyan')
ax01.set_xlabel('Time')
ax01.set_ylabel('Depth')
ax01.text(0.9,0.05,'M2 Temperature',transform=ax01.transAxes,ha='right',va='bottom')


ax01.set_xlabel('Time')
ax01.set_ylabel('Depth')
# ax01.set_ylim([-70,0])
ax01.grid(zorder=5000)

fig0.tight_layout()




t_list=[15,50,75,90]
fig1 = plt.figure(figsize=(3*len(t_list),8))
count=1
for t in t_list:
    ax10 = fig1.add_subplot(2,len(t_list),count)
    ax10.plot(temp[t,:],z,linestyle='solid',color='orchid',marker='none',lw=2.0)
    ax10.axhline(y=mld[t],linestyle='dashed',color='cyan',lw=1)
    ax10.set_xlabel('Temp (°C)')
    if count==1:
        ax10.set_ylabel('Depth (m)')
    ax10.text(0.1,0.9,dt_array[t].strftime('%b %d, %Y'),transform=ax10.transAxes)
    ax10.grid(True)
    
    ax11 = fig1.add_subplot(2,len(t_list),len(t_list)+count)
    ax11.plot(dtdz[t,:],0.5*(z.values[:-1]+z.values[1:]),linestyle='solid',color='seagreen',marker='none',lw=2.0)
    ax11.axhline(y=mld[t],linestyle='dashed',color='cyan',lw=1)
    ax11.set_xlabel('dTemp/dZ (°C/m)')
    if count==1:
        ax11.set_ylabel('Depth (m)')
    # ax11.text(0.1,0.9,dt_array[t].strftime('%b %d, %Y'),transform=ax11.transAxes)
    ax11.grid(True)
    count+=1

plt.show(block=False)
plt.pause(0.1)

