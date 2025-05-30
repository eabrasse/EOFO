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

opendap_url_regrid= 'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/regrid/r20241015/MLD_restrat.nep.full.hcast.monthly.regrid.r20241015.199301-201912.nc'

ds_regrid= xr.open_dataset(opendap_url_regrid)
regrid = ds_regrid.sel(lat=slice(50,70)).load()
regrid['year'] = regrid.time.dt.year

opendap_url_raw= 'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/raw/r20241015/MLD_restrat.nep.full.hcast.monthly.raw.r20241015.199301-201912.nc'
ds_raw = xr.open_dataset(opendap_url_raw)
ds_static = xr.open_dataset(
    "http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/raw/r20241015/ocean_static.nc"
)
raw = xr.merge([ds_raw,ds_static])

raw = raw.sel(yh=slice(50,70)).load()
# geolon,geolat = np.meshgrid(ds_mom6.geolon,ds_mom6.geolat)
# h = ds_mom6.deptho
raw['year'] = raw.time.dt.year
Sep_regrid = regrid.where((regrid.time.dt.month>6)&(regrid.time.dt.month<10),drop=True)[["MLD_restrat","year"]]
Sep_regrid = Sep_regrid.groupby('year').mean()
Sep_raw = raw.where((raw.time.dt.month>6)&(raw.time.dt.month<10),drop=True)[["MLD_restrat","year"]]
Sep_raw = Sep_raw.groupby('year').mean()

t=1

# # initialize plot
fw,fh = efun.gen_plot_props()
fig = plt.figure(figsize=(fw*3,fh))
ax0 = fig.add_subplot(1,2,1)
# ax0 = fig.gca()

# & to replot with everything demeaned
fig2 = plt.figure(figsize=(fw*4,fh))
ax2 = fig2.add_subplot(1,2,1)

p=ax0.pcolormesh(lon,lat,Sep_regrid_masked.MLD_restrat,shading='nearest',cmap=cmo.cm.deep,vmin=0,vmax=25)
# p=ax0.contourf(ds_mom6.geolon,ds_mom6.geolat,Sep_masked.MLD_restrat,levels=range(0,100),cmap=cmo.cm.deep,zorder=50)
# ax0.pcolormesh(lon,lat,~np.isnan(Sep_regrid.MLD_restrat.isel(time=t)),shading='nearest',cmap=cmap_mask,zorder=100)
ax0.pcolormesh(lon,lat,~np.isnan(Sep_regrid.MLD_restrat.isel(year=t)),shading='nearest',cmap=cmap_mask,zorder=100)
ax0.contour(raw.geolon,raw.geolat,raw.deptho,levels=[50,100,200,1000,3500],linewidths=0.5,linestyles='dashed',colors='cyan',zorder=200)
ax0.axhline(y=60,linestyle='dashed',color='magenta',lw=0.7,zorder=250)

cbaxes = inset_axes(ax0, width="4%", height="40%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_ylabel('m')
ax0.set_xlabel('Longitude')
ax0.set_xlim(175,205)
ax0.set_ylabel('Latitude')
efun.dar(ax0)
ax0.text(0.1,1.03,'Mixed Layer Depth (MLD_restrat)',transform=ax0.transAxes,ha='left',fontsize=12,zorder=600,va='bottom')
# ax0.text(0.05,0.13,pd.to_datetime(Sep_regrid.time[t].values).strftime("%b %Y"), transform=ax0.transAxes, ha='left', zorder=205, fontsize=8, color='k', fontweight='bold', bbox=dict(facecolor='white',edgecolor='None'))
ax0.text(0.05,0.13,f"Jul–Sep {int(Sep_regrid_masked.year.values):}", transform=ax0.transAxes, ha='left', zorder=205, fontsize=8, color='k', fontweight='bold', bbox=dict(facecolor='white',edgecolor='None'))
ax0.text(0.05,0.05,'Bathy contours at 50, 100, 200, 1000, 3500 m', transform=ax0.transAxes, ha='left', zorder=205, fontsize=8, color='k', bbox=dict(facecolor='white',edgecolor='None'))

ax0.plot(360-164.06,56.87,marker='*',mfc='yellow',mec='none',markersize=12)

ax1 = fig.add_subplot(1,2,2)
#using definitions for inner/mid/outer EBS shelf from Stabeno et al 2001
# get a time series of mean Sep EBS shelf region MLD
inner_mld = {}
inner_mld['north'] = Sep_raw.where((raw.deptho<50)&(raw.geolat>60)).MLD_restrat.mean(dim=["xh","yh"])
inner_mld['south'] = Sep_raw.where((raw.deptho<50)&(raw.geolat<60)).MLD_restrat.mean(dim=["xh","yh"])

middle_mld = {}
middle_mld['north'] = Sep_raw.where((raw.deptho>50)&(raw.deptho<100)&(raw.geolat>60)).MLD_restrat.mean(dim=["xh","yh"])
middle_mld['south'] = Sep_raw.where((raw.deptho>50)&(raw.deptho<100)&(raw.geolat<60)).MLD_restrat.mean(dim=["xh","yh"])

outer_mld = {}
outer_mld['north'] = Sep_raw.where((raw.deptho>100)&(raw.deptho<180)&(raw.geolat>60)).MLD_restrat.mean(dim=["xh","yh"])
outer_mld['south'] = Sep_raw.where((raw.deptho>100)&(raw.deptho<180)&(raw.geolat<60)).MLD_restrat.mean(dim=["xh","yh"])

inner_mld['north_dm'] = inner_mld['north']-inner_mld['north'].mean()
inner_mld['south_dm'] = inner_mld['south']-inner_mld['south'].mean()

middle_mld['north_dm'] = middle_mld['north']-middle_mld['north'].mean()
middle_mld['south_dm'] = middle_mld['south']-middle_mld['south'].mean()

outer_mld['north_dm'] = outer_mld['north']-outer_mld['north'].mean()
outer_mld['south_dm'] = outer_mld['south']-outer_mld['south'].mean()

# ax1.plot(Sep_raw.time.dt.year.values,inner_mld,marker='o',linestyle='solid',color=tab10(0),label='inner')
# ax1.plot(Sep_raw.time.dt.year.values,middle_mld,marker='o',linestyle='dashed',color=tab10(1),label='middle')
# ax1.plot(Sep_raw.time.dt.year.values,outer_mld,marker='o',linestyle='dotted',color=tab10(2),label='outer')




#add M2 to ax1
m2_fn = home+'data/M2_composite_1m.daily.2024.nc'
ds = nc.Dataset(m2_fn)
temp = ds['temperature'][:]
z = -ds['depth'][:]
ot = ds['time'][:]
dt_list = []
for ott in ot:
    dtt = datetime(1900,1,1)+timedelta(days=int(ott))
    dt_list.append(dtt)
dt_array = np.array(dt_list)
nz = z.shape[0]
mask = [dt.month in [7,8,9] for dt in dt_list]
mask_tile = np.tile(mask,(nz,1)).T
temp_ma = ma.masked_where(~mask_tile,temp)
dtdz = np.diff(temp_ma,axis=1) # implied /1m because the cells are all 1m
mld_ind = np.argmax(np.abs(dtdz),axis=1)
mld = np.array([0.5*(z[mi]+z[mi+1]) for mi in mld_ind])
mld_ma = ma.masked_where(~np.array(mask),mld)
years = list(set([dt.year for dt in dt_list]))
summer_mld_year_avg = np.zeros(len(years))
for yr in range(len(years)):
    year = years[yr]
    year_mask = np.array([dt.year!=year for dt in dt_list])
    mld_year_ma= ma.masked_where(year_mask,mld_ma)
    summer_mld_year_avg[yr] = np.mean(mld_year_ma)
demeaned_M2 = -(summer_mld_year_avg-summer_mld_year_avg.mean())

# compare M2 with subregions
ax3 = fig2.add_subplot(1,2,2)
ymmod = [yr in years for yr in Sep_raw.year.values] # year mask model
ymM2 = [yr in Sep_raw.year.values for yr in years] # year mask M2


flag=0
fs = 10
for key,ls in [['north','solid'],['south','dashed']]:
    ax1.plot(Sep_raw.year.values,inner_mld[key],marker='None',linestyle=ls,color=Paired(0+flag),label='inner',lw=1)
    ax1.plot(Sep_raw.year.values,middle_mld[key],marker='None',linestyle=ls,color=Paired(2+flag),label='middle',lw=1)
    ax1.plot(Sep_raw.year.values,outer_mld[key],marker='None',linestyle=ls,color=Paired(4+flag),label='outer',lw=1)
    
    ax2.plot(Sep_raw.year.values,inner_mld[key]-inner_mld[key].mean(),marker='None',linestyle=ls,color=Paired(0+flag),label='inner',lw=1)
    ax2.plot(Sep_raw.year.values,middle_mld[key]-middle_mld[key].mean(),marker='None',linestyle=ls,color=Paired(2+flag),label='middle',lw=1)
    ax2.plot(Sep_raw.year.values,outer_mld[key]-outer_mld[key].mean(),marker='None',linestyle=ls,color=Paired(4+flag),label='outer',lw=1)
    
    if key=='north':
        ax3.plot(demeaned_M2[ymM2],inner_mld[key+'_dm'][ymmod],ls='none',marker='o',mfc=Paired(0+flag),mec='none')
        ax3.plot(demeaned_M2[ymM2],middle_mld[key+'_dm'][ymmod],ls='none',marker='o',mfc=Paired(2+flag),mec='none')
        ax3.plot(demeaned_M2[ymM2],outer_mld[key+'_dm'][ymmod],ls='none',marker='o',mfc=Paired(4+flag),mec='none')
    else:
        ax3.plot(demeaned_M2[ymM2],inner_mld[key+'_dm'][ymmod],ls='none',marker='o',mec=Paired(0+flag),mfc='none')
        ax3.plot(demeaned_M2[ymM2],middle_mld[key+'_dm'][ymmod],ls='none',marker='o',mec=Paired(2+flag),mfc='none')
        ax3.plot(demeaned_M2[ymM2],outer_mld[key+'_dm'][ymmod],ls='none',marker='o',mec=Paired(4+flag),mfc='none')
    
    flag+=1

ax1.plot(years,-summer_mld_year_avg,marker='none',linestyle='solid',color='k',lw=1)
ax2.plot(years,-(summer_mld_year_avg-summer_mld_year_avg.mean()),marker='none',linestyle='solid',color='k',lw=1)

for ax in ax1,ax2:
    ax.text(0.93,0.81,'Inner',color=Paired(1),transform=ax.transAxes,va='top',ha='right',fontsize=fs)
    ax.text(0.93,0.88,'Middle',color=Paired(3),transform=ax.transAxes,va='top',ha='right',fontsize=fs)
    ax.text(0.93,0.95,'Outer',color=Paired(5),transform=ax.transAxes,va='top',ha='right',fontsize=fs)
    ax.text(0.1,0.93,'North = solid\nSouth = dashed',color='k',transform=ax.transAxes,va='top',ha='left',fontsize=fs)
    ax.text(0.93,0.74,'M2',color='k',transform=ax.transAxes,va='top',ha='right',fontsize=fs)
    ax.set_xlabel('Year')

ax1.set_ylim(7.5,30)
ax1.set_ylabel('Mean Summer MLD (m)')
ax2.set_ylabel('Mean Summer MLD (m)')
ax2.text(0.1,0.1,'Interannual average removed',transform=ax2.transAxes,ha='left',va='bottom')
ax2.axhline(0,color='gray',alpha=0.5,linestyle='dashed')

ax3.set_xlabel('M2 Mean Summer MLD (m)')
ax3.set_ylabel('MOM6 NEP10k Mean Summer MLD (m)')
ax3.axline((0, 0), slope=1,color='gray',alpha=0.5,linestyle='dashed')
ax3.set_aspect(1)



fig.subplots_adjust(left=0.1,right=0.95,wspace=0.4,bottom=0.2,top=0.95)
fig2.subplots_adjust(left=0.1,right=0.95,wspace=0.4,bottom=0.2,top=0.95)

fig3 = plt.figure(figsize=(fw,fh))

yrind = [years[i] for i in range(len(years)) if ymM2[i]]

df = pd.DataFrame(index=yrind,\
    data={'M2':demeaned_M2[ymM2],'N Inner':inner_mld['north_dm'][ymmod],'N Middle':middle_mld['north_dm'][ymmod],'N Outer':outer_mld['north_dm'][ymmod],\
    'S Inner':inner_mld['south_dm'][ymmod],'S Middle':middle_mld['south_dm'][ymmod],'S Outer':outer_mld['south_dm'][ymmod]})

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))


plt.show(block=False)
plt.pause(0.1)

