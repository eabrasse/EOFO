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

t=1
lon,lat = np.meshgrid(Sep_regrid.lon.values,Sep_regrid.lat.values)
a = np.array([Point(x, y) for x, y in zip(lon.flatten(), lat.flatten())], dtype=object)
ebs_mask_regrid = np.array([MPoly.contains(point) for point in a])
# ebs_mask_regrid = ebs_mask_regrid.reshape(Sep_regrid.MLD_restrat.isel(time=t).shape)
ebs_mask_regrid = ebs_mask_regrid.reshape(Sep_regrid.MLD_restrat.isel(year=t).shape)
# Sep_regrid_masked = Sep_regrid.isel(time=t).where(ebs_mask_regrid)
Sep_regrid_masked = Sep_regrid.isel(year=t).where(ebs_mask_regrid)

a = np.array([Point(x, y) for x, y in zip(raw.geolon.values.flatten(), raw.geolat.values.flatten())], dtype=object)
ebs_mask_raw = np.array([MPoly.contains(point) for point in a])
# ebs_mask_raw = ebs_mask_raw.reshape(Sep_raw.MLD_restrat.isel(time=t).shape)
ebs_mask_raw = ebs_mask_raw.reshape(Sep_raw.MLD_restrat.isel(year=t).shape)
# Sep_raw_masked = Sep_raw.isel(time=t).where(ebs_mask_raw)
Sep_raw_masked = Sep_raw.isel(year=t).where(ebs_mask_raw)


# Note: this works. Going to see if I can get around it though
# ebs_geom['geometry'] = MPoly
# geom = geopandas.points_from_xy(lon.flatten(),lat.flatten())
#
# gdf = geopandas.GeoDataFrame(Sep.isel(time=1).MLD_restrat.values.flatten(), geometry=geom)
# gdf.set_crs('EPSG:4326')
# subset = gdf.clip(ebs_geom)


# # initialize plot
fw,fh = efun.gen_plot_props()
fig = plt.figure(figsize=(fw*3,fh))
ax0 = fig.add_subplot(1,2,1)
# ax0 = fig.gca()

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

# ax1.plot(Sep_raw.time.dt.year.values,inner_mld,marker='o',linestyle='solid',color=tab10(0),label='inner')
# ax1.plot(Sep_raw.time.dt.year.values,middle_mld,marker='o',linestyle='dashed',color=tab10(1),label='middle')
# ax1.plot(Sep_raw.time.dt.year.values,outer_mld,marker='o',linestyle='dotted',color=tab10(2),label='outer')
flag=0
fs = 10
for key,ls in [['north','solid'],['south','dashed']]:
    ax1.plot(Sep_raw.year.values,inner_mld[key],marker='None',linestyle=ls,color=Paired(0+flag),label='inner',lw=1)
    ax1.plot(Sep_raw.year.values,middle_mld[key],marker='None',linestyle=ls,color=Paired(2+flag),label='middle',lw=1)
    ax1.plot(Sep_raw.year.values,outer_mld[key],marker='None',linestyle=ls,color=Paired(4+flag),label='outer',lw=1)
    if flag==0:
        ax1.text(0.93,0.81,'Inner',color=Paired(1),transform=ax1.transAxes,va='top',ha='right',fontsize=fs)
        ax1.text(0.93,0.88,'Middle',color=Paired(3),transform=ax1.transAxes,va='top',ha='right',fontsize=fs)
        ax1.text(0.93,0.95,'Outer',color=Paired(5),transform=ax1.transAxes,va='top',ha='right',fontsize=fs)
        ax1.text(0.1,0.93,'North = solid\nSouth = dashed',color='k',transform=ax1.transAxes,va='top',ha='left',fontsize=fs)
        flag+=1

ax1.set_xlabel('Year')
ax1.set_ylabel('Mean Summer MLD (m)')
# ax1.legend()
ax1.set_ylim(7.5,27.5)

fig.subplots_adjust(left=0.1,right=0.95,wspace=0.4,bottom=0.2,top=0.95)
plt.show(block=False)
plt.pause(0.1)

# # for t in range(len(Sep.time.dt.year.values):
# #     # t=100 #while testing
#
#
#
#
#     # now add heatmap
#
#
#
#
#
#
#
#     # now tidy up and show the plot
#     fig.subplots_adjust(right=0.9,left=0.08,bottom=0.15,top = 0.9)
#     # fig.tight_layout()
#     # plt.show(block=False)
#     # plt.pause(0.1)
#     outfn = home+f'plots/MOM6 MLD/figure_{t:0>4}.png'
#     plt.savefig(outfn)
#     plt.close()
# #
# # # close netCDF datasets
# # ds_mom6_subset.close()
