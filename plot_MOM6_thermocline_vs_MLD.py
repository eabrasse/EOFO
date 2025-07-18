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
tab10 = plt.get_cmap('tab10',10)

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
ds = ds.where((ds.time.dt.month>6)&(ds.time.dt.month<10),drop=True)[["thetao","time","lon","lat","z_l"]]
nt,nz,ny,nx = ds['thetao'].shape

# Add MLD
mld_name = 'MLD_restrat'
opendap_url_mld= f'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northeast_pacific/full_domain/hindcast/monthly/regrid/r20250509/{mld_name:}.nep.full.hcast.monthly.regrid.r20250509.199301-202412.nc'

ds_mld= xr.open_dataset(opendap_url_mld)
ds_mld['year'] = ds_mld.time.dt.year
ds_mld=ds_mld.sel(lat=slice(50,70),lon=slice(175,205))
ds_mld=ds_mld.where((ds_mld.time.dt.month>6)&(ds_mld.time.dt.month<10),drop=True)[[mld_name,"year"]]
mld=ds_mld[mld_name].values.reshape(nt,ny,nx)


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
thermocline_mo = np.zeros((nt))
mld_mo = np.zeros((nt))
diff_mo = np.zeros((nt))
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
    mld_ma = ma.masked_where((tcd_nan)|(~ebs_mask)|(~thermo_pres),mld[t,:,:])
    
    thermocline_mo[t] = tcd_ma.mean()
    mld_mo[t] = mld_ma.mean()
    diff= mld_ma-tcd_ma
    diff_mo[t] = diff.mean()

ds['thermocline'] = (['time'],thermocline_mo)
ds['mld'] = (('time'),mld_mo)
ds['diff'] = (('time'),diff_mo)
ds['year'] = ds.time.dt.year
ds.set_coords('year')
yearsmod = list(set(ds.year.values))
thermocline_year_avg = ds.groupby('time.year').mean()['thermocline']
mld_year_avg = ds.groupby('time.year').mean()['mld']
diff_year_avg = ds.groupby('time.year').mean()['diff']

fig = plt.figure(figsize=(14,5))

ax0 = fig.add_subplot(1,3,1)
p=ax0.pcolormesh(lon,lat,tcd_ma,shading='nearest',cmap=cmo.cm.deep,vmin=0,vmax=25)
ax0.pcolormesh(lon,lat,~np.isnan(ds['thetao'][0,0,:,:].values),shading='nearest',cmap=cmap_mask,zorder=1000)
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
cbaxes = inset_axes(ax0, width="40%", height="4%", loc='upper right',bbox_transform=ax0.transAxes,bbox_to_anchor=(0,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig.colorbar(p, cax=cbaxes,orientation='horizontal')
cbaxes.set_xlabel('m')
efun.dar(ax0)
date_str = pd.to_datetime(ds['time'].values[-1]).strftime('%b %Y')
ax0.text(0.9,0.05,f'Thermocline depth\n{date_str:}',transform=ax0.transAxes,ha='right',va='bottom')


ax1 = fig.add_subplot(1,3,2)
ax1.pcolormesh(lon,lat,mld_ma,shading='nearest',cmap=cmo.cm.deep,vmin=0,vmax=25)
ax1.pcolormesh(lon,lat,~np.isnan(ds['thetao'][0,0,:,:].values),shading='nearest',cmap=cmap_mask,zorder=1000)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
efun.dar(ax1)
ax1.text(0.9,0.05,f'{mld_name:} depth\n{date_str:}',transform=ax1.transAxes,ha='right',va='bottom')

ax2 = fig.add_subplot(1,3,3)
p2=ax2.pcolormesh(lon,lat,mld_ma-tcd_ma,shading='nearest',cmap=cmo.cm.balance,vmin=-10,vmax=10)
ax2.pcolormesh(lon,lat,~np.isnan(ds['thetao'][0,0,:,:].values),shading='nearest',cmap=cmap_mask,zorder=1000)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cbaxes2 = inset_axes(ax2, width="40%", height="4%", loc='upper right',bbox_transform=ax2.transAxes,bbox_to_anchor=(0,-0.05,1,1))
cbaxes2.tick_params(axis='both',which='both',labelsize=8,size=2)
cb2 = fig.colorbar(p2, cax=cbaxes2,orientation='horizontal')
cbaxes2.set_xlabel('m')
efun.dar(ax2)
ax2.text(0.9,0.05,f'{mld_name:}-Thermocline\n{date_str:}',transform=ax2.transAxes,ha='right',va='bottom',zorder=1001)
ax2.text(0.025,0.3,'mean diff = {:.2} m'.format(np.mean(mld_ma-tcd_ma)),transform=ax2.transAxes,ha='left',va='bottom')

fig.tight_layout()

fig2 = plt.figure(figsize=(10,5))
ax3 = fig2.add_subplot(1,2,1)
ax3.plot(yearsmod,thermocline_year_avg,marker='o',linestyle='solid',color=tab10(0),mfc=tab10(0),mec=tab10(0))
ax3.text(0.2,0.92,'Thermocline',transform=ax3.transAxes,color=tab10(0))
ax3.plot(yearsmod,mld_year_avg,marker='o',linestyle='solid',color=tab10(1),mfc=tab10(1),mec=tab10(1))
ax3.text(0.2,0.85,mld_name,transform=ax3.transAxes,color=tab10(1))
ax4 = fig2.add_subplot(1,2,2)
ax4.plot(yearsmod,diff_year_avg,marker='o',linestyle='solid',color=tab10(2),mfc=tab10(2),mec=tab10(2))
ax4.text(0.3,0.9,f'Difference ({mld_name:}-Thermocline)',transform=ax4.transAxes,color=tab10(2))
ax4.text(0.1,0.2,'Mean = {:.2} m'.format(diff_year_avg.mean()),transform=ax4.transAxes,color=tab10(2))
for ax in ax3,ax4:
    ax.set_xlabel('Year')
    ax.set_ylabel('Depth (m)')

plt.show(block=False)
plt.pause(0.1)