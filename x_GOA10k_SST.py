# Code to calculate some statistics on SST from GOA SSP 585 projection
# EAB 5/29/2025

import os
import sys
import numpy as np
import netCDF4 as nc
import pandas as pd
from datetime import datetime, timedelta
import pytz
import geopandas
import xarray as xr
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
import numpy.ma as ma
import pickle

# load in data and grid
home = '/home/eabrasse/'

dir0 = '/home/darr/goa-output/'
data_dir  = dir0+'monthly_aves_nep_wb_ssp585/'
f_list = os.listdir(data_dir) # list all files in folder
f_list = [x for x in f_list if x[:20]=='nep_wb_ssp585_moave_'] # keep only ocean model files in list
year_list = [x[20:24] for x in f_list]
unique_year_list = list(set(year_list))
month_list = [x[25:27] for x in f_list]
unique_month_list = list(set(month_list))
nmonths = len(unique_month_list)

gridfn = dir0+'roms_grd_nep.nc'
dg = xr.load_dataset(gridfn)

sf_path = home+'data/GOA shapefile/iho/iho.shp'
goa_geom = geopandas.read_file(sf_path)
poly = goa_geom.geometry[0]
lons, lats = poly.exterior.coords.xy
newlons = [lon%360 for lon in lons]
newpoly = Polygon(zip(newlons,lats))

lon = dg['lon_rho'][:]
lat = dg['lat_rho'][:]
a = np.array([Point(x, y) for x, y in zip(lon.values.flatten(), lat.values.flatten())], dtype=object)
goa_mask = np.array([newpoly.contains(point) for point in a])
goa_mask = goa_mask.reshape(lon.shape)

#initialize some arrays, lists and dataframes
monthly_goa_mean_SST = []
dt_list = []
goa_mean_first_8deg_month = np.zeros((len(unique_year_list)))
goa_mean_first_10deg_month = np.zeros((len(unique_year_list)))
first_8deg_month = np.nan*np.zeros((len(unique_year_list),lon.shape[0],lon.shape[1]))
first_10deg_month = np.nan*np.zeros((len(unique_year_list),lon.shape[0],lon.shape[1]))

count=0
for year in unique_year_list:
    print(f'Working on year {year:}')
    ds = xr.open_mfdataset(data_dir+f'nep_wb_ssp585_moave_{year:}_*.nc')
    SST = ds['temp'].values[:,-1,:,:]
    SST_ma = ma.masked_where(~goa_mask,ds.temp.values[t,-1,:,:])
    
    # np.argmax(condition,axis=x) works to identify the first argument meeting the condition along the axis of interest (axis=0, in this case, time)
    # dividing by np.any marks any locations where the condition did not occur as np.nan's
    first8degmo = np.argmax(SST>8,axis=0)/np.any(SST>8,axis=0)
    first8degmo = ma.masked_where(~(goa_mask&~np.isnan(first10degmo)),first8degmo)
    first_8deg_month[count,:,:] = first8degmo
    goa_mean_first_8deg_month[count] = first8degmo.mean()
    
    # repeat for 10 deg to compare
    first10degmo = np.argmax(SST>10,axis=0)/np.any(SST>10,axis=0)
    first10degmo = ma.masked_where(~(goa_mask&~np.isnan(first10degmo)),first10degmo)
    first_10deg_month[count,:,:] = first10degmo
    goa_mean_first_10deg_month[count] = first10degmo.mean()
    
    #now calculate a few things for each month
    for t in range(nmonths):
        dt = ds['ocean_time'].values[t]
        dt_list.append(dt)
        monthly_goa_mean_SST.append(SST_ma.mean())
    
    
    ds.close()
    count+=1

print('All done!')
print('Packing data for output')
df_monthly = pd.DataFrame({'Monthly GOA mean SST':monthly_goa_mean_SST,'Datetime':dt_list})
df_monthly.to_csv(home+'data/nep_wb_ssp585_Monthly_GOA_Mean_SST.csv')

D = {'first_8deg_month':first_8deg_month,'goa_mean_first_8deg_month':goa_mean_first_8deg_month,'first_10deg_month':first_10deg_month,'goa_mean_first_10deg_month':goa_mean_first_10deg_month,\
'year_list':unique_year_list,'goa_mask':goa_mask,'lon_rho':lon,'lat_rho':lat,'mask_rho':dg['mask_rho'].values[:],'h':dg['h'].values[:]}
    
outfn = home+'data/nep_wb_ssp585_first_xdeg_month_maps.p'
pickle.dump(D,open(outfn,'wb'))
dg.close()
print('All done!')
print(':)')