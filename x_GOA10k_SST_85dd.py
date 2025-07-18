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

which_SSP = 585

match which_SSP:
    case 126:
        dir0 = '/ahr0/emilyln/goa-output/nep_klone/h126wb/'
        print(f'Loading multifile dataset for SSP {which_SSP}...')
        ds = xr.open_mfdataset(dir0+'h126wb03_avg_*.nc') # this step takes a few minutes, as i/o always does
        print('Done loading!')
    case 585:
        dir0 = '/ahr0/hermann/goa-output/nep_klone/h585wb/'
        print(f'Loading multifile dataset for SSP {which_SSP}...')
        ds = xr.open_mfdataset(dir0+'h585wb0[4-5]_avg_*.nc') # this step takes a few minutes, as i/o always does
        print('Done loading!')    

nt,nz,ny,nx = ds['temp'].shape
years = np.unique(ds['ocean_time.year'].values)
nyears = len(years)

dd85 = np.zeros((nyears,ny,nx)) 

testing=False

if testing:
    nyears=3
    print('Testing turned on')
    
for i in range(nyears):
    print(f'Working on year {years[i]}')

    ds0 = ds.where(ds['ocean_time.year']==years[i],drop=True)[['ocean_time','temp']]

    surftemp = ds0['temp'][:,-1,:,:].values
    surftemp_cs = np.cumsum(surftemp,axis=0)
    dd85[i,:,:] = np.argmax(surftemp_cs>85,axis=0)/np.any(surftemp_cs>85,axis=0)

outfn = home +f'h{which_SSP}wb_2015-2100_85dd.p'
D = {'dd85':dd85,'years':years}
pickle.dump(D,open(outfn,'wb'))

print(f'Saved to {outfn}')

ds.close()