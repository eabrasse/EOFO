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
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import rasterio

# # from stack overflow
# def cartopy_example(raster, shapefile):
#     with rasterio.open(raster, 'r') as src:
#         raster_crs = src.crs
#         left, bottom, right, top = src.bounds
#         landsat = src.read()[0, :, :]
#         landsat = np.ma.masked_where(landsat <= 0,
#                                      landsat,
#                                      copy=True)
#         landsat = (landsat - np.min(landsat)) / (np.max(landsat) - np.min(landsat))
#
#     proj = ccrs.LambertConformal(central_latitude=40,
#                                  central_longitude=-110)
#
#     fig = plt.figure(figsize=(20, 16))
#     ax = plt.axes(projection=proj)
#     ax.set_extent([-110.8, -110.4, 45.3, 45.6], crs=ccrs.PlateCarree())
#
#     shape_feature = ShapelyFeature(Reader(shapefile).geometries(),
#                                    ccrs.PlateCarree(), edgecolor='blue')
#     ax.add_feature(shape_feature, facecolor='none')
#     ax.imshow(landsat, transform=ccrs.UTM(raster_crs['zone']),
#               cmap='inferno',
#               extent=(left, right, bottom, top))
#     plt.show(block=False)
#     plt.pause(0.1)
#
#
# feature_source = 'fields.shp'
# raster_source = 'surface_temperature_32612.tif'
#
# cartopy_example(raster_source, feature_source)



