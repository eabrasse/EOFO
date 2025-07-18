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
import pickle
import matplotlib.dates as mdates

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
plasma12 = plt.get_cmap('plasma',12)

# load in data and grid
home = '/Users/elizabethbrasseale/projects/EOFO/'

datafn = home+'data/nep_wb_ssp585_first_xdeg_month_maps.p'
D = pickle.load(open(datafn,'rb'))

nt,ny,nx = D['first_8deg_month'].shape

# # initialize plot
fw,fh = efun.gen_plot_props()
fig0 = plt.figure(figsize=(fw*2.5,fh))
ax0 = fig0.add_subplot(1,2,1)
# ax0 = fig.gca()
years = list(map(int, D['year_list']))

ax0.plot(years,D['goa_mean_first_8deg_month'],linestyle='solid',color=tab10(0))
ax0.text(0.1,0.3,'GOA Mean\n'+r'First month > $8^{\circ}$ C'+'\nSSP 585',transform=ax0.transAxes)

ax1 = fig0.add_subplot(1,2,2)
ax1.plot(years,D['goa_mean_first_10deg_month'],linestyle='solid',color=tab10(0))

ax1.text(0.1,0.3,'GOA Mean\n'+r'First month > $10^{\circ}$ C'+'\nSSP 585',transform=ax1.transAxes)

for ax in ax0, ax1:
    # ax.xaxis.set_major_locator(mdates.YearLocator(5))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
    # ax.set_xticks(range(years[0]))
    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    ax.set_ylim(1.75,5.75)
    ax.set_yticks([2,3,4,5])
    ax.set_yticklabels(['Mar','Apr','May','Jun'])

fig0.subplots_adjust(wspace=0.2,left=0.1,right=0.98,bottom=0.15,top=0.98)
outfn = home+'plots/GOA first month/SST_firstmonth_trends_SSP585.png'
fig0.savefig(outfn)
plt.close(fig0)

first_8deg = {}
first_10deg = {}

goa_mask = np.broadcast_to(~D['goa_mask'],D['first_8deg_month'].shape)
first_8deg['ma'] = ma.masked_where(goa_mask,D['first_8deg_month'][:])
first_10deg['ma'] = ma.masked_where(goa_mask,D['first_10deg_month'][:])

yrbar = np.mean(years)
yrMinusyrbar = years-yrbar
deno = np.sum(yrMinusyrbar**2)
for x in first_8deg,first_10deg:
    xbar = x['ma'].mean(axis=0)
    xMinusxbar = x['ma']-xbar
    numer = np.zeros(x['ma'].shape)
    for i in range(len(years)):
        numer[i,:,:] = yrMinusyrbar[i]*xMinusxbar[i,:,:]
    numer = np.sum(numer,axis=0)
    x['slope'] = numer/deno
    x['intercept'] = xbar-x['slope']*yrbar

fig1 = plt.figure(figsize=(fw*1.75,fh*1.3))
ax2 = fig1.add_subplot(2,1,1)
p=ax2.pcolormesh(D['lon_rho'],D['lat_rho'],first_8deg['slope'],cmap=cmo.cm.balance_r,vmin=-0.06,vmax=0.06)
ax2.text(0.05,0.9,r'Change in first month > $8^{\circ}$ C'+'\n2100-2015',transform=ax2.transAxes,va='top',ha='left')

ax3 = fig1.add_subplot(2,1,2)

p=ax3.pcolormesh(D['lon_rho'],D['lat_rho'],first_10deg['slope'],cmap=cmo.cm.balance_r,vmin=-0.06,vmax=0.06)
ax3.text(0.05,0.9,r'Change in first month > $10^{\circ}$ C'+'\n2100-2015',transform=ax3.transAxes,va='top',ha='left')

cbaxes = inset_axes(ax3, width="4%", height="90%", loc='upper right',bbox_transform=ax3.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
cb = fig1.colorbar(p, cax=cbaxes,orientation='vertical')
cbaxes.set_ylabel('Month/Year')


for ax in ax2,ax3:
    ax.pcolormesh(D['lon_rho'],D['lat_rho'],D['mask_rho'],cmap=cmap_mask)
    ax.set_xlim(195,225)
    ax.set_ylim(54,62)
    efun.dar(ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

fig1.subplots_adjust(right=0.8)

outfn = home+'plots/GOA first month/SST_change_in_first_month_map_SSP585.png'

fig1.savefig(outfn)
plt.close(fig1)
    
# t=0
for t in range(len(years)):
    fig2 = plt.figure(figsize=(fw*1.75,fh*1.3))

    ax4 = fig2.add_subplot(2,1,1)
    p=ax4.pcolormesh(D['lon_rho'],D['lat_rho'],first_8deg['ma'][t,:],cmap=plasma12,vmin=0,vmax=12)
    ax4.text(0.05,0.9,r'First month > $8^{\circ}$ C',transform=ax4.transAxes,va='top',ha='left')


    ax5 = fig2.add_subplot(2,1,2)

    p=ax5.pcolormesh(D['lon_rho'],D['lat_rho'],first_10deg['ma'][t,:],cmap=plasma12,vmin=0,vmax=12)
    # cbaxes = inset_axes(ax5, width="4%", height="40%", loc='upper right',bbox_transform=ax5.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
    # cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
    # cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
    # cbaxes.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11])
    # cbaxes.set_yticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax5.text(0.05,0.9,r'First month > $10^{\circ}$ C',transform=ax5.transAxes,va='top',ha='left')

    cbaxes = inset_axes(ax4, width="4%", height="90%", loc='upper right',bbox_transform=ax4.transAxes,bbox_to_anchor=(0.1,-0.05,1,1))
    cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
    cb = fig2.colorbar(p, cax=cbaxes,orientation='vertical')
    cbaxes.set_yticks([i+0.5 for i in range(12)])
    cbaxes.set_yticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

    for ax in ax4,ax5:
        ax.pcolormesh(D['lon_rho'],D['lat_rho'],D['mask_rho'],cmap=cmap_mask)
        ax.set_xlim(195,225)
        ax.set_ylim(54,62)
        efun.dar(ax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.text(0.1,0.75,D['year_list'][t]+' SSP 585',transform=ax.transAxes,va='top',ha='left')

    fig2.subplots_adjust(right=0.85)
    
    outfn = home+'plots/GOA first month/SST_first_month_map_{}_SSP585.png'.format(D['year_list'][t])
    
    fig2.savefig(outfn)
    plt.close(fig2)

# plt.show(block=False)
# plt.pause(0.1)

#calculate slope & intercept
xbar = 
