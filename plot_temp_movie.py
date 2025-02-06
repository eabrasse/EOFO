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

data_fn = home+'data/B10K-K20P19_CORECFS_2020-2024_average_temp.nc'
grid_fn = home+'data/Bering10K_extended_grid.nc'
ds = nc.Dataset(data_fn)
dsg = nc.Dataset(grid_fn)

temp = ds['temp'][:]
temp = np.ma.filled(temp,np.nan)
# initialize variables to plot
surf_temp = {}
surf_temp['var'] = temp[:,-1,:,:]
surf_temp['label'] = 'Surface temperature'

bottom_temp = {}
bottom_temp['var'] = temp[:,0,:,:]
bottom_temp['label'] = 'Bottom temperature'

mean_temp = {}
mean_temp['var'] = np.nanmean(temp,axis=1)
mean_temp['label'] = 'Water column mean\ntemperature'

# through the variables into an iterable list
var_list = [surf_temp, bottom_temp, mean_temp]

# all will use same cmap and vlims, but I'm going to add these to the indiviual dicts anyway
# for future generalizeability
vmin = np.nanpercentile(temp,3,method='nearest')
vmax = np.nanpercentile(temp,97,method='nearest')
for var in var_list:
    var['vmin'] = vmin
    var['vmax'] = vmax
    var['norm'] = matplotlib.colors.Normalize(vmin=var['vmin'],vmax=var['vmax'])
    var['cmap'] = cmo.cm.thermal
    var['units'] = r'$^{\circ}$C'

# alias some grid variables for legibility
lonr = dsg['lon_rho'][:]
latr = dsg['lat_rho'][:]
h = dsg['h'][:]
maskr = dsg['mask_rho'][:]

# some helpful features for cartopy to map
xlim =[185,200]
ylim = [53,67.5]
# rect = mpath.Path([[xlim[0], ylim[0]],
#                    [xlim[1], ylim[0]],
#                    [xlim[1], ylim[1]],
#                    [xlim[0], ylim[1]],
#                    [xlim[0], ylim[0]],
#                    ]).interpolated(20)


# process time stamps into a human-readable format for labeling
ot = ds['ocean_time'][:]
dt_list = [pytz.utc.localize(datetime(1900,1,1)+timedelta(seconds=ott)) for ott in ot[:]]

for t in range(len(dt_list)):
    # t=0 #while testing

    # initialize plot
    fw,fh = efun.gen_plot_props()
    fig = plt.figure(figsize=(12,5))
    gs = GridSpec(1,len(var_list))

    # add a count while looping through variables
    var_count=0
    # nooooow loop!
    for var in var_list:
        # grab axis
        axll = fig.add_subplot(gs[var_count])#,projection=ccrs.AlbersEqualArea())

    
        # plot coastlines and bathymetry contours
        axll.pcolormesh(lonr,latr,maskr,cmap=cmap_mask,shading='nearest',zorder=5)
        axll.contour(lonr,latr,h,levels=np.arange(0,1000,50),colors='k',linewidths=0.5,zorder=6,linestyles=':')
    
        # now add the temperature heatmap
        p=axll.pcolormesh(lonr,latr,var['var'][t,:,:],shading='nearest',cmap=var['cmap'],norm=var['norm'])
    
        # because all these use same units, only plot colorbar once
        if var_count==0:
            cbaxes = inset_axes(axll, width="4%", height="40%", loc='upper right',bbox_transform=axll.transAxes,bbox_to_anchor=(0.05,-0.05,1,1))
            cbaxes.tick_params(axis='both',which='both',labelsize=8,size=2)
            cb = fig.colorbar(p, cax=cbaxes,orientation='vertical')
            cbaxes.set_ylabel(var['units'])
    
        # add axis labels
        axll.set_xlim(xlim)
        axll.set_ylim(ylim)
        # gl=axll.gridlines(draw_labels=False, x_inline=False, y_inline=False)
    
        # gl.top_labels    = False
        # gl.right_labels  = False
        # axll.set_frame_on(False)
        axll.set_xlabel('Longitude')
        axll.set_ylabel('Latitude')
        efun.dar(axll)
        axll.text(0.1,1.05,'{}) {}'.format(atoz[var_count],var['label']),transform=axll.transAxes,ha='left',fontsize=12,zorder=600,va='bottom')
        # axll.text(0.5,-.05,'Longitude',transform=axll.transAxes,ha='center',fontsize=12,zorder=600,va='top')
        # axll.text(-0.05,.5,'Latitude',transform=axll.transAxes,ha='right',fontsize=12,zorder=600,va='center')
    
    
        # include datetime only once
        if var_count==0:
            axll.text(0.05,0.1,dt_list[t].strftime("%m/%d/%Y"), transform=axll.transAxes, ha='left', zorder=55, fontsize=8, color='k', fontweight='bold', bbox=dict(facecolor='white',edgecolor='None'))
    
        var_count+=1

    # now tidy up and show the plot
    # fig.subplots_adjust(right=0.98,left=0.08,bottom=0.15,top = 0.9,wspace=0.2)
    fig.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)
    outfn = home+f'plots/B10k temp/figure_{t:0>4}.png'
    plt.savefig(outfn)
    plt.close()

# close netCDF datasets
ds.close()
dsg.close()