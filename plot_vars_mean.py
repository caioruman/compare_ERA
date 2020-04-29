import numpy as np
#import pandas as pd
import sys
import calendar

import numpy.ma as ma

from glob import glob
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib as mpl # in python
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans
from netCDF4 import Dataset

from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon


def plotMaps_pcolormesh(data, figName, values, mapa, lons2d, lats2d, var, cbar_l):
    '''
    fnames: List of filenames. Usually 2 (clim mean and future projection)
    varnames: list of variables to plot
    titles: list of the titles
    '''

    # GPCP
    fig = plt.figure(1, figsize=(14, 22), frameon=False, dpi=150)

    bn = BoundaryNorm(values, ncolors=len(values) - 1)

    #b = Basemap(projection='npstere',boundinglat=50,lon_0=-90,resolution='l', round=True)
    b = Basemap(llcrnrlon=-130.,llcrnrlat=40.,urcrnrlon=-7.,urcrnrlat=62., 
            resolution='l', projection='lcc', lat_0 = 50., lon_0 = -105)
    x, y = b(lons2d, lats2d)

    img = b.pcolormesh(x, y, data, cmap=mapa, vmin=values[0], vmax=values[-1], norm=bn)
#    img = b.contourf(x, y, data, cmap=mapa, norm=bn, levels=values, extend='both')

    b.drawcoastlines(zorder=1)
    cbar = b.colorbar(img, pad=0.75, ticks=values)
    cbar.ax.tick_params(labelsize=20)
    #cbar.ax.set_yticklabels(values)
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor('black')
    cbar.ax.set_title('{0}'.format(cbar_l), fontsize=24)

    b.drawcountries(linewidth=0.5, zorder=1)
    b.drawstates(linewidth=0.5, zorder=1)


    #parallels = np.arange(0.,81,10.)
    # labels = [left,right,top,bottom]
    #b.drawparallels(parallels,labels=[True,True,True,True], fontsize=16)
    meridians = np.arange(0.,351.,45.)
    b.drawmeridians(meridians,labels=[True,True,True,True], fontsize=16)   


    plt.subplots_adjust(top=0.75, bottom=0.25)


    plt.savefig('{0}.png'.format(figName), pad_inches=0.0, bbox_inches='tight')
    plt.close()

datai = 1981
dataf = 1990

# simulation
#exp = "PanArctic_0.5d_CanHisto_NOCTEM_RUN"
exp = "cPanCan_011deg_675x540_SPN_ERA5_80lvl"

folder_era = "/pixel/project01/shared/Data/ERA5/MonthlyMeans/MultiYearMeans"
folder_gem = "/pixel/project01/cruman/Data/{0}/MultiYearMeans".format(exp)
folder_daymet = "/pixel/project01/shared/Data/Daymet_V3_Monthly_Climatology/data/MultiYearMeans"

# Sounding Data
#sounding_file = "/home/cruman/project/cruman/Scripts/soundings/inv_list_DJF.dat"

period = ["DJF", "JJA", 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Nov', 'Dec']
period = ["DJF", "JJA", "SON", "MAM"] #, "JFM", "JAS"]
#period = ["DJF", "JJA"]
varlist = [('t2m', 'TT')]
#height = [925, 900, 850]

from matplotlib.colors import  ListedColormap
# Open the monthly files

for per in period:

    #read file
    
    file_era = "{0}/ERA5-monthly_{1}_{2}-{3}_PanCanada.nc".format(folder_era, per, datai, dataf)
    file_gem = "{0}/dm{1}_{2}_{3}-{4}_multimean".format(folder_gem, exp, per, datai, dataf)
    file_gz = "{0}/dm{1}_{2}_{3}-{4}_GZ_multimean".format(folder_gem, exp, per, datai, dataf)
    file_gem_pm = "{0}/pm{1}_{2}_{3}-{4}_multimean".format(folder_gem, exp, per, datai, dataf)

    file_daymet_tmax = "{0}/Daymet-tmax-monthly_{1}_{2}-{3}_PanCanada.nc".format(folder_daymet, per, datai, dataf)
    file_daymet_tmin = "{0}/Daymet-tmin-monthly_{1}_{2}-{3}_PanCanada.nc".format(folder_daymet, per, datai, dataf)
    print(file_gem)   
    print(file_era)
    print(file_daymet_tmin)
    print(file_gem_pm) 
    arq_era = Dataset(file_era, 'r')
    arq_gem = RPN(file_gem)
    arq_gz = RPN(file_gz)
    arq_pm = RPN(file_gem_pm)

    arq_daymet_tmax = Dataset(file_daymet_tmax, 'r')
    arq_daymet_tmin = Dataset(file_daymet_tmin, 'r')

    #for var in varlist:

    #data_era = np.transpose(np.squeeze(arq_era.variables['msl'][:]))/100 #- 273.15
    data_era = np.transpose(np.squeeze(arq_era.variables['t2m'][:]))- 273.15

    data_tmin = np.transpose(np.squeeze(arq_daymet_tmin['tmin'][:]))
    data_tmax = np.transpose(np.squeeze(arq_daymet_tmax['tmax'][:]))

    data_daymet = (data_tmin + data_tmax)/2

    data_gem = np.squeeze(arq_gem.variables['TT'][:])[-1,:,:]
    data_gem_tmax = np.squeeze(arq_pm.variables['T9'][:]) - 273.15
    data_gem_tmin = np.squeeze(arq_pm.variables['T5'][:]) - 273.15

    data_gz = np.squeeze(arq_gz.variables['GZ'][:])

    print(per)
    print("gem: {0}".format(data_gem[334, 235]))
    print("gem tmax: {0}".format(data_gem_tmax[334, 235]))
    print("gem tmin: {0}".format(data_gem_tmin[334, 235]))
    print("gem /2: {0}".format((data_gem_tmin[334, 235]+data_gem_tmax[334, 235])/2))
    print("daymet tmax: {0}".format(data_tmax[334, 235]))
    print("daymet tmin: {0}".format(data_tmin[334, 235]))
    print("daymet mean: {0}".format(data_daymet[334, 235]))
    #continue
    #sys.exit()
 

    #for i in range(0,data_gem.shape[0]):
    #    print(i)
    #    print(np.mean(data_gem[i,:,:]), np.mean(data_gz[i,:,:]))

    lon = np.squeeze(arq_era.variables["lon"][:])
    lat = np.squeeze(arq_era.variables["lat"][:])

    lons2d, lats2d = arq_gem.get_longitudes_and_latitudes_for_the_last_read_rec()

    #figName = "{0}_testeSummer".format(var)

    #
    colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'][::-1]
    cmap = mpl.colors.ListedColormap(colors)

    #if var == "DZ" or var == "ZBAS":
    #    values = np.arange(0,1001,100)
    #    colors = ['#ffffff', '#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
    #else:                
    values = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
#    values = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
    
    #data = data*(-1)
    cbar_l = "\N{DEGREE SIGN}C"                                
    
    data = data_tmax - data_gem_tmax
    figName = "TT_Daymet_{0}_{1}_tmax".format(exp, per)

    plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d, 'TT', cbar_l)

    data = data_tmin - data_gem_tmin
    figName = "TT_Daymet_{0}_{1}_tmin".format(exp, per)

    plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d, 'TT', cbar_l)

    #data = data_daymet - data_era
    #figName = "TT_Daymet-ERA_{0}_{1}".format(exp, per)

    #plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d, 'TT', cbar_l)

    arq_era.close()
    arq_gem.close()
    arq_gz.close()
    arq_daymet_tmax.close()
    arq_daymet_tmin.close()
