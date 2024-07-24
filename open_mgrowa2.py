# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 08:51:21 2022
extract mGROWA for metamodel
REXUS h2020 Pineos pilot
@author: hermawan
"""

#--------------
import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from osgeo import gdal, osr
import os
import glob
import warnings
from rasterstats import zonal_stats

#get name of shp files as index name
#remove it for debugging
warnings.filterwarnings("ignore")

# raster = "D:/Rexus/data/out.tif"
# asc = "D:/Rexus/data/r203 p19712000y.asc"

#define path
path_src = "D:/Rexus/data/"
shapefile = gpd.read_file(path_src+"pineosraster.shp")
landuse   = rasterio.open(path_src+'clc18_osde21_mgrowa_map.asc').read(1)
crop_type = rasterio.open(path_src+'irrfunct.asc').read(1)

mGROWA = pd.DataFrame()
#aggregation level
data_agg = pd.read_csv (path_src+"mgrowa_metadata.csv")

#create ETc====================================================================
if not os.path.exists(path_src+'etc'):
    os.makedirs(path_src+'etc')

asc_files = glob.glob(path_src+'et0/'+'*.asc')

# Function to read ESRI ASCII raster file
def read_ascii_raster(filename):
    with open(filename, 'r') as file:
        header = {line.split()[0]: float(line.split()[1]) for line in [next(file) for _ in range(6)]}
        data = np.loadtxt(file, skiprows=0)
    return header, data

# Function to write ESRI ASCII raster file
def write_ascii_raster(filename, header, data):
    with open(filename, 'w') as file:
        for key, value in header.items():
            file.write(f"{key} {value}\n")
        np.savetxt(file, data, fmt='%f')

# Load crop function raster
irrfunct_header, irrfunct_data = read_ascii_raster(path_src+'irrfunct.asc')
# Load crop coefficients
kc_df = pd.read_csv(path_src+"Kc.csv")

asc_files = glob.glob(path_src+'et0/'+'*.asc')
for file_name in asc_files:
    asc_index = file_name.find('.asc')
    month_str = file_name[asc_index - 2: asc_index]  # Extracting the two characters before ".asc"
    if month_str.startswith('0'):  # Removing leading zeros
        month_str = month_str[1:]
    month = int(month_str)

    # Read the current month's ET0 raster
    et0_header, et0_data = read_ascii_raster(file_name)

    # Initialize an empty array for the result
    result_data = np.full_like(et0_data, -9999.0)  # Use no data value for initialization

    for crop_type in np.unique(irrfunct_data):
        if crop_type == -9999.0:
            continue  # Skip no data values
        if crop_type not in kc_df['Code'].values:
            continue  # Skip crop types not in Kc.csv
        kc_value = kc_df.loc[kc_df['Code'] == crop_type, str(month)].values[0]
        mask = irrfunct_data == crop_type
        result_data[mask] = et0_data[mask] * kc_value

    # Write the result to a new ascii raster file
    
    result_filename = file_name.replace('et0', 'etc')
    write_ascii_raster(os.path.join(path_src+'etc', result_filename), et0_header, result_data)
#create mia====================================================================
# create the mia folder if it doesn't exist
if not os.path.exists(path_src+'mia'):
    os.makedirs(path_src+'mia')

asc_files = glob.glob(path_src+'mi/'+'*.asc')

for asc_id in range(len(asc_files)):
    if 'mi' in asc_files[asc_id]:
        with rasterio.open(asc_files[asc_id], 'r') as src:
            meta = src.meta.copy()
            mi = src.read(1)
        mia = mi.astype('float32')
        mia[mi == 0] = -9999
        # get the original filename without the extension
        filename = os.path.splitext(os.path.basename(asc_files[asc_id]))[0]
        # create the new filename by replacing 'mi' with 'mia'
        new_filename = filename.replace('mi', 'mia')
        # create the full path for the new file in the mia folder
        new_path = os.path.join(path_src+'mia', new_filename + '.asc')
        with rasterio.open(new_path, 'w', **meta) as dst:
            dst.write(mia, 1)

#read all files===============================================================        
#read all available *asc file in the folder, reference it, and convert it to .tiff file
#list all asc files in a folder - parameters      
for i in range(len(data_agg)):
    path = path_src+data_agg.loc[i,"folder_name"]+"/"
    agg  = data_agg.loc[i,"aggregation"]
    asc_files = glob.glob(path+'*.asc')

    #loop over the available .asc files
    for asc_id in range (len (asc_files)) :
        asc = asc_files[asc_id]
        print ("converting "+asc_files[asc_id]+" to Geotiff files.....")    
        drv = gdal.GetDriverByName('GTiff')
        ds_in = gdal.Open(asc)
        gtiff_name = asc.split("\\")[-1].replace (".asc",".tif")
        #convert .asc files into .tif files
        ds_out = drv.CreateCopy(path+gtiff_name, ds_in)
        srs = osr.SpatialReference()
        #add greek reference
        srs.ImportFromEPSG(2100)
        ds_out.SetProjection(srs.ExportToWkt())
        ds_in = None
        ds_out = None
        
        #overlay with sub-basin polygon to get aggregated value, note they need to have the same reference system
        
        with rasterio.open(path+gtiff_name) as src:
            affine = src.transform
            array = src.read(1)
            ndval = -9999
            array[array==ndval] = np.nan
            
            df_zonal_stats = pd.DataFrame()
            
            #to do per crops, ETa and ETc
            if "eta" in gtiff_name or "etc" in gtiff_name:
                categories = [3000.0, 12000.0, 1000.0, 8000.0]
                for cat in categories:
                    mask = (irrfunct_data == cat)
                    array_copy = array.copy()
                    array_copy[~mask] = np.nan
                    
                    #sampling trhough polygon
                    print ("calculating statistics for "+asc_files[asc_id]+".....")   
                    df_zonal_stat = pd.DataFrame(zonal_stats(shapefile, array_copy, affine=affine))
                    df_zonal_stat ["sum"] = df_zonal_stat ["mean"] * df_zonal_stat ["count"]
                    df_zonal_stat ["filename"] = gtiff_name
                    df_zonal_stat ["category"] = cat
                    
                    #get basin name based on index
                    subbasin_id = dict(zip(shapefile.index, shapefile['layer']))
                    df_zonal_stat.index = df_zonal_stat.index.map(subbasin_id)
                    
                    df_zonal_stats = df_zonal_stats.append(df_zonal_stat)
                    
            elif "ndswd" in gtiff_name:
                cat = [311, 312, 313, 321,322,323,324, 331,332,333,334,411, 421]
                mask = np.isin(landuse, cat)
                array[~mask] = np.nan
                
                #sampling trhough polygon
                print ("calculating statistics for "+asc_files[asc_id]+".....")   
                df_zonal_stat = pd.DataFrame(zonal_stats(shapefile, array, affine=affine))
                df_zonal_stat ["sum"] = df_zonal_stat ["mean"] * df_zonal_stat ["count"]
                df_zonal_stat ["filename"] = gtiff_name
                cat = 34
                df_zonal_stat ["category"] = cat
                
                #get basin name based on index
                subbasin_id = dict(zip(shapefile.index, shapefile['layer']))
                df_zonal_stat.index = df_zonal_stat.index.map(subbasin_id)
                
                df_zonal_stats = df_zonal_stats.append(df_zonal_stat)
                
            else:
                cat = 0
                array = array.copy()
                
                #sampling trhough polygon
                print ("calculating statistics for "+asc_files[asc_id]+".....")   
                df_zonal_stat = pd.DataFrame(zonal_stats(shapefile, array, affine=affine))
                df_zonal_stat ["sum"] = df_zonal_stat ["mean"] * df_zonal_stat ["count"]
                df_zonal_stat ["filename"] = gtiff_name
                df_zonal_stat ["category"] = cat
                
                #get basin name based on index
                subbasin_id = dict(zip(shapefile.index, shapefile['layer']))
                df_zonal_stat.index = df_zonal_stat.index.map(subbasin_id)
                df_zonal_stats = df_zonal_stats.append(df_zonal_stat)
        
        #extract only important information for the indicator
        df_results = df_zonal_stats[["filename", agg, "category"]]
        df_results ['aggregation'] = agg
        df_results = df_results.rename(columns={agg: "values"}, errors="raise")
    
        
        #combine database
        mGROWA = mGROWA.append(df_results)

mGROWA.to_csv(path_src+"mGROWA_results_case.csv")
