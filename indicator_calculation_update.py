# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:51:29 2023

@author: hermawan
"""

# read all excel

path = "D:/Rexus/mgrowa_results/xlsresults/"

import pandas as pd
import glob
import os
import numpy as np
import datetime as datetime

# Search for all CSV files starting with "3"
files = glob.glob(path + "*.csv")

ky = pd.read_csv("D:/Rexus/data/"+'ky.csv', delimiter=';')

# Open each CSV file as a pandas dataframe and store it in a list
dfs = pd.DataFrame()  
for file in files:
    df = pd.read_csv(file)
    print (file)

    df = df.rename(columns={df.columns[0]: 'subbasin'})
    df['filename'] = df['filename'].str.replace('et0', 'eto')
    df[['parameter', 'year', 'month']] = df['filename'].str.extract(r'^(?P<prefix>[a-zA-Z]+)(?P<year>\d{4})(?P<month>\d{2})\..*$')
    df['parameter'] = df['parameter'].astype(str)
    
    df['category'] = df['category'].astype(int)
    df['category'] = df['category'].astype(str)
    
    df['parameter'] = df.apply(lambda row: row['parameter'] + "_" + row['category'] if row['category'] != str(0) else row['parameter'], axis=1)
    df =df.drop(columns=['filename'])
    df_pivot = pd.pivot_table(df, values='values', index=['year', 'month','subbasin', 'aggregation'], columns=['parameter'], aggfunc='first').reset_index()
    df_pivot.rename(columns=lambda x: x.replace('_34', ''), inplace=True)

    #clean up dataframe
    num_cols = df_pivot.select_dtypes(include='number').columns
    df_pivot[num_cols] = df_pivot[num_cols].apply(lambda x: [0 if val < 0.01 else val for val in x])
    df_pivot[num_cols] = df_pivot[num_cols].apply(lambda x: [0 if val > 100000 else val for val in x])
    df_pivot ['year'] = pd.to_numeric(df_pivot ['year'], errors='coerce')
    df_pivot ['month'] = pd.to_numeric(df_pivot ['month'], errors='coerce')
    
    #AGGREGATE SEASON / YEAR==============================================================================================================
    df_pivot_aggs = pd.DataFrame ()
    
    ts = 4
    te = 9
    params = ["p", "qrn", "q", "mi", "mia", "eto", "ndswd"] #also et0
    #month all = 1000, month dry = 2000 month wet = 3000
    for param in params:
        df_pivot_annual = df_pivot[df_pivot['aggregation'] == 'mean'].groupby(['subbasin', 'aggregation', 'year'])[param].sum().reset_index()
        df_pivot_annual = df_pivot_annual.rename(columns={param: param+"_annual"})
        #dry period
        dry_condition = (df_pivot['aggregation'] == 'mean') & (df_pivot['month'].between(ts, te)) #include ts and te
        df_pivot_dry = df_pivot.loc[dry_condition].groupby(['subbasin', 'aggregation', 'year'])[param].sum().reset_index()
        df_pivot_dry = df_pivot_dry.rename(columns={param: param+"_dry"})
        #wet period
        wet_condition = ((df_pivot['aggregation'] == 'mean') & ((df_pivot['month'] < ts) | (df_pivot['month'] > te)))
        df_pivot_wet = df_pivot.loc[wet_condition].groupby(['subbasin', 'aggregation', 'year'])[param].sum().reset_index()
        df_pivot_wet = df_pivot_wet.rename(columns={param: param+"_wet"})
        
        df_pivot_merge = pd.merge(df_pivot_annual, df_pivot_dry, on=['subbasin', 'aggregation', 'year'], how='outer')
        df_pivot_merge = pd.merge(df_pivot_merge, df_pivot_wet, on=['subbasin', 'aggregation', 'year'], how='outer')
        
        if df_pivot_aggs.empty:
            df_pivot_aggs = df_pivot_merge
        else:
            df_pivot_aggs = pd.merge(df_pivot_aggs, df_pivot_merge, on=['subbasin', 'aggregation', 'year'], how='outer')
    
    
    #calculate area, area of month 7 ==== area whole year (July all crops are cultivated
    #calculate aggregation per indicator: season wet/dry or cultivation period. this aggregation should be stored as month annual, dry, wet
    for col_name in df_pivot.columns:
    # Check if "eta" or "etc" is in the column name
        if "eta" in col_name or "etc" in col_name:
            #get only crop cultivation period
            crop_id = float(col_name.split("_")[1]) 
            crop = ky[ky['crop_id'] == crop_id]
            # Get the values of lower_equal and greater_equal for the desired crop_id
            ts = crop['lower_equal'].values[0]
            te = crop['greater_equal'].values[0]
            
            #area
            if "eta" in col_name:
                area_condition = ((df_pivot['aggregation'] == 'count') & (df_pivot['month'] == 7))
                df_pivot_area = df_pivot.loc[area_condition].groupby(['subbasin', 'aggregation', 'year'])[col_name].mean().reset_index() 
                df_pivot_area = df_pivot_area.rename(columns={col_name: col_name+"_annualarea"})
                df_pivot_aggs = pd.merge(df_pivot_aggs, df_pivot_area, on=['subbasin', 'aggregation', 'year'], how='outer')
                
            if crop_id == 1000: #winter wheat
                wet_condition = ((df_pivot['aggregation'] == 'mean') & ((df_pivot['month'] <= ts) | (df_pivot['month'] >= te)))
                df_pivot_culc = df_pivot.loc[wet_condition].groupby(['subbasin', 'aggregation', 'year'])[col_name].sum().reset_index() 
            else:
                dry_condition = (df_pivot['aggregation'] == 'mean') & (df_pivot['month'].between(ts, te)) 
                df_pivot_culc = df_pivot.loc[dry_condition].groupby(['subbasin', 'aggregation', 'year'])[col_name].sum().reset_index()   
            df_pivot_culc = df_pivot_culc.rename(columns={col_name: col_name+"_annual"})
            
            df_pivot_aggs = pd.merge(df_pivot_aggs, df_pivot_culc, on=['subbasin', 'aggregation', 'year'], how='outer')
    
    for col_name in df_pivot.columns:
        if "ndswd" in col_name:
            area_condition = ((df_pivot['aggregation'] == 'count') & (df_pivot['month'] == 7))
            df_pivot_area = df_pivot.loc[area_condition].groupby(['subbasin', 'aggregation', 'year'])[col_name].mean().reset_index() 
            df_pivot_area = df_pivot_area.rename(columns={col_name: col_name+"_annualarea"})
            df_pivot_aggs = pd.merge(df_pivot_aggs, df_pivot_area, on=['subbasin', 'aggregation', 'year'], how='outer')
        
    # split the season as "months"
    df_pivot_aggs_melted = pd.melt(df_pivot_aggs, id_vars=['subbasin', 'aggregation', 'year'], var_name='variable', value_name='value')
    df_pivot_aggs_melted = df_pivot_aggs_melted.dropna(subset=['value'])

    df_pivot_aggs_melted[['parameter', 'month']] = df_pivot_aggs_melted['variable'].str.rsplit('_', n=1, expand=True)
    df_pivot_aggs = df_pivot_aggs_melted.pivot_table(index=['subbasin', 'aggregation', 'year', 'month'], columns='parameter', values='value').reset_index()
    
    aggregation_mapping = {
    'annual': 1000,
    'annualarea': 1000,
    'dry': 2000,
    'wet': 3000}
    
    df_pivot_aggs['month'] = df_pivot_aggs['month'].map(aggregation_mapping).fillna(df_pivot_aggs['month'])
    df_pivot = df_pivot.append (df_pivot_aggs)
    
    #combine indicator ==========================
    #1 Groundwater Exploitation Index (mi/qrn)
    df_pivot['indicator1'] = df_pivot.apply(lambda row: 0 if np.isnan(row['mi']) or row['qrn'] == 0 else row['mi'] / row['qrn'], axis=1)
    df_pivot['indicator1'] = np.where(df_pivot['aggregation'] == 'mean', df_pivot['indicator1'], np.nan)  
    
    #2 Percolation into groundwater aquifers (qrn/p)
    df_pivot['indicator2'] = df_pivot.apply(lambda row: 0 if row['p'] == 0 else row['qrn'] / row['p'], axis=1)*100
    df_pivot['indicator2'] = np.where(df_pivot['aggregation'] == 'mean', df_pivot['indicator2'], np.nan)
    
    #3 Irrigation water needs (summer) mia
    df_pivot['indicator3'] = df_pivot['mia']
    df_pivot['indicator3'] = np.where(df_pivot['aggregation'] == 'mean', df_pivot['indicator3'], np.nan)
    
    #4 Direct runoff q
    df_pivot['indicator4'] = df_pivot['q']
    df_pivot['indicator4'] = np.where(df_pivot['aggregation'] == 'mean', df_pivot['indicator4'], np.nan)
    
    #5 Drought Index (Rainfall/Evapotranspiration) (p/et0)
    df_pivot['indicator5'] = df_pivot.apply(lambda row: 0 if row['eto'] == 0 else row['p'] / row['eto'], axis=1)
    df_pivot['indicator5'] = np.where(df_pivot['aggregation'] == 'mean', df_pivot['indicator5'], np.nan)  
    
    #crop production indicator 6-9
    #act_yield = pot_yield * (-ky+1+ky*(eta/etc)) #only for annual df_pivot ['month'] == '1000' and df_pivot['aggregation'] == 'mean'
    indicator_crop = {
    'indicator6': 8000,  # 8000        Forage
    'indicator7': 3000,  #3000         Maize
    'indicator8': 12000, #12000        Cotton
    'indicator9': 1000}  #1000  Winter wheat
    ky
    for ind, crop_id in indicator_crop.items(): #0.1 because results in kg/ strema!
        crop        = ky[ky['crop_id'] == crop_id]
        ky_coeff    = crop['ky'].values[0]
        pot_yield   = crop['potential_yield'].values[0]
        df_pivot[ind] = 0.1* pot_yield * (-ky_coeff+1+ky_coeff*(df_pivot['eta_'+str(crop_id)]/df_pivot['etc_'+str(crop_id)])) #eta
        df_pivot[ind] = np.where((df_pivot['aggregation'] == 'mean') & (df_pivot['month'] == 1000), df_pivot[ind], np.nan)

    #count areas in hectares
    df_pivot['indicator3000'] = df_pivot['mia'] #Area	Irrigation
    df_pivot['indicator3000'] = np.where((df_pivot['aggregation'] == 'count') & (df_pivot['month'] == 1000), df_pivot['mia'], np.nan)
    
    df_pivot['indicator6000'] = df_pivot['eta_8000'] #8000        Forage
    df_pivot['indicator6000'] = np.where((df_pivot['aggregation'] == 'count') & (df_pivot['month'] == 1000), df_pivot['eta_8000'], np.nan)

    df_pivot['indicator7000'] = df_pivot['eta_3000'] #3000         Maize
    df_pivot['indicator7000'] = np.where((df_pivot['aggregation'] == 'count') & (df_pivot['month'] == 1000), df_pivot['eta_3000'], np.nan)    

    df_pivot['indicator8000'] = df_pivot['eta_12000'] #12000        Cotton
    df_pivot['indicator8000'] = np.where((df_pivot['aggregation'] == 'count') & (df_pivot['month'] == 1000), df_pivot['eta_12000'], np.nan)

    df_pivot['indicator9000'] = df_pivot['eta_1000'] #1000  Winter wheat
    df_pivot['indicator9000'] = np.where((df_pivot['aggregation'] == 'count') & (df_pivot['month'] == 1000), df_pivot['eta_1000'], np.nan)
    
    df_pivot['indicator12000'] = df_pivot['ndswd'] #Area	Forage
    df_pivot['indicator12000'] = np.where((df_pivot['aggregation'] == 'count') & (df_pivot['month'] == 1000), df_pivot['ndswd'], np.nan)
    
    #4 Soil moisture
    df_pivot['indicator12'] = df_pivot['ndswd']
    df_pivot['indicator12'] = np.where(df_pivot['aggregation'] == 'mean', df_pivot['indicator12'], np.nan)
    
    name, ext = os.path.splitext(os.path.basename(file))
    df_pivot['subcase'] = name
    
    dfs = dfs.append(df_pivot)
    
#make average for all subcase======================================================
dfs['case'] = dfs['subcase'].apply(lambda x: x[0])
dfs = dfs.reset_index().drop(columns='subcase')
dfs_grouped = dfs.groupby(['year', 'month', 'subbasin', 'case', 'aggregation']).agg(lambda x: x.mean(skipna=True))
dfs_grouped.to_csv (path+"raw_results.csv")

#remove some years====================================================================

dfs_grouped = dfs_grouped.drop(columns=[col for col in dfs_grouped.columns if 'indicator' not in col])
dfs_grouped.columns = [col.replace('indicator', '') if 'indicator' in col else col for col in dfs_grouped.columns]
dfs_grouped= dfs_grouped.reset_index().drop(columns='aggregation')

#tranform to database formatting
col_indicator = [col for col in dfs_grouped.columns if pd.Series(str(col)).str.contains('[1-9]').any()]
melted_dfs = pd.melt(dfs_grouped, id_vars=['year', 'month', 'subbasin', 'case'], value_vars=col_indicator, var_name='indicator_id', value_name='value')

melted_dfs['year'] = melted_dfs['year'].astype(float)
melted_dfs['value'] = melted_dfs['value'].astype(float)
melted_dfs['period'] = np.nan
melted_dfs.loc[(melted_dfs['year'] >= 1971) & (melted_dfs['year'] <= 2000), 'period'] = '1971 - 2000'
melted_dfs.loc[(melted_dfs['year'] >= 2011) & (melted_dfs['year'] <= 2040), 'period'] = '2011 - 2040'
melted_dfs.loc[(melted_dfs['year'] >= 2041) & (melted_dfs['year'] <= 2070), 'period'] = '2041 - 2070'
melted_dfs.loc[(melted_dfs['year'] >= 2071) & (melted_dfs['year'] <= 2100), 'period'] = '2071 - 2100'

melted_dfs = melted_dfs.dropna(subset=['period']).dropna(subset=['value'])
filtered_df = melted_dfs[melted_dfs['case'] == '1']

# Duplicate the filtered DataFrame for each period
periods     = ['2011 - 2040', '2041 - 2070', '2071 - 2100']
final_df    = pd.concat([filtered_df.assign(period=period) for period in periods])
melted_dfs  = pd.concat([melted_dfs, final_df])

melted_dfs = melted_dfs.groupby(['indicator_id', 'subbasin', 'case', 'month', "period"]).mean()
melted_dfs = melted_dfs.reset_index()
melted_dfs['time'] = melted_dfs['month'] #pd.to_datetime(melted_dfs[['month']].assign(day=1, year=2000))

melted_dfs = melted_dfs.drop(['month'], axis=1)
melted_dfs = melted_dfs.loc[:, ['time', 'subbasin', 'case', 'indicator_id', 'value', 'period']].rename(columns={'subbasin': 'location'})

melted_dfs.to_csv (path+"results.csv")