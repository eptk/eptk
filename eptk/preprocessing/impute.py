# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..utils import missing_timestamp

"""

This module contains imputation methods, which are suitable for all the datasets. Dataset specific imputation is part
of the dataset loading class.

"""

def add_missing_timestamp(df):
    """
    Parameters
    ----------
    df : pandas dataframe
    A pandas dataframe containing the column timestamp.
    (optional column "site_id" if present, is used to fill out missing timestamps for each of the different sites.)

    Returns
    ---------
    pandas dataframe
    A pandas dataframe with added rows. All the feature values for the added rows is NaNs.

    Description
    -------------
    A method to add rows pertaining to the missing timestamps from the start and the end time.

    """

    remove_site_id = False
    if 'site_id' not in df.columns:
      print("No site_id column found in the dataframe. Creating one with all entries with default value 'x'.")
      df["site_id"] = "x"
      remove_site_id = True

    new_rows = []
    for site_id in df["site_id"].unique():  #add new rows for missing timestamp data
             new_rows = pd.DataFrame(missing_timestamp(df, site_id), columns=['timestamp'])
             new_rows['site_id'] = site_id
             df = pd.concat([df, new_rows])

    print (f"{len(new_rows)} rows added" )

    if remove_site_id:
        df.drop(columns=["site_id"], inplace=True)
        print("Column site_id which was not present originally is removed.")
    return df

def impute_weather(weather_df, weather_columns = []):
   """
   Parameters
   ----------

    weather_df : pandas dataframe
    A pandas dataframe containing the weather data.

    weather_columns : list
    Provide a list of weather columns to impute.
    The default column list is ['air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed','precipDepth6HR']
    The column list will be extended if the weather_column list is not empty (which is by default)

   Returns
   ---------
   weather_df: pandas dataframe
   The updated weather dataset with the features imputed.

   Description
   -----------
   A method to impute the weather data. The dataframe is grouped by ['site_id','day','month'] and the mean is taken as the filler value
   to fill the missing value for a site on that particular month and day of the month.  
   
   The features given by the list cols.   
   
   cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed','precipDepth6HR']

   These features are imputed, one by one, if found in the dataframe.  

   """ 
   
   weather_df = weather_df.copy()
   print("Before imputing percentage of missing values:")
   print( 100 * weather_df.isna().sum().sort_values(ascending = False) / len(weather_df))   
   
   #adding features for better indexing
   weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
   remove=[]  # we add some temporal features for indexing, if not already present needs to be removed at the end.
   if "day" not in weather_df.columns:
       weather_df["day"] = weather_df["timestamp"].dt.day
       remove.append("day")
   if "month" not in weather_df.columns:
       weather_df["month"] = weather_df["timestamp"].dt.month
       remove.append("month")
   if "site_id" not in weather_df:  # if not found every entry is assumed to be coming from 1 site alone.
       weather_df["site_id"] = "x"
       remove.append("site_id")

   weather_df = weather_df.set_index(['site_id','day','month'])

   # The columns 
   cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed','precipDepth6HR']
   cols = list(set(cols).union(set(weather_columns)))
   for i in cols:
      if i in list(weather_df.columns):
        filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])[i].mean(), columns = [i])
        # forward fill if for a specific day we don't get a mean value.
        filler = pd.DataFrame(filler.fillna(method = 'ffill'), columns = [i])
        weather_df.update(filler, overwrite = False)
        print(f"{i} values imputed")
   weather_df = weather_df.reset_index()
   weather_df.drop(columns = remove, inplace = True)
   print("After imputing percentage of missing values:")
   print(100 * weather_df.isna().sum().sort_values(ascending = False) / len(weather_df))
   return  weather_df