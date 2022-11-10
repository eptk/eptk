# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

"""
##################################
A collection of utility functions.
##################################
"""

# 1.
def missing_timestamp(df,site_id):
    """
    Parameters
    -----------
      df : pandas dataframe
      A pandas dataframe containing columns site_id and timestamp.

      site_id : int
      A positve integer corresponding to a particular site location.

    Returns
    ----------
      numpy.ndarray 
      An array containing the missing timestamps.

    Description
    ------------
    A method to check for missing timestamps from the start time till the end time for the site given by the site_id. (hourly basis)
    Sets the timestamp column as the index of the dataframe.
    """
    if 'timestamp' not in df.columns:
        raise ValueError("timestamp column not in the dataframe")

    if 'site_id' not in df.columns:
        raise ValueError("site id not in the dataframe")    

    
    df['timestamp'] = df['timestamp'].astype(str)
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(df['timestamp'].min(), time_format)
    end_date = datetime.strptime(df['timestamp'].max(), time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - timedelta(hours = x)).strftime(time_format) for x in range(total_hours)]
    site_hours = np.array(df[df['site_id'] == site_id]['timestamp'])
    
    return np.setdiff1d(hours_list, site_hours)




#2. reducing memory usage
def reduce_mem_usage(df, use_float16=False):
    """
    Parameters
    -------------
    df : pandas dataframe

    Description
    ---------------
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.  
    
    Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin      
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return 

#3. getting meter data according to the metertype:
def get_meter_df(df, meter = 0):
   
   """
   Parameters
   -----------

   df: dataframe containing meter data.

   meter: int
   an integer associated with a meter type. 

   Returns
   ---------
   A dataframe containing only the readings given by the selected metertype. 
   """
   
   
   if 'meter' not in df.columns:
        raise ValueError (" meter not in the dataframe") 

   return df[df['meter'] == meter]

#4. merging
"""
Supports the BDG2 and the ASHRAE datasets format.
     Assumptions
     ------------
     weather_df : site id and timestamp columns should be present
     meter_df   :  timestamp and building_id columns should be present
     meta_df    :    site_id  and building_id columns should be present
 
"""

def merge_data(meter_df, meta_df = None, weather_df = None):
 
    """
    Parameters
    ------------

    meter_df : pandas dataframe
    A dataframe containing meter readings.
    It should contain columns timestamp and building_id.

    meta_df : pandas dataframe
    A dataframe containing the building meta data.
    It should contain column "building_id" and "site_id".

    weather_df : pandas dataframe
    A dataframe containing the weather  data for all the sites.(optional,default=None)
    It should contain column "timestamp" and "site_id".

    Returns
    --------

    pandas dataframe: A merged dataset.
    The dataframe obtained from merging the 3 dataframes.
  
    Description
    ------------
    Combines the weather, meta and meter data into one. meta and meter dataframes are must, weather dataframe is optional.





    """
    if meter_df is None:
       raise ValueError("meter_df cannot be a Nonetype.")
    
    
    df = meter_df.copy()
    
    if meta_df is None:
       raise ValueError("meta_df cannot be a Nonetype.")


    df = df.merge(meta_df, on = 'building_id', how= 'left')
       
          
    if weather_df is not None:
       print("merging weather data")
       df = df.merge(weather_df, on = ['site_id', 'timestamp'], how='left')
    
    print("merged dataset created")
    return df 


#5. getting a slice of the data oject according to the index_list array.

def get_slice(X, index_list):
    """
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        

    index_list: An array of indices (integers)

    Returns
    ---------
    Slice of X according to the indices given in index_list
    
    Description
    -------------
    This method is used in cross validation to get the sliced data according to the indices (train/test). 
    The returned object is of the same type as of the input. The pandas dataframe dont allow slicing the same way as a numpy array. 

   """


    try:
         #if its a pandas dataframe
         X_ = X[X.index.isin(index_list)]
         return X_

    except AttributeError:
        # if its a numpy array
        try:
          X_ = X[index_list]
          return X_

        except TypeError:
           print("Input should be a numpy array or a pandas dataframe")





def absolute_quantity(timestamp, t_type = "month"):
  """
  Utility function used by the cross-validator.
  Parameters
  ------------
  timestamp: pandas.core.series.Series
  provide the timestamp series

  t_type: str
  time_bins, options- {"month", "week"}

  Returns
  -------------
  The value of total {t_type} ellapsed from the start time till now.
  : pandas.core.series.Series

  """
  if t_type == "month":
     x = "M"
  if t_type =="week":
     x = "W"
  start_time = min(timestamp)
  return ((timestamp - start_time)/np.timedelta64(1, f'{x}')).round()



