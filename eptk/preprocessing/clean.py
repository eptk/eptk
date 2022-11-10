# -*- coding: utf-8 -*-
import pandas as pd
"""
A module containing methods to clean the data. The data is in the form of a pandas dataframe.
"""

def remove_missing_meter_reading(df):
  """
    Parameters
    ----------
    df : pandas dataframe
    A pandas dataframe containing columns "meter_reading".
        
    Returns
    -------
    dataframe with the rows removed.
         
    Description
    ------------    
    A method for removing the rows with missing meter reading values. 


  """
  df = df.copy() #not altering the original dataframe
  remove_index = df[df['meter_reading'].isnull()].index.tolist()
  df.drop(remove_index, inplace=True)
 
  print(f" {len(remove_index)} Missing readings removed " )
  return df

def remove_readings_threshold(df, threshold , meter = None):
   """
    Parameters
    ----------
    df : pandas dataframe
    A pandas dataframe containing columns "meter_reading", "meter" (optional)
 
    threshold : float
    A threshold value for removing meter readings below it.

    meter :  str / int (optional)
    An integer/ string value for only considering the rows corresponding to the meter type give by the number/ string.
    If it is set to None, all of the meter readings are taken into consideration. 
    Any other integer value is only acceptable when the dataframe contains a "meter" column. 
    Refer https://www.kaggle.com/c/ashrae-energy-prediction/data for BDG2 and Ashrae dataset

    Returns
    -------
    updated dataframe.
         
    Description
    ------------    
    A method for removing the rows with meter reading value below the set threshold for the given meter type.


   """
   df = df.copy()
   if "meter" in df.columns and meter != None:
    temp = df[df['meter'] == meter]
   else:
    temp = df
   remove_index = temp[temp['meter_reading'] < threshold].index.tolist()
   df.drop(remove_index, inplace = True)
   del temp
   print (f"{len(remove_index)}  rows removed") 
   return df 
   

def remove_feature_above_missing_percentage(df, percentage):
  """
    Parameters
    ----------
    df : pandas dataframe
    A pandas dataframe.

    percentage: float
    A numeric value between 0 and 100. The columns with missing values above the set percetage are removed.
        
    Returns
    -------
    str : A string providing the list of columns removed from the input dataframe. 
         
    Description
    ------------    
    A method for removing the rows with missing meter reading values. 


  """
  df = df.copy() #not altering the original dataframe
  temp = 100*df.isna().sum().sort_values(ascending = False)/ len(df)
  columns_to_remove = []
  for i in temp.index:
   if temp[i] > percentage:
     columns_to_remove.append(i)
  df.drop(columns = columns_to_remove, inplace = True)
  print (f"feature removed : {columns_to_remove}")
  return df





def remove_constant_reading(df, duration, meter = "all", ignore_months = None):
    """
       Parameters
       ----------
       df : pandas dataframe
       A pandas dataframe containing columns "meter_reading", "timestamp".
       "building_id", "meter" are optional.

       meter : int/str (optional) default = "all"
       A value for only considering the rows corresponding to the meter type.
       It is not required if the dataframe doesn't contain the column "meter".
       Refer https://www.kaggle.com/c/ashrae-energy-prediction/data for ashrae/bdg2 dataset

       duration : int
       A positive interger value for duration in hours.
       ignore_months: tuple of length 2, contains integers ( between 1 and 12)
       A tuple of integers to indicate the start month and the end month of the ignore period.

       Returns
       -------
       updated dataframe with rows removed.

       Description
       ------------
       A method for removing the constant reading values which remained constant for more than the set duration (hours).
       A constant reading for a longer duration may occur because of a faulty meter, the readings for the set duration are removed.
       For some meters, i.e "hotwater" in summer (ignore_period = (7,8)) may have a constant zero value.
       We ignore the constant readings found in the period given by ignore_period.
       """
    cols = ['building_id', 'meter']  # columns to group by
    avail_cols = []
    for c in cols:
        if c in df.columns:
            avail_cols.append(c)

    df = df.copy()  # not altering the original dataframe
    df["meter_reading"] = pd.to_numeric(df["meter_reading"])
    df = df.sort_values(by=avail_cols + ["timestamp"])
    df.reset_index(inplace=True, drop=True)

    if type(ignore_months) == tuple:
        temp = df[~df["timestamp"].dt.month.between(ignore_months[0], ignore_months[1])].copy()
    else:
        temp = df.copy()

    print(f"grouping data with {avail_cols}")
    temp['shifted'] = temp.groupby(avail_cols)['meter_reading'].shift(1)
    temp['difference'] = temp['meter_reading'] - temp['shifted']
    if meter != 'all':
        x = temp[(temp['difference'] == 0) & (temp['meter'] == meter)].index.values.tolist()
    else:
        x = temp[(temp['difference'] == 0)].index.values.tolist()
    # find the streaks of zeros and remove
    x.sort()
    y = []
    c = duration - 2
    if c < 0:
        raise ValueError("duration cant be less than 2")
    print(f"const readings for {duration} consecutive rows are removed")
    for i in range(len(x) - c):
        if x[i + c] - x[i] == c:
            y.append(x[i])
    remove_index = y
    del temp  # clearing memory
    df.drop(remove_index, inplace=True)
    print(f"{len(remove_index)}  constant readings removed")
    return df

def remove_group_std_below_threshold(dataset,threshold = 0.1,group_by = ["building_id"],target = "meter_reading"):
    """
    remove the groups with std below threshold for target feature.
    primarily used to remove building with unacceptable values of std.
    parameters
    ----------
    dataset: pandas dataframe
    threshold: float
              std threshold, below which value is unacceptable
    group_by : list of column names
    target : target feature column name

    returns
    ---------
    modified dataframe
    """
    temp = dataset.groupby(group_by)
    final_df = pd.DataFrame()
    for id, id_df in temp:
        value = id_df[target].std()
        if value <= threshold:
            print(f"group : {id} has below threshold std, is removed")
        else:
            final_df = pd.concat([final_df, id_df])
    return final_df


def select_top_k_buildings_std(dataset, k = 20):
    """
    Gets a subset of the dataset by selecting k buildings based on overall std deviation in the meter readings.
    Parameters
    -------------
    dataset: Pandas dataframe
     The dataset containing columns "building_id" and "meter_reading"
    k : int
     No. of buildings to select.
    """

    buildings = []
    b = dataset.groupby(["building_id"])["meter_reading"].std().sort_values().keys().tolist()
    buildings.extend(b)
    list_of_selected_buildings = b[:k]
    return dataset[dataset["building_id"].isin(list_of_selected_buildings)]


