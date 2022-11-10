# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import holidays
from meteocalc import Temp, dew_point, heat_index, wind_chill, feels_like
from geopy.geocoders import Nominatim

"""A module for extracting features."""

def add_temporal_features(df, cyclic = True):
    """
  
    Parameters
    ----------
    df : pandas dataframe
    A pandas dataframe containing column "timestamp".

    cyclic : Boolean (True or False)
    If cyclic is set to True (default), it adds cyclic cordinates to the periodic features weekday, month and year.
        
    Returns
    -------
    updated dataframe with added features.
         
    Description
    ------------    
    A method for adding temporal features to the dataset. These features are extracted from the column "timestamp".
       
    """
    if 'timestamp' not in df.columns:
        raise ValueError("Cannot add temporal features, timestamp not in the dataframe")
    
    df = df.copy() #not altering the original dataframe
    # just in case timestamp is not in datetime format
    df.timestamp = pd.to_datetime(df.timestamp)

    #add the hour of the day 
    df["hour"] = df["timestamp"].dt.hour
    #add the day of the month
    df["day"]  = df["timestamp"].dt.day
    #add the weekday (0-6)
    df["weekday"] = df["timestamp"].dt.weekday
    #add the month of the year
    df["month"] = df["timestamp"].dt.month
    
    if cyclic == True:
      #adding relative time since "2016-01-01" in hours.
      df["rel_time"] = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600  

      # apply cyclic encoding of periodic features using rel_time
      df["hour_x"] = np.cos(2*np.pi*df.rel_time/24)
      df["hour_y"] = np.sin(2*np.pi*df.rel_time/24)
    
      df["month_x"] = np.cos(2*np.pi*df.rel_time/(30.4*24))
      df["month_y"] = np.sin(2*np.pi*df.rel_time/(30.4*24))
    
      df["weekday_x"] = np.cos(2*np.pi*df.rel_time/(7*24))
      df["weekday_y"] = np.sin(2*np.pi*df.rel_time/(7*24))
      #removing rel time 
      df.drop(columns = ["rel_time"], inplace=True)
    print("Temporal features added")
    return df
   


""" Some utility functions """


def holidays_from_country(string, timestamp):
    """
    Parameter
    ----------
    string: str
    The country name as a string.

    timestamp: str
    Timestamp
    
    Returns
    ---------
    0 : if no holiday on the given timestamp. (according to the country)
    1 : if the day given by the timestamp is a holiday. (according to the country)
    
    """
    country = string.replace(" ", "") 
    if country == "Nederland":
          country = "Netherlands"
    try:
     CountryHolidays = getattr(holidays, country)
    except AttributeError:
        raise ValueError(f"refer https://pypi.org/project/holidays/ to check if the country name  {country}  is supported."  )
     
    country_holidays =  CountryHolidays()
    val = country_holidays.get(timestamp, default = 0)
    if val != 0:
        val = 1

    return val     


def get_country(df):
  """
  Parameter
  ----------
  df: pandas dataframe
  df contains columns latitude and longitude

  Returns
  ---------
  Dataframe containing column "country".

  Description
  -----------
  adds the country name containing the geographical location given by the longitude and latitude columns to the dataframe.
  """

  
  geolocator = Nominatim(user_agent="geoapiExercises")

  def get_location(latitude, longitude):
   try:
    location = geolocator.reverse(str(latitude) +","+str(longitude))
   except:
      return "-"
   return location.raw['address']["country"]

   
  df = df.copy() 
  df["country"] = df.apply(lambda row: get_location(row["latitude"],row["longitude"]), axis=1)
  return df




def include_holidays(df, location_info = "site_id"):

      """
      Parameters
      ----------
      df : pandas dataframe
      A pandas dataframe containing column "timestamp". 
      The dataframe should contain any of the following columns provided as options for location_info
      
      location_info : str  (default = "site_id")
      Name of the column containing the location_info
      Options: {"site_id", "country", "longitude_latitude","None"}
          
          "site_id": The site_id column is present in both ashrae and bdg2 datasets. Each site is given a unique site_id. The holidays are calculated by taking 
          both the site_id and timestamp information.
         
          "country" : The country column and timestamp column is taken into consideration for calculating holidays.

          "longitude_latitude" : The country value is evaluated using longitude and latitude columns and based on the country value and the timestamp, holiday feature is calculated.
          
    


        
      Returns
      -------
      updated dataframe with added features
         
      Description
      ------------    
      A method for adding IsHoliday feature to the dataframe. IsHoliday takes boolean values (0,1)
      If the day is a holiday : 1  
      If the day is not a holiday : 0

      The method requires a timestamp column and a column for extracting location information. The holidays for the given location are decided as per the country.          
      """
      
      df = df.copy() #not altering the original dataframe
      if 'timestamp' not in df.columns:
        raise ValueError ("Cannot add holiday, timestamp not in the dataframe")  
      

      # just in case timestamp is not in datetime format
      df.timestamp = pd.to_datetime(df.timestamp)
      
      if location_info == 'site_id':

        if 'site_id' not in df.columns:
         raise ValueError ("Cannot add holiday, site_id not in the dataframe") 

        en_holidays = holidays.England()
        ir_holidays = holidays.Ireland()
        ca_holidays = holidays.Canada()
        us_holidays = holidays.UnitedStates()

        en_idx = df.query('site_id == 1 or site_id == 5 or site_id == 16 or site_id == 18').index
        ir_idx = df.query('site_id == 12').index
        ca_idx = df.query('site_id == 7 or site_id == 11').index
        us_idx = df.query('site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15 or site_id == 17').index

        df['IsHoliday'] = 0
        df.loc[en_idx, 'IsHoliday'] = df.loc[en_idx, 'timestamp'].apply(lambda x: en_holidays.get(x, default=0))
        df.loc[ir_idx, 'IsHoliday'] = df.loc[ir_idx, 'timestamp'].apply(lambda x: ir_holidays.get(x, default=0))
        df.loc[ca_idx, 'IsHoliday'] = df.loc[ca_idx, 'timestamp'].apply(lambda x: ca_holidays.get(x, default=0))
        df.loc[us_idx, 'IsHoliday'] = df.loc[us_idx, 'timestamp'].apply(lambda x: us_holidays.get(x, default=0))

        holiday_idx = df['IsHoliday'] != 0
        df.loc[holiday_idx, 'IsHoliday'] = 1
        df['IsHoliday'] = df['IsHoliday'].astype(np.uint8)
 
      if location_info == 'country':  
        
        if 'country' not in df.columns:
           raise ValueError ("Cannot add holiday, country not in the dataframe") 

        df["IsHoliday"] = df.apply(lambda row: holidays_from_country(row["country"],row["timestamp"]), axis=1)      

      if location_info == "longitude_latitude":
          if 'latitude' not in df.columns:
            raise ValueError ("Cannot add holiday, latitude not in the dataframe")
          if 'longitude' not in df.columns:
            raise ValueError ("Cannot add holiday, longitude not in the dataframe")

          df = get_country(df)
          df["IsHoliday"] = df.apply(lambda row: holidays_from_country(row["country"],row["timestamp"]), axis=1)  
          df.drop("country", axis =1, inplace =True)
          
         
  
      print("IsHoliday feature added")
      return df




"""
Some utility functions
--------------------------
1. c2f : converting temperature from celcius to farenheit. 

2. windchill : calculate the windchill index
    Refer: https://en.wikipedia.org/wiki/Wind_chill
"""

#1.
def c2f(T):
    return T * 9 / 5. + 32

#2.
def windchill(T, v):
    """
    Parameters
    ----------
    T : float 
    Air temperature
    v : float 
    Wind_speed

    Returns
    ----------
    
    windchill : float

    """
    return (10 * v ** .5 - v + 10.5) * (33 - T)


def add_weather_features(df):
    """
    Parameters
    ----------
      df : pandas dataframe
      A pandas dataframe containing columns "air_temperature" and "dew_temperature" (optional) , "wind_speed" (optional).


      Assumptions: If the following columns are present in the dataframe, the mesurement unit considered is described below.
      

      airTemperature: The temperature of the air in degrees Celsius (ºC).
      dewTemperature: The dew point (the temperature to which a given parcel of air must be cooled at constant pressure and water vapor content in order for saturation to occur) in degrees Celsius (ºC).
      windSpeed: The rate of horizontal travel of air past a fixed point (m/s).
     

    Returns
    --------
      updated dataframe with added features. 
         
    Description
    ------------    
    A method to extract weather features from the weather data. 
    The added features are relative humidity, heat, windchill and feellike.

    """
    
    new = []
    # not altering the original dataframe.
    df = df.copy()
    if "timstamp" in df.columns:
      df = df.sort_values("timestamp")
    
    # if dew_temperature and air_temp are present in the dataframe.
    if 'dew_temperature' in df.columns and 'air_temperature' in df.columns:
      df['RH'] = 100 - 5 * (df['air_temperature'] - df['dew_temperature']) 
      df['heat'] = df.apply(lambda x: heat_index(c2f(x.air_temperature), x.RH).c, axis = 1)
      new.extend(["RH", "Heat"])
      if "wind_speed" in df.columns:
         df['feellike'] = df.apply(lambda x: feels_like(c2f(x.air_temperature), x.RH, x.wind_speed * 2.237).c, axis = 1)
         new.extend(["feellike"])
   
    # if wind_speed and air_temp are present in the dataframe.
    if "wind_speed" in df.columns and 'air_temperature' in df.columns:
      df['windchill'] = df.apply(lambda x: windchill(x.air_temperature, x.wind_speed), axis = 1)
      new.extend(["windchill"])   
    
    print(f"weather features added : {new} ")
    return df

#categorical features

def category_to_numeric(df, cat = []):
    """ 
    Parameters
    ----------
      df : pandas dataframe
      A pandas dataframe containing categorical features.

      cat : list of strings
      Pass the list of column names of the categorical features.


    Returns
    ---------
       dataframe : pandas dataframe
       A dataframe with the categorical features converted to numeric by performing dummy encoding. 

    """
    df = df.copy()
    for feature in cat:
      #if the feature is present then apply dummy encoding.
     if feature in df.columns: 

      temp = pd.get_dummies(df[feature], drop_first = True)                   
      df.drop(feature, axis = 1, inplace = True)
      df = pd.concat([df, temp], axis = 1)
    
    return df


# Basic stats

#1. Mean

def add_mean(df, feature, group_by = None):
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column.The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    A method to add a new column to the dataframe. If group_by = None, all the entries in the new volumn will have same value (mean)  . 
    Some of the columns can be used for grouping subsets of the dataframe. Each group will have the same value (mean) in the new column. 
    
    """
    df = df.copy()
    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire dataframe is considered as one unit. Mean is evaluated of the entire column.

       df[f"{feature}_mean"] = df[feature].mean()
       return df
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + "_mean"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name + feature + "_mean"            


    df[col_name] = 0

    for group, group_df in df.groupby(group_by):
     mean = group_df[feature].mean()
     indices = group_df.index.to_list()
     df.loc[indices, col_name] = mean

    return df

#2. Median

def add_median(df, feature, group_by = None):
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column.The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    A method to add a new column to the dataframe. If group_by = None, all the entries in the new volumn will have same value (median)  . 
    Some of the columns can be used for grouping subsets of the dataframe. Each group will have the same value (median) in the new column. 
    
    """
    df = df.copy()
    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire dataframe is considered as one unit. Median is evaluated of the entire column.

       df[f"{feature}_median"] = df[feature].median()
       return df
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + "_median"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name + feature + "_median"            

    # Initialzing the new column.
    df[col_name] = 0

    for group, group_df in df.groupby(group_by):
     median = group_df[feature].median()
     indices = group_df.index.to_list()
     df.loc[indices, col_name] = median

    return df

#3. Standard_Deviation

def add_std(df, feature, group_by = None):
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column.The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    A method to add a new column to the dataframe. If group_by = None, all the entries in the new volumn will have same value (std)  . 
    Some of the columns can be used for grouping subsets of the dataframe. Each group will have the same value (std) in the new column. 
    
    """
    df = df.copy()
    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire dataframe is considered as one unit. Median is evaluated of the entire column.

       df[f"{feature}_std"] = df[feature].std()
       return df
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + "_std"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name = col_name + f +"_" 
       col_name = col_name + feature + "_std"            

    # Initialzing the new column.
    df[col_name] = 0

    for group, group_df in df.groupby(group_by):
     std = group_df[feature].median()
     indices = group_df.index.to_list()
     df.loc[indices, col_name] = std

    return df    


#4. Maximum


def add_max(df, feature, group_by = None):

    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    A method to add a new column to the dataframe. If group_by = None, all the entries in the new volumn will have same value (max)  . 
    Some of the columns can be used for grouping subsets of the dataframe. Each group will have the same value (max) in the new column. 
    
    """
    df = df.copy()
    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire dataframe is considered as one unit. Max is evaluated of the entire column.

       df[f"{feature}_max"] = df[feature].max()
       return df
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + "_max"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name = col_name + f +"_" 
       col_name = col_name + feature + "_max"            

    # Initialzing the new column.
    df[col_name] = 0

    for group, group_df in df.groupby(group_by):
     maximum = group_df[feature].max()
     indices = group_df.index.to_list()
     df.loc[indices, col_name] = maximum

    return df    


#5. Minimum


def add_min(df, feature, group_by = None):
   
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    A method to add a new column to the dataframe. If group_by = None, all the entries in the new volumn will have same value (min)  . 
    Some of the columns can be used for grouping subsets of the dataframe. Each group will have the same value (min) in the new column. 
    
    """
    df = df.copy()
    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire dataframe is considered as one unit. Min is evaluated of the entire column.

       df[f"{feature}_min"] = df[feature].min()
       return df
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + "_min"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name  +feature + "_min"            

    # Initialzing the new column.
    df[col_name] = 0

    for group, group_df in df.groupby(group_by):
     minimum = group_df[feature].min()
     indices = group_df.index.to_list()
     df.loc[indices, col_name] = minimum

    return df    






#6. Percentile
    
def add_percentile(df, feature, percentile = 50, group_by = None):
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subgroups of the dataset.

    percentile : int (between 0 to 100) 
    Set the percentile value to be extracted from the feature.

    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    A method to add a new column to the dataframe. If group_by = None, all the entries in the new volumn will have same value (percentile)  . 
    Some of the columns can be used for grouping subsets of the dataframe. Each group will have the same value (percentile) in the new column. 
    
    """
    df = df.copy()
    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire dataframe is considered as one unit. Median is evaluated of the entire column.

       df[f"{feature}_{percentile}%"] = np.percentile(df[feature], percentile)
       return df
    
    if type(group_by) == str:
      col_name = group_by + "_" + feature + f"_{percentile}%"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name  + feature + f"_{percentile}%"            

    # Initialzing the new column.
    df[col_name] = 0

    for group, group_df in df.groupby(group_by):
     per = np.percentile(group_df[feature], percentile)
     indices = group_df.index.to_list()
     df.loc[indices,col_name] = per

    return df  


#  Rolling Feature extraction

def add_moving_mean(df, feature, group_by = None, window = 6):
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subset of the dataset. (If the dataset contains multiple timeseries) 
    
    window : int
    Provide a rolling window for calculating the moving stats.


    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    For the timeseries data, add rolling mean column for the provided feature. If the dataset contains multiple timeseries, use group_by to isolate them.
    For example: The weather data for ASHRAE dataset contains multiple timeseries per feature across multiple sites. At one particular timestamp each site has input present in the column.
                 Use the group_by parameter to seperate the timeseries :  add_moving_mean(weather_df, group_by = "site_id").
            
    """
    df = df.copy() 
    
    # just to make sure, entries are in a sequential order
    if "timestamp" in df.columns:
      df = df.sort_values("timestamp")
      df.reset_index(drop = True, inplace = True) 
    

    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire feature timeseries is considered as one timeseries. Moving average is evaluated for the feature column.

       df[f"{feature}_moving_mean_{window}"] = df[feature].rolling(window, 0).mean()
       return df
    
    
    # All the different timeseries coexisting in the dataframe of the selected feature are grouped by using group_by. The moving feature values are calculated on each of them seperately. 
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + f"_moving_mean_{window}"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name + feature + f"_moving_mean_{window}"            

    # calculating the rolling(moving) feature column.
    df[col_name] = df.groupby(group_by)[feature].transform(lambda x: x.rolling(window, 0).mean())

    return df






def add_moving_median(df, feature, group_by = None, window = 6):

    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subset of the dataset. (If the dataset contains multiple timeseries) 
    
    window : int
    Provide a rolling window for calculating the moving stats.


    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    For the timeseries data, add rolling median column for the provided feature. If the dataset contains multiple timeseries, use group_by to isolate them.
    For example: The weather data for ASHRAE dataset contains multiple timeseries per feature across multiple sites. At one particular timestamp each site has input present in the column.
                 Use the group_by parameter to seperate the timeseries :  add_moving_median(weather_df, group_by = "site_id").
            
    """
    df = df.copy() 
    
    # just to make sure, entries are in a sequential order
    if "timestamp" in df.columns:
      df = df.sort_values("timestamp")
      df.reset_index(drop = True, inplace = True) 
    

    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire feature timeseries is considered as one timeseries. Moving median is evaluated for the feature column.

       df[f"{feature}_moving_median_{window}"] = df[feature].rolling(window, 0).median()
       return df
    
    
    # All the different timeseries coexisting in the dataframe of the selected feature are grouped by using group_by. The moving feature values are calculated on each of them separately. 
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + f"_moving_median{window}"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name + feature + f"_moving_median{window}"            

    # calculating the rolling (moving) feature column.
    df[col_name] = df.groupby(group_by)[feature].transform(lambda x: x.rolling(window, 0).median())

    return df


    

def add_moving_std(df, feature, group_by = None, window = 6):
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subset of the dataset. (If the dataset contains multiple timeseries) 
    
    window : int
    Provide a rolling window for calculating the moving stats.


    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    For the timeseries data, add rolling standard deviation column for the provided feature. If the dataset contains multiple timeseries, use group_by to isolate them.
    For example: The weather data for ASHRAE dataset contains multiple timeseries per feature across multiple sites. At one particular timestamp each site has input present in the column.
                 Use the group_by parameter to seperate the timeseries :  add_moving_std(weather_df, group_by = "site_id").
            
    """
    df = df.copy() 
    
    # just to make sure, entries are in a sequential order
    if "timestamp" in df.columns:
      df = df.sort_values("timestamp")
      df.reset_index(drop = True, inplace = True) 
    

    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire feature timeseries is considered as one timeseries. Moving standard deviation is evaluated for the feature column.

       df[f"{feature}_moving_median_{window}"] = df[feature].rolling(window, 0).std()
       return df
    
    
    # All the different timeseries coexisting in the dataframe of the selected feature are grouped by using group_by. The moving feature values are calculated on each of them separately. 
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + f"_moving_std{window}"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name + feature + f"_moving_std{window}"            

    # calculating the rolling (moving) feature column.
    df[col_name] = df.groupby(group_by)[feature].transform(lambda x: x.rolling(window, 0).std())

    return df
    


def add_moving_max(df, feature, group_by = None, window = 6): 
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subset of the dataset. (If the dataset contains multiple timeseries) 
    
    window : int
    Provide a rolling window for calculating the moving stats.


    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    For the timeseries data, add rolling max value column for the provided feature. If the dataset contains multiple timeseries, use group_by to isolate them.
    For example: The weather data for ASHRAE dataset contains multiple timeseries per feature across multiple sites. At one particular timestamp each site has input present in the column.
                 Use the group_by parameter to seperate the timeseries :  add_moving_max(weather_df, group_by = "site_id").
            
    """
    df = df.copy() 
    
    # just to make sure, entries are in a sequential order
    if "timestamp" in df.columns:
      df = df.sort_values("timestamp")
      df.reset_index(drop = True, inplace = True) 
    

    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire feature timeseries is considered as one timeseries. Moving maxium is evaluated for the feature column.

       df[f"{feature}_moving_max_{window}"] = df[feature].rolling(window, 0).max()
       return df
    
    
    # All the different timeseries coexisting in the dataframe of the selected feature are grouped by using group_by. The moving feature values are calculated on each of them separately. 
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + f"_moving_max{window}"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name + feature + f"_moving_max{window}"            

    # calculating the rolling (moving) feature column.
    df[col_name] = df.groupby(group_by)[feature].transform(lambda x: x.rolling(window, 0).max())

    return df

    

def add_moving_min(df, feature, group_by = None, window = 6):
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subset of the dataset. (If the dataset contains multiple timeseries) 
    
    window : int
    Provide a rolling window for calculating the moving stats.


    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    For the timeseries data, add rolling minimum value column for the provided feature. If the dataset contains multiple timeseries, use group_by to isolate them.
    For example: The weather data for ASHRAE dataset contains multiple timeseries per feature across multiple sites. At one particular timestamp each site has input present in the column.
                 Use the group_by parameter to seperate the timeseries :  add_moving_min(weather_df, group_by = "site_id").
            
    """
    df = df.copy() 
    
    # just to make sure, entries are in a sequential order
    if "timestamp" in df.columns:
      df = df.sort_values("timestamp")
      df.reset_index(drop = True, inplace = True) 
    

    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire feature timeseries is considered as one timeseries. Moving minimum is evaluated for the feature column.

       df[f"{feature}_moving_min_{window}"] = df[feature].rolling(window, 0).min()
       return df
    
    
    # All the different timeseries coexisting in the dataframe of the selected feature are grouped by using group_by. The moving feature values are calulated on each of them separately. 
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + f"_moving_min{window}"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name + feature + f"_moving_min{window}"            

    # calculating the rolling (moving) feature column.
    df[col_name] = df.groupby(group_by)[feature].transform(lambda x: x.rolling(window, 0).min())

    return df
    
    
    
    

def add_moving_percentile(df, feature, group_by = None, window = 6, percentile = 50):
    """
    Parameters
    -----------
    df : pandas dataframe
    
    feature : str
    String name of the feature column. The base feature from which the new feature is extracted.
    
    group_by : str or a list
    Provide a string of column name or a list of column names to create subset of the dataset. (If the dataset contains multiple timeseries) 
    
    window : int
    Provide a rolling window for calculating the moving stats.

    percentile : int (between 0 to 100) 
    Set the percentile value to be extracted from the feature in the rolling interval. 


    
    Returns
    --------
    The input dataframe with added feature.

    Description
    -----------
    For the timeseries data, add rolling percentile column for the provided feature. If the dataset contains multiple timeseries, use group_by to isolate them.
    For example: The weather data for ASHRAE dataset contains multiple timeseries per feature across multiple sites. At one particular timestamp each site has input present in the column.
                 Use the group_by parameter to seperate the timeseries :  add_moving_percentile(weather_df, group_by = "site_id", window = 6, percentile = 80).
            
    """
    df = df.copy() 
    
    # just to make sure, entries are in a sequential order
    if "timestamp" in df.columns:
      df = df.sort_values("timestamp")
      df.reset_index(drop = True, inplace = True) 
    

    if feature not in df.columns:
        raise ValueError(f"The feature {feature} is not present in the dataframe")

    if group_by == None:
    # The entire feature timeseries is considered as one timeseries. Moving percentile is evaluated for the feature column.

       df[f"{feature}_moving_{percentile}%_{window}"] = df[feature].rolling(window, 0).quantile(percentile/100)
       return df
    
    
    # All the different timeseries coexisting in the dataframe of the selected feature are grouped by using group_by. The moving feature values are calulated on each of them separately. 
    
    if type(group_by) == str:
      col_name = group_by + "_" +feature + f"_moving_{percentile}%_{window}"
    
    else:  # if it is a list
       col_name = ""
       for f in group_by:
           col_name= col_name + f +"_" 
       col_name = col_name + feature + f"_moving_{percentile}%_{window}"            

    # calculating the rolling (moving) feature column.
    df[col_name] = df.groupby(group_by)[feature].transform(lambda x: x.rolling(window, 0).quantile(percentile/100))

    return df







def add_periodic_mean(df, feature, period = "day_of_the_week", group_by = None):
      """
      Parameters
      -----------
      df : pandas dataframe
    
      feature : str
      String name of the feature column.The base feature from which the new feature is extracted.
    
      period: str
        Options :- {"day_of_the_week", "hour_of_the_day", "month_of_the_year"}


      group_by : str or a list
      Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
      Returns
      --------
      The input dataframe with added feature.

      Description
      -----------
      A method to add a new column to the dataframe. If group_by = None, The new column for a particular row will contain the mean value of the selected feature taken over the periodic bin.
      For example: period = "day_of_the_week". The dataset is divided over 7 bins, one for each day. The mean value for the feature is taken for each of them seperately. The row entry will get the mean
      value corresponding to it's  day of the week. The dataset can be further grouped by different columns present by passing the columns in the group_by parameter.
      For each combination of column(s) used for grouping and the periodic bins, the mean value is calculated seperately.  
      
      """ 
      
      
      df = df.copy()      
      if "timestamp" not in df.columns:
          raise ValueError("Cant extract periodic mean, timestamp column missing.")
      
      if period == "day_of_the_week":
           df[period] = df["timestamp"].dt.weekday    
           
      if period == "hour_of_the_day":
           df[period] = df["timestamp"].dt.hour 
      
      if period == "month_of_the_year":
          df[period] = df["timestamp"].dt.month

      if group_by is None:
          group_by = period    
  
      elif type(group_by) == str:
          group_by = [group_by, period]
     
      else:
          group_by.append(period)
  
      df = add_mean(df,feature,group_by)
      df.drop(period, axis = 1, inplace = True)
      return df
 
def add_periodic_median(df, feature, period = "day_of_the_week", group_by = None):
      
      """
      Parameters
      -----------
      df : pandas dataframe
    
      feature : str
      String name of the feature column.The base feature from which the new feature is extracted.
    
      period: str
        Options :- {"day_of_the_week", "hour_of_the_day", "month_of_the_year"}


      group_by : str or a list
      Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
      Returns
      --------
      The input dataframe with added feature.

      Description
      -----------
      A method to add a new column to the dataframe. If group_by = None, The new column for a particular row will contain the median value of the selected feature taken over the periodic bin.
      For example: period = "day_of_the_week". The dataset is divided over 7 bins, one for each day. The mean value for the feature is taken for each of them seperately. The row entry will get the mean
      value corresponding to it's  day of the week. The dataset can be further grouped by different columns present by passing the columns in the group_by parameter. 
      For each combination of column(s) used for grouping and the periodic bins, the median value is calculated seperately. 
      
      """ 
      
      df = df.copy()      
      if "timestamp" not in df.columns:
          raise ValueError("Cant extract periodic mean, timestamp column missing.")
      
      if period == "day_of_the_week":
           df[period] = df["timestamp"].dt.weekday    
           
      if period == "hour_of_the_day":
           df[period] = df["timestamp"].dt.hour 
      
      if period == "month_of_the_year":
          df[period] = df["timestamp"].dt.month

      if group_by is None:
          group_by = period    
  
      elif type(group_by) == str:
          group_by = [group_by, period]
     
      else:
          group_by.append(period)
  
      df = add_median(df,feature,group_by)
      df.drop(period, axis = 1, inplace = True)
      return df
 

def add_periodic_std(df, feature, period = "day_of_the_week", group_by = None):

      """
      Parameters
      -----------
      df : pandas dataframe
    
      feature : str
      String name of the feature column.The base feature from which the new feature is extracted.
    
      period: str
        Options :- {"day_of_the_week", "hour_of_the_day", "month_of_the_year"}


      group_by : str or a list
      Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
      Returns
      --------
      The input dataframe with added feature.

      Description
      -----------
      A method to add a new column to the dataframe. If group_by = None, The new column for a particular row will contain the standard deviation value of the selected feature taken over the periodic bin.
      For example: period = "day_of_the_week". The dataset is divided over 7 bins, one for each day. The mean value for the feature is taken for each of them seperately. The row entry will get the mean
      value corresponding to it's  day of the week. The dataset can be further grouped by different columns present by passing the columns in the group_by parameter. 
      For each combination of column(s) used for grouping and the periodic bins, the standard deviation value is calculated seperately.   
      
      """ 
      
      df = df.copy()      
      if "timestamp" not in df.columns:
          raise ValueError("Cant extract periodic mean, timestamp column missing.")
      
      if period == "day_of_the_week":
           df[period] = df["timestamp"].dt.weekday    
           
      if period == "hour_of_the_day":
           df[period] = df["timestamp"].dt.hour 
      
      if period == "month_of_the_year":
          df[period] = df["timestamp"].dt.month

      if group_by is None:
          group_by = period    
  
      elif type(group_by) == str:
          group_by = [group_by, period]
     
      else:
          group_by.append(period)
  
      df = add_std(df,feature,group_by)
      df.drop(period, axis = 1, inplace = True)
      return df


def add_periodic_min(df, feature, period = "day_of_the_week", group_by = None):

      """
      Parameters
      -----------
      df : pandas dataframe
    
      feature : str
      String name of the feature column.The base feature from which the new feature is extracted.
    
      period: str
        Options :- {"day_of_the_week", "hour_of_the_day", "month_of_the_year"}


      group_by : str or a list
      Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
      Returns
      --------
      The input dataframe with added feature.

      Description
      -----------
      A method to add a new column to the dataframe. If group_by = None, The new column for a particular row will contain the minimum value of the selected feature taken over the periodic bin.
      For example: period = "day_of_the_week". The dataset is divided over 7 bins, one for each day. The mean value for the feature is taken for each of them seperately. The row entry will get the mean
      value corresponding to it's  day of the week. The dataset can be further grouped by different columns present by passing the columns in the group_by parameter. 
      For each combination of column(s) used for grouping and the periodic bins, the minimum value is calculated seperately.
      
      """ 
      
      df = df.copy()      
      if "timestamp" not in df.columns:
          raise ValueError("Cant extract periodic mean, timestamp column missing.")
      
      if period == "day_of_the_week":
           df[period] = df["timestamp"].dt.weekday    
           
      if period == "hour_of_the_day":
           df[period] = df["timestamp"].dt.hour 
      
      if period == "month_of_the_year":
          df[period] = df["timestamp"].dt.month

      if group_by is None:
          group_by = period    
  
      elif type(group_by) == str:
          group_by = [group_by, period]
     
      else:
          group_by.append(period)
  
      df = add_min(df,feature,group_by)
      df.drop(period, axis = 1, inplace = True)
      return df


def add_periodic_max(df, feature, period = "day_of_the_week", group_by = None):
      """
      Parameters
      -----------
      df : pandas dataframe
    
      feature : str
      String name of the feature column.The base feature from which the new feature is extracted.
    
      period: str
        Options :- {"day_of_the_week", "hour_of_the_day", "month_of_the_year"}


      group_by : str or a list
      Provide a string of column name or a list of column names to create subgroups of the dataset. 

    
      Returns
      --------
      The input dataframe with added feature.

      Description
      -----------
      A method to add a new column to the dataframe. If group_by = None, The new column for a particular row will contain the maximum value of the selected feature taken over the periodic bin.
      For example: period = "day_of_the_week". The dataset is divided over 7 bins, one for each day. The mean value for the feature is taken for each of them seperately. The row entry will get the mean
      value corresponding to it's  day of the week. The dataset can be further grouped by different columns present by passing the columns in the group_by parameter. 
      For each combination of column(s) used for grouping and the periodic bins, the maximum value is calculated seperately.
      
      """ 
      
      df = df.copy()      
      if "timestamp" not in df.columns:
          raise ValueError("Cant extract periodic mean, timestamp column missing.")
      
      if period == "day_of_the_week":
           df[period] = df["timestamp"].dt.weekday    
           
      if period == "hour_of_the_day":
           df[period] = df["timestamp"].dt.hour 
      
      if period == "month_of_the_year":
          df[period] = df["timestamp"].dt.month

      if group_by is None:
          group_by = period    
  
      elif type(group_by) == str:
          group_by = [group_by, period]
     
      else:
          group_by.append(period)
  
      df = add_max(df,feature,group_by)
      df.drop(period, axis = 1, inplace = True)
      return df


def add_periodic_percentile(df, feature, period = "day_of_the_week", group_by = None, percentile = 50):
      """
      Parameters
      -----------
      df : pandas dataframe
    
      feature : str
      String name of the feature column.The base feature from which the new feature is extracted.
    
      period: str
        Options :- {"day_of_the_week", "hour_of_the_day", "month_of_the_year"}


      group_by : str or a list
      Provide a string of column name or a list of column names to create subgroups of the dataset.

      percentile : int (between 0 to 100) 
      Set the percentile value to be extracted from the feature.



    
      Returns
      --------
      The input dataframe with added feature.

      Description
      -----------
      A method to add a new column to the dataframe. If group_by = None, The new column for a particular row will contain the percentile value of the selected feature taken over the periodic bin.
      For example: period = "day_of_the_week". The dataset is divided over 7 bins, one for each day. The mean value for the feature is taken for each of them seperately. The row entry will get the mean
      value corresponding to it's day of the week. The dataset can be further grouped by different columns present by passing the columns in the group_by parameter. 
      For each combination of column(s) used for grouping and the periodic bins, the percentile value is calculated seperately.
      
      """  


      
      df = df.copy()      
      if "timestamp" not in df.columns:
          raise ValueError("Cant extract periodic mean, timestamp column missing.")
      
      if period == "day_of_the_week":
           df[period] = df["timestamp"].dt.weekday    
           
      if period == "hour_of_the_day":
           df[period] = df["timestamp"].dt.hour 
      
      if period == "month_of_the_year":
          df[period] = df["timestamp"].dt.month

      if group_by is None:
          group_by = period    
  
      elif type(group_by) == str:
          group_by = [group_by, period]
     
      else:
          group_by.append(period)
  
      df = add_percentile(df,feature, percentile, group_by)
      df.drop(period, axis = 1, inplace = True)
      return df














