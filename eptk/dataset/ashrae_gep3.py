# -*- coding: utf-8 -*-
import os
from ..utils import reduce_mem_usage, get_meter_df
from ..preprocessing.clean import remove_missing_meter_reading, remove_readings_threshold
from  ..preprocessing.impute import add_missing_timestamp, impute_weather
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from .base import Dataset
from os.path import dirname, join
import json
from .download_from_gdrive import download_file_from_google_drive
from zipfile import ZipFile

def weather_time_correction(weather_df):
    
    """ timestamp error correction.
  
  Parameters
  ----------
  weather_df : pandas dataframe
  weather dataframe with "timestamp" and "site_id" columns
  
  Returns
  --------
  str : "Weather Time Corrected"
  returns the above message after the timestamp is adjusted.

  Description
  -------------

  The weather dataset has timestamps with utc time. The timestamps are adjusted for each of the sites (16 sites in total) according to their local time.
         
    """
      
    if 'timestamp' not in weather_df.columns:
        raise ValueError("Cannot correct time, timestamp not in the dataframe")

    if 'site_id' not in weather_df.columns:
        raise ValueError("Cannot correct time, site_id not in the dataframe")

    country = ['UnitedStates', 'England', 'UnitedStates', 'UnitedStates', 'UnitedStates',
           'England', 'UnitedStates', 'Canada', 'UnitedStates', 'UnitedStates',
           'UnitedStates', 'Canada', 'Ireland', 'UnitedStates', 'UnitedStates', 'UnitedStates']

    city = ['Jacksonville', 'London', 'Phoenix', 'Philadelphia', 'San Francisco',
            'Loughborough', 'Philadelphia', 'Montreal', 'Jacksonville', 'San Antonio',
             'Las Vegas', 'Montreal', 'Dublin', 'Minneapolis', 'Philadelphia', 'Pittsburgh']

    UTC_offset = [-4, 0, -7, -4, -9, 0, -4, -4, -4, -5, -7, -4, 0, -5, -4, -4]
    location_data = pd.DataFrame(np.array([country, city, UTC_offset]).T, index = range(16), columns = ['country', 'city', 'UTC_offset'])
    
    for idx in location_data.index:
       weather_df.loc[weather_df['site_id'] == idx, 'timestamp'] += timedelta(hours = int(location_data.loc[idx, 'UTC_offset']))
    
    return "Weather Time corrected"





# loading the config file
module_path = dirname(__file__)
with open(join(module_path, "configs/ashrae_config.json"), "r") as f:
    configuration = json.load(f)



class ASHRAE_GEP3(Dataset):
    """
    
    Parameters
    ------------
    config : dict 
    Metadata describing the dataset name, download link etc.
    
    Attributes
    -----------

    meter : pandas dataframe (or None)
    Datframe containing the meter readings.
    
    weather : pandas dataframe (or None)
    Dataframe containing the weather information.
    
    meta : pandas dataframe (or None)
    Dataframe containing the building meta information.

    """
    def __init__(self, config = configuration):
        super().__init__(config)
        self.meter = None
        self.weather = None
        self.meta = None

    def _download(self, files_to_download):
        """
        Parameters
        -------------
        files_to_download : List of name of files to download.

        Description
        ------------
        A method to fetch the link from the metafile to download the files.
        A private method called by the load method.


        """
        if files_to_download == []:
            print("files already present in the working directory.")
            return
        else:
            _ids = configuration["gdrive_ids"]
            if "train.csv" in files_to_download:
                download_file_from_google_drive(_ids["ashrae_train.zip"],"datasets/ashrae_gep3/ashrae_train.zip")
                # extracting the zip file
                with ZipFile("datasets/ashrae_gep3/ashrae_train.zip", 'r') as zip:
                    zip.extractall("datasets/ashrae_gep3")


            if "building_metadata.csv" in files_to_download:
                download_file_from_google_drive(_ids["building_metadata.csv"], "datasets/ashrae_gep3/building_metadata.csv")

            if "weather_train.csv" in files_to_download:
                download_file_from_google_drive(_ids["ashrae_weather.zip"], "datasets/ashrae_gep3/ashrae_weather.zip")
                with ZipFile("datasets/ashrae_gep3/ashrae_weather.zip", 'r') as zip:
                    zip.extractall("datasets/ashrae_gep3")

            if "site_info.csv" in files_to_download:
                download_file_from_google_drive(_ids["site_info.csv"], "datasets/ashrae_gep3/site_info.csv")




    def check_path(self):
        """
        Returns
        ---------
        A list of files to download.
        Empty list if all the files are present.


        Description
        _____________
        Name of the csv files will be present in the config dictionary.
        A method to fetch it and check if the path contains the csv files. """


        #current working directory

        current = os.getcwd()
        if os.path.exists("datasets/ashrae_gep3"):
            print("checking the relevant sub-directory ...")
        else:
            os.makedirs("datasets/ashrae_gep3")
            print("unable to locate csv files")
            download = self.config["file_names"]
            return download
        download = []
        for file_name in self.config["file_names"]:
            path = "datasets/ashrae_gep3"
            if file_name not in os.listdir(path):
                 download.append(file_name)

        if download == []:
            print("All the required files found locally.")
        return download


    def _clean(self, compress = True, fill_missing_timestamp = True, metertype = "all", remove_electric_zero = True, weather_impute = True):        
        
        """ A private method to clean the datasets. Called by the load method"""
        if self.weather is not None:  
            weather_time_correction(self.weather)

            if fill_missing_timestamp == True:
                self.weather = add_missing_timestamp(self.weather)
            
            if weather_impute == True:
                self.weather = impute_weather(self.weather)

            
            if compress == True:
                reduce_mem_usage(self.weather) 
            

        if self.meter is not None: 
            self.meter = remove_missing_meter_reading(self.meter)
            if metertype != "all" :
              self.meter = get_meter_df(self.meter, metertype)
            if remove_electric_zero == True:
              self.meter = remove_readings_threshold(self.meter, 0.01 , meter = 0)
            
            if compress == True:
                reduce_mem_usage(self.meter)



        if self.meta is not None:
            if compress == True:
                reduce_mem_usage(self.meta)            
         
        return

    


    def load(self, data_type = "all", compress = True, metertype = "all", fill_missing_timestamp = True, remove_electric_zero = True, weather_impute = True):
       
        """        
        Parameters
        ----------
        data_type : str
        Select from "weather", "meta", "meter" and "all" (default)  to access the type of dataframe(s).

        compress : Boolean (True or False)
        If True, compresses the dataframe(s) before returning.

        metertype: an integers or string "all" (default)
        An integer corresponding to a metertype.
        To extract readings only from the selected meter type.
        "all" will extract readings from all the meters.

        meter   :  meter type
         -------   ------------
          0      :   electric
          1      :   chilled_water
          2      :   steam
          3      :   hotwater
   
        If set to all, the entire meter dataset is extracted.
        If set to an integer, for example metertype = 0
        then only the electric readings are extracted.

        fill_missing_timestamp : Boolean (True or False)
        If True, adds missing timestamps from the start time till the end time in the weather dataframe.

        remove_electric_zero : Boolean (True or False)
        If True, the electric meter readings which are below 0.01 are removed.
        
        weather_impute : Boolean (True or False)
        If True, imputes the weather features. 

         



        Returns
        -------

        Dataframe(s): pandas dataframe, or a tuple of pandas dataframes. 
        Returns the type of dataframe specified, or returns a tuple of (meta dataframe, meter dataframe, weather dataframe).  
  
        Description
        ------------
        A method to load the dataframe specified by type. There are 3 diffrent types of dataframes.
        meta  :  contains the meta information of the buildings from where the meter reading are collected.
        meter : contains the meter readings collected from all the buidlings.
        weather: contains the weather information of all the building sites.

        The weather dataframe timestamps are adjusted by the utc offset, to be coherent with the meter dataframe timestamps.
  
  

        """   
            


        if self.check_path() != []:
           self._download(self.check_path())

        #  ["train.csv", "building_metadata.csv", "weather_train.csv", "site_info.csv"]
        meta = "datasets/ashrae_gep3/building_metadata.csv"
        meter = "datasets/ashrae_gep3/train.csv"
        weather = "datasets/ashrae_gep3/weather_train.csv"

       
        # parse the data into pandas dataframe
        if data_type == "all":
           self.meter =  pd.read_csv(meter, parse_dates = ['timestamp'])
           self.weather = pd.read_csv(weather, parse_dates = ['timestamp'])
           self.meta = pd.read_csv(meta)
           self._clean(compress, fill_missing_timestamp, metertype, remove_electric_zero, weather_impute)
           return (self.meta, self.meter, self.weather)
        
        else:
         if data_type == "meter":
            self.meter =  pd.read_csv(meter, parse_dates = ['timestamp'])
            # change others to none, for reusing the object instance
            self.meta = None
            self.weather = None
            self._clean(compress, fill_missing_timestamp, metertype, remove_electric_zero, weather_impute)
            return (self.meter)
        

         if data_type == "weather":
            self.weather = pd.read_csv(weather, parse_dates = ['timestamp'])
            self._clean(compress, fill_missing_timestamp, metertype, remove_electric_zero, weather_impute)
            # change others to none, for reusing the object instance
            self.meter = None
            self.meta = None
            return (self.weather)
            

         if data_type == "meta":
                self.meta = pd.read_csv(meta)
                self._clean(compress, fill_missing_timestamp, metertype, remove_electric_zero, weather_impute)
                
                # change others to none, for reusing the object instance
                self.meter = None
                self.weather = None
                
                return (self.meta)
                

    def get_site_info(self):

        """
        A method to fetch the site information.
        It returns a dataframe consisting of columns: [site, site_id, longitute, latitute, country] 
        """
        site_info = "datasets/ashrae_gep3/site_info.csv"
        x = pd.read_csv(site_info)
        return x





