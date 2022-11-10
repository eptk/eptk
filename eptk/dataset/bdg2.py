# -*- coding: utf-8 -*-

import os
from ..utils import reduce_mem_usage
from ..preprocessing.clean import remove_missing_meter_reading, remove_readings_threshold
from  ..preprocessing.impute import add_missing_timestamp, impute_weather
import pandas as pd
from os.path import dirname, join
from .base import Dataset
import json
from .download_from_gdrive import download_file_from_google_drive
from zipfile import ZipFile


      
 

# loading the config file
module_path = dirname(__file__)
with open(join(module_path, "configs/bdg2_config.json"), "r") as f:
    configuration = json.load(f)



class BuildingDataGenome2(Dataset):
    """
    Parameters
    ------------
    config : dict 
    Metadata describing the dataset name, download link etc.

    
    Attributes
    ----------


    meter : pandas dataframe (or None)
    Datframe containing the meter readings.
    
    weather : pandas dataframe (or None)
    Dataframe containing the weather information.
    
    meta : pandas dataframe (or None)
    Dataframe containing the building meta information.
    """
    def __init__(self, config = configuration):
        # set the relative locations of the csv files.
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

            if "weather.csv" in files_to_download:
                download_file_from_google_drive(_ids["bdg2_weather.zip"], "datasets/building_genome2/bdg2_weather.zip")
                with ZipFile("datasets/building_genome2/bdg2_weather.zip", 'r') as zip:
                    zip.extractall("datasets/building_genome2")


            if "electric_meter.csv" in files_to_download:
                download_file_from_google_drive(_ids["bdg2_electric_meter.zip"], "datasets/building_genome2/electric_meter.zip")
                # extracting the zip file
                with ZipFile("datasets/building_genome2/electric_meter.zip", 'r') as zip:
                    zip.extractall("datasets/building_genome2")

            if "meta.csv" in files_to_download:
                download_file_from_google_drive(_ids["meta.csv"],
                                                "datasets/building_genome2/meta.csv")



            if "site_info.csv" in files_to_download:
                download_file_from_google_drive(_ids["site_info.csv"], "datasets/building_genome2/site_info.csv")

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

        # current working directory

        current = os.getcwd()
        if os.path.exists("datasets/building_genome2"):
            print("checking the relevant sub-directory ...")
        else:
            os.makedirs("datasets/building_genome2")
            print("unable to locate csv files")
            download = self.config["file_names"]
            return download
        download = []
        for file_name in self.config["file_names"]:
            path = "datasets/building_genome2"
            if file_name not in os.listdir(path):
                download.append(file_name)

        if download == []:
            print("All the required files found locally.")
        return download

    def _clean(self, compress = True, fill_missing_timestamp = True, remove_electric_zero = True, weather_impute = True):        
        """ A private method to clean the datasets. Called by the load method."""
        if self.weather is not None:
            if fill_missing_timestamp == True:
                self.weather = add_missing_timestamp(self.weather)
            
            if weather_impute == True:
                self.weather = impute_weather(self.weather)

            
            if compress == True:
                reduce_mem_usage(self.weather) 
            

        if self.meter is not None: 
            self.meter = remove_missing_meter_reading(self.meter)
            if remove_electric_zero == True:
              self.meter = remove_readings_threshold(self.meter, 0.01)
            
            if compress == True:
                reduce_mem_usage(self.meter)



        if self.meta is not None:
            if compress == True:
                reduce_mem_usage(self.meta)
                            
         
        return

    


    def load(self, data_type = "all", compress = True, fill_missing_timestamp = True, remove_electric_zero = True, weather_impute = True):
       
        """        
        Parameters
        ----------
        data_type : str
        Select from "weather", "meta", "meter" and "all" (default)  to access the type of dataframe(s).

        compress : Boolean (True or False)
        If True, compresses the dataframe(s) before returning.

        

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

        
  
  

        """

        if self.check_path() != []:
            self._download(self.check_path())

        # ["electric_meter.csv", "meta.csv", "weather.csv", "site_info.csv"
        meta = "datasets/building_genome2/meta.csv"
        meter = "datasets/building_genome2/electric_meter.csv"
        weather = "datasets/building_genome2/weather.csv"

       
        # parse the data into pandas dataframe
        if data_type == "all":
           self.meter =  pd.read_csv(meter, parse_dates = ['timestamp'])
           self.weather = pd.read_csv(weather, parse_dates = ['timestamp'])
           self.meta = pd.read_csv(meta)
           self._clean(compress, fill_missing_timestamp, remove_electric_zero, weather_impute)
           return (self.meta, self.meter, self.weather)
        
        else:
         if data_type == "meter":
            self.meter =  pd.read_csv(meter, parse_dates = ['timestamp'])
            self._clean(compress, fill_missing_timestamp, remove_electric_zero, weather_impute)
            # change others to none, for reusing the object instance
            self.meta = None
            self.weather = None
            
            return (self.meter)
        

         if data_type == "weather":
            self.weather = pd.read_csv(weather, parse_dates = ['timestamp'])
            self._clean(compress, fill_missing_timestamp, remove_electric_zero, weather_impute)
            # change others to none, for reusing the object instance
            self.meter = None
            self.meta = None
            return (self.weather)
            

         if data_type == "meta":
                self.meta = pd.read_csv(meta)
                self._clean(compress, fill_missing_timestamp, remove_electric_zero, weather_impute)
                # change others to none, for reusing the object instance
                self.meter = None
                self.weather = None 
                return (self.meta)
                

    def get_site_info(self):
        """
        A method to fetch the site information.
        It returns a dataframe consisting of columns: [site, site_id, longitute, latitute, country] 
        """
        site_info = "datasets/building_genome2/site_info.csv"
        x = pd.read_csv(site_info)
        return x    



