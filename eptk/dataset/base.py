# -*- coding: utf-8 -*-

from abc import ABC,abstractmethod

class Dataset(ABC):
  """
  Base class for all the dataset extraction.
     
  Parameters
  ------------
  config : Metadata describing the dataset name, download link etc.
    
  

  
  """
  def __init__(self, config):
        self.config = config

  @abstractmethod 
  def _download(self):
      # method to download the required csv files, if not found locally.
      pass
  
  @abstractmethod
  def check_path(self):
       # name of the csv files will be present in the config info, fetch it and check if the working directory contains the csv files
       pass

  @abstractmethod
  def _clean(self):
     # A method to clean the dataset(s)
    pass

  @abstractmethod
  def load(self):
    """  A method to load the csv files into dataframes."""
    pass




