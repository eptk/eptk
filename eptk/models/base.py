# -*- coding: utf-8 -*-

from collections import defaultdict
from abc import ABC,abstractmethod
import inspect
import os
import json
from os import environ, listdir, makedirs
from os.path import dirname, expanduser, isdir, join, splitext




"""Base class for all the energy predictors."""

class BasePredictor(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Fit predictor.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        y : ndarray of shape (n_samples,)
            The input target value. ( meter readings)


        Returns
        -------
        self : object
            Fitted estimator.

        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        pass

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the predictor"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for the predictor.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of the predictor.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


    
    def reset(self):
       """A method for reseting the predictor"""   
       new = self.__class__(**self.get_params())
       return new

    
    
    def load_params(self, json_file):
      """A method to load kaggle competition configurations.
      
      Parameters
      -----------
      json_file : str
      Name of the json file containing the configuration.

      Returns
      ---------
      A new model instance with the new parameters.  

      """

      module_path = dirname(__file__)
      with open(join(module_path, "model_configurations", json_file)) as json_data:
         config = json.load(json_data)
      
      link = config["link"]
      m = config["model"]
      print(f"model : {m}")
      print(f"reference:{link}" )
      self = self.__class__(**config["params"])
      print("params loaded")
      
      return self
      
      
      

    
    
    
    
    

