# -*- coding: utf-8 -*-

from .linear import LinearRegressionPredictor
from .linear import LassoRegressionPredictor
from .linear import RidgeRegressionPredictor
from .linear import ElasticNetPredictor
from.linear import SGDRPredictor
from .knn import KNNPredictor
from .svr import SVRPredictor

__all__ = [
    "LinearRegressionPredictor",
    "LassoRegressionPredictor",
    "RidgeRegressionPredictor",
    "KNNPredictor",
    "SVRPredictor",
    "SGDRPredictor",
    "ElasticNetPredictor"
    ]