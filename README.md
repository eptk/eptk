# eptk
Energy Prediction Toolkit (eptk) is a python package for implementing and benchmarking energy use prediction models on a collection of large datasets using standard performance metrics. The package includes a variety of predictive models along with a set of configurations that were picked from the top performers in the ASHRAE - Great Energy Predictor III competition hosted on Kaggle.
 The package provides methods for engineering additional features (temporal, weather and rolling stats) from the datasets. The package also provides ensembling techniques such as meta-regressors, Bayesian optimization and subsampling to combine multiple models. The custom cross validator which is used for benchmarking can resume evaluation from a stored checkpoint in the event of failure during runtime. The datasets are first downloaded in the working directory if not found while calling the load method.

## Installation guide (pip)
```
pip install git+https://github.com/eptk/eptk.git
```

## Getting started

Run the following python script, in order perform the following tasks.
- Load the dataset
- Perform basic feature engineering 
- Load a model and two evaluation metrics
- Benchmarking with forward chaining crossvalidation with the selected metrics

```
# Imports
from eptk.dataset import ASHRAE_GEP3
from eptk.utils import merge_data
from eptk.preprocessing.feature_extraction import add_weather_features, add_temporal_features, include_holidays, category_to_numeric
from eptk.evaluation import prepare_data, CrossValidation
from eptk.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error, mean_squared_error
from eptk.models.ensemble import LightGBMPredictor
# 1. Dataset loading
ashrae = ASHRAE_GEP3()
meta, meter, weather = ashrae.load(remove_electric_zero = True) 
# 2. Feature engineering
weather = add_temporal_features(weather, cyclic = True)
dataset = merge_data(meter_df = meter, meta_df = meta, weather_df = weather)
# 3. final preparations : Seperate the dataset into X (features) and Y (Meter readings)
X,y = prepare_data(dataset, remove_timestamp = False, drop_features = ["primary_use"])
model = LightGBMPredictor()

metrics_used = [root_mean_squared_log_error,mean_absolute_error]
cv = CrossValidation(model = model,metric = metrics_used,time_bin_type="month",verbose= True,test_period =3,title = "lgbm")
results = cv.evaluate(X,y) 
```
