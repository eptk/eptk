import pytest
import pandas as pd
import numpy as np
from eptk.evaluation import *  # import all the functions
from eptk.metrics import *
from eptk.models.classical import LinearRegressionPredictor
import datetime

"""
Unit tests for the evaluation module
"""

# Dummy_data
@pytest.fixture
def dummy_data():
    df = pd.DataFrame()
    df["meter"]  = np.random.randint(1,4,1000)
    df["meter_reading"] = np.random.randint(10,100,1000)
    df["feature_1"] = np.random.randint(10,100,1000)
    df["feature_2"] = np.random.randint(0,50,1000)
    n_days = 1000
    # today's date in timestamp
    base = pd.Timestamp.today()
    # calculating timestamps for the next 10 days
    timestamp_list = [base + datetime.timedelta(days=x) for x in range(n_days)]
    df["timestamp"] = timestamp_list
    df["groups_for_cv"] = np.random.randint(0, 24, 1000)
    return df


def test_simple_evaluator(dummy_data):
    Y = dummy_data["meter_reading"]
    X = dummy_data.drop(columns=["meter_reading"])
    l_model = LinearRegressionPredictor()
    l_model.fit(X, Y)
    evaluation = simple_evaluator(X, Y, l_model,
                                  [mean_squared_error, mean_absolute_error, mean_absolute_percentage_error])

    assert [mean_squared_error(Y,l_model.predict(X)),mean_absolute_error(Y,l_model.predict(X)), mean_absolute_percentage_error(Y,l_model.predict(X))] == evaluation


def test_EptkCrossValidator(dummy_data):
    min_size = 5
    test_period = 3
    Y = dummy_data["meter_reading"]
    X = dummy_data.drop(columns=["meter_reading"])
    expected_train = [i for i in range(min_size)]
    expected_test = [i + min_size for i in range(test_period)]
    cv = EptkCrossValidator(test_period, min_size)
    for train, test in cv.split(X, groups=X["groups_for_cv"]):  # train and test are the indexes for each split.
        training_data = X[X.index.isin(train)]
        testing_data = X[X.index.isin(test)]
        train_groups = training_data["groups_for_cv"].unique()
        test_groups = testing_data["groups_for_cv"].unique()
        train_groups.sort(), test_groups.sort()
        print("Expected : ")
        print(expected_train,expected_test)
        test_value = list(train_groups) == expected_train and list(test_groups) == expected_test
        if test_value == False:
            assert test_value
        print(list(train_groups) == expected_train and list(test_groups) == expected_test)
        expected_train.append(expected_train[-1] + 1)
        expected_test.append(expected_test[-1] + 1)
        expected_test.pop(0)
    assert True

def test_get_slice(dummy_data):
    indices = [1, 2, 3, 4, 10, 20, 50,0,13]
    X = get_slice(dummy_data,indices)
    indices.sort()
    assert  list(X.index) == indices







