import pandas as pd
import numpy as np
import pytest
from eptk.preprocessing.clean import *  # import all the functions
import datetime

"""
Unit tests for the dataset cleaning methods.  
"""


@pytest.fixture
def dummy_data_1():
    df = pd.DataFrame()
    n_days = 100
    # today's date in timestamp
    base = pd.Timestamp.today()
    # calculating timestamps for the next 10 days
    timestamp_list = [base + datetime.timedelta(days=x) for x in range(n_days)]
    df["timestamp"] = timestamp_list

    df["meter"] = np.random.randint(1, 4, 100)
    df["meter_reading"] = np.random.randint(10, 100, 100)
    df.loc[0:14, "meter_reading"] = np.random.randint(1, 5, 15)  # readings below 5
    df.loc[0:10, "meter"] = 1
    df.loc[10:14, "meter"] = 2
    df.loc[20:30, "meter_reading"] = pd.NA  # missing readings
    df.loc[70:100, "meter_reading"] = 10  # constant reading
    df["feature_1"] = np.random.randint(10, 100, 100)
    df.loc[50:, "feature_1"] = pd.NA  # feature 1 is 50% empty
    return df


def test_remove_missing_meter_reading(dummy_data_1):
    after = remove_missing_meter_reading(dummy_data_1)
    assert len(dummy_data_1) == 100 and len(after) == 89 and after["meter_reading"].isna().any() == False


def test_remove_readings_threshold_without_meter_select(dummy_data_1):
    dummy_data_1 = remove_missing_meter_reading(dummy_data_1)
    after = remove_readings_threshold(dummy_data_1, 6)
    assert (dummy_data_1["meter_reading"] > 6).all() == False and (after["meter_reading"] > 6).all()


def test_remove_readings_threshold_with_meter_select(dummy_data_1):
    dummy_data_1 = remove_missing_meter_reading(dummy_data_1)
    after = remove_readings_threshold(dummy_data_1, 6, 1)
    assert (dummy_data_1["meter_reading"] > 6).all() == False and (after["meter_reading"] > 6).all() == False and (
                after[after["meter"] == 1]["meter_reading"] > 6).all()


def test_remove_const_reading(dummy_data_1):
    dummy_data_1 = remove_missing_meter_reading(dummy_data_1)
    after = remove_constant_reading(dummy_data_1, 2)  # any two consecutive entries for all the meters for duration = 2
    temp = after.groupby("meter")
    val = False
    for m, m_df in temp:
        if (m_df["meter_reading"].shift(1) - m_df["meter_reading"] != 0).all():
            val = True
        else:
            val = False
            break
    assert val
