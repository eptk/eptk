import pytest
import numpy as np
from eptk.models.classical import RidgeRegressionPredictor, LinearRegressionPredictor
from eptk.models import SubsamplingPredictor, Ensembler
import pandas as pd
from eptk.metrics import mean_squared_error

""""
Unit tests for  ensemble predictors.
"""

@pytest.fixture
def dummy_data_1():
    """
    Dummy data to test the ensembler
    """
    X = np.random.randint(10, size=(5, 3))
    y = [3, 2, 1, 3, 1]
    return X, y


def test_Ensembler(dummy_data_1):
    """
    Testing the ensembler without optimizing the weights, weights are taken as [0.3,0.7]. Weighted prediction of the 2 models is considered
    """
    model_1 = RidgeRegressionPredictor()
    model_2 = LinearRegressionPredictor()
    weights = [0.3, 0.7]
    e = Ensembler([model_1, model_2], weights=weights)
    X, y = dummy_data_1
    e.fit(X, y)
    pred = e.predict(X)
    model_1.fit(X, y)
    model_2.fit(X, y)
    pred_1, pred_2 = model_1.predict(X), model_2.predict(X)
    assert mean_squared_error(weights[0] * pred_1 + weights[1] * pred_2, pred) < 0.00001


@pytest.fixture
def dummy_data_2():
    x1 = np.random.randint(5, size=(10))
    x2 = np.random.randint(5, size=(10))
    z = np.random.randint(2, size=(10))
    y = np.random.randint(5, size=(10))
    X = pd.DataFrame()
    X["var_1"] = x1
    X["var_2"] = x2
    X["groups"] = z
    X["target"] = y
    return X


def test_subsampling(dummy_data_2):
    """
    Testing the subsampling predictor without optimizing the weights, weights are taken as [0.3,0.7]. Weighted prediction of the models trained per group is considered
    """
    print(dummy_data_2)
    model = RidgeRegressionPredictor()
    X = dummy_data_2.iloc[:, :-1]
    y = dummy_data_2.iloc[:, -1]
    s_model = SubsamplingPredictor(group_by="groups", predictor=model, weights=[0.3, 0.7])
    s_model.fit(X, y)
    pred = s_model.predict(X)
    temp = dummy_data_2.groupby("groups")
    models = []
    for id, id_df in temp:
        model = RidgeRegressionPredictor()
        models.append(model.fit(id_df.iloc[:, :-2], id_df.iloc[:, -1]))
    X = X.iloc[:, :-1]

    predictions = [model.predict(X) for model in models]
    _pred = 0.3 * predictions[0] + 0.7 * predictions[1]

    assert mean_squared_error(_pred, pred) < 0.00001
