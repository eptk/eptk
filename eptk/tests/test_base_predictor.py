import pytest
from eptk.models.base import BasePredictor
from eptk.models.nn import FFNNPredictor

"""
Unit tests for the base predictor class.  
"""


@pytest.fixture
def DummyModel():
    """
    A dummy model class which inherits the Basepredictor. Used for testing the parent functionality.
    """

    class DummyModel(BasePredictor):
        def __init__(self, attr_1="string_1", attr_2=3):
            self.attr_1 = attr_1
            self.attr_2 = attr_2
            self.weight = 1

        def fit(self):
            # training updated the weight (learned parameter)
            self.weight = 100

        def predict(self):
            pass

    return DummyModel


def test_get_params(DummyModel):
    """
  Testing the get_params method. It should return all the model attributes.
  """
    model = DummyModel()
    assert model.get_params() == {'attr_1': 'string_1', 'attr_2': 3}


def test_set_params(DummyModel):
    """
  Testing the ability to update the params.
  """

    model = DummyModel()
    model.set_params(attr_1="changed")
    assert model.get_params() == {'attr_1': 'changed', 'attr_2': 3}


def test_reset(DummyModel):
    """
  Reset method should reset the model to forget any learned parameter while resetting it with the initial configuration.
  """
    inital_config = {'attr_1': 'default', 'attr_2': 0}
    model = DummyModel(**inital_config)
    model.fit()  # model.weight is updated to 100
    model = model.reset()
    assert model.weight == 1
    pass


def test_load_params():
    """
  Testing loading of configuration from the existing configuration jsons.
  """

    model = FFNNPredictor()
    model = model.load_params("ffnn_cHaOs_3.json")
    assert model.get_params() == {'activation': ['relu', 'tanh', 'tanh'], 'batch_size': 1000, 'dropout': 0.2,
                                  'epochs': 31, 'hidden_layers': [128, 32, 4]}
