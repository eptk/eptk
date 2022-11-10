# -*- coding: utf-8 -*-
"""A module for energy predictor evaluation"""

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_array, _num_samples, column_or_1d
from sklearn.utils import indexable
from .utils import get_slice, absolute_quantity
import pickle
import os
from os.path import dirname, join



def prepare_data(dataset, drop_features=[], remove_na=True, target_feature="meter_reading", remove_timestamp=False):
    """

    Parameters
    -----------
    dataset: a pandas dataframe
    A pandas dataframe on which the models are to be trained.

    drop_features: a list
    Provide a list of features to be removed.

    remove_na: Boolean (True or False)
    If remove_na is true, removes all the rows with atleast one missing value.

    target_feature: str
    column name of the target feature (default= "meter_reading")

    remove_timestamp: Boolean (True or False)
    If remove_timestamp == True, if the dataframe has a timestamp column, it gets removed.

    Returns
    --------
    (X,y)

    X : dataframe (n_samples, n_features)
    Feature set.

    y :  array-like of shape (n_samples,)
    The target value for dataset.

    """

    # remove missing rows
    if remove_na == True:
        dataset = dataset.copy()
        dataset.dropna(inplace=True)

    if target_feature not in dataset.columns:
        raise ValueError(f"The target feature {target_feature} is not present in the dataset")
    else:
        y = dataset[target_feature]
        drop_features.extend([target_feature])
    if 'timestamp' in dataset.columns and remove_timestamp == True:
        drop_features.extend(["timestamp"])

    X = dataset.drop(columns=drop_features)
    X.reset_index(drop=True, inplace=True)
    y = check_array(y, ensure_2d=False)

    return X, y


def simple_evaluator(X, y_true, model, metric):
    """
    Parameters
    -----------
     X : {array-like, sparse matrix} of shape (n_samples, n_features)
         Testing data.

    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
         Ground truth (correct) target values.

    model: A trained (fitted on training set)  model.

    metric: A metric method to evaluate the model. (or a list of metric methods)
    It can be imported from eptk.metrics
            example: eptk.metrics.root_mean_squared_error
   It can be a list of metric methods. [m1,m2,m3 ..mn]

    Returns
    -------

    loss : float or list of floats
         float - A non-negative floating point value (the best value is 0.0)
         It is returned for a single metric method input.
         list of float -  A list of non-negative floats, scores for each metric
         example:-
          metric input - [m1, m2, m3 ...mn] --> output - [score(m1), score(m2), (score m3) ...score(mn)]

    """
    y_pred = model.predict(X)

    if type(metric) == list:
        score = []
        for m in metric:
            score.append(m(y_true, y_pred))
        return score

    return metric(y_true, y_pred)


class EptkCrossValidator(BaseCrossValidator):
    """base class for crossvalidatior"""

    def __init__(self, test_period, min_train_size=3):
        self.test_period = test_period
        self.min_train_size = min_train_size

    def _iter_masks(self, X, y=None, groups=None):
        n_bins = len(np.unique(groups))
        if n_bins <= self.test_period:
            raise ValueError(
                "test_period={} must be strictly less than the number of different groups={}".format(self.test_period,
                                                                                                     n_bins)
            )
        groups = check_array(groups, ensure_2d=False)  # checks for missing value and converts into suitable format
        bins = np.unique(groups)
        i = 0
        while self.min_train_size + i + self.test_period <= n_bins:
            print(
                f"train:{bins[:self.min_train_size + i]}, test:{bins[self.min_train_size + i: self.min_train_size + i + self.test_period]}")
            testgroups = bins[self.min_train_size + i: self.min_train_size + i + self.test_period]
            traingroups = bins[:self.min_train_size + i]
            test_mask = np.zeros(len(groups), dtype=bool)
            train_mask = np.zeros(len(groups), dtype=bool)
            for j in testgroups:
                test_mask[groups == j] = True
            for j in traingroups:
                train_mask[groups == j] = True
            i = i + 1

            yield test_mask, train_mask

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index, train_index in self._iter_masks(X, y, groups):
            train_index = indices[train_index]
            test_index = indices[test_index]
            yield train_index, test_index

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""

        if X is None:
            raise ValueError("The 'X' parameter should not be None.")

        if len(groups) != len(X):
            raise ValueError("The 'groups' parameter should be an array equal to the len of X.")

        return len(np.unique(groups)) - self.min_train_size - 1


def _cross_validation_scores(X, y, groups, model, metric, test_period, verbose=False, start_iteration=0, store=False,
                             title=None, time_bin_type="month", _scores=[], min_train_size=3):
    """
     Parameters
     ----------
       X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
       y : array-like of shape (n_samples,)
            The target variable for energy prediction problems.

       groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.  (len(Group) == n_samples of X)
            for examples: month, week etc. (time groups)

       model: model object.
       A model object with fit() and predict() methods.

       metric: metric method
       A metric method to evaluate the model performance.

       test_period: int
       An integer less than the total number of groups.

       verbose: Boolean (True or False)

       start_iteration: int
       The iteration from which the validator resumes the evaluation (useful for storing the cross validation progress)

       store: Boolean (True or False) Default = False
       Stores all the progress made by the cross validator in the working directory.The files are stored in the working directory inside the directory "/configuration/cross_validation/title. The title is passed by the parameter title.
        If the path is not available in the working directory, first the directories of the path are created.

       min_train_size: int
       minimum size of the training data (at first iteration)

       title: str or None (Default = None)
        Creates a directory with title name For storing.


       Returns
       ---------

       scores: an array of model error scores over all the iterations.


       Description
       -------------

       A method to evaluate model performance over the dataset. The data is split into groups (time groups), at each iteration, test_period number
       of consecutive groups are selected as test set for the model evaluation, and the training group is selected as per the forward chaining.
       At each iteration, the model is trained and tested using an error function (metric). The evaluated score is added to the list of scores.
       The list of scores is returned as the final output.


    """
    # cross validation object requires parameter : test_period
    cv = EptkCrossValidator(test_period, min_train_size)

    # scores keep the list of scores for all the train_test iteration
    scores = _scores
    y_ = check_array(y, ensure_2d=False)  # check and converts target into array format.
    # iteration:
    iteration = 0

    for train, test in cv.split(X, y, groups):
        iteration = iteration + 1
        if start_iteration < iteration:
            # only perform fit/predict if start_iteration >= current iteration
            model.fit(get_slice(X, train), y_[train])
            score = simple_evaluator(get_slice(X, test), y_[test], model, metric)
            if verbose:
                print(f"score : {score}")
            scores.append(score)
            if store == True:
                # current working directory
                current = os.getcwd()
                temp_path = join(current, "temp")
                # if not found create temp in wd
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)

                config = {"start_iteration": iteration, "test_period": test_period, "time_bin_type": time_bin_type,
                          "scores": scores, "title": title, "metric": metric,
                          "model": model.reset(), "min_train_size": min_train_size}
                store_predict_target = model.predict(get_slice(X, test)), y_[test]

                with open(join(temp_path, f"{title}_Iteration_{iteration}"), "wb") as f:
                    pickle.dump(store_predict_target, f)
                try:
                   with open(join(temp_path, f"{title}_config"), "wb") as f:
                      pickle.dump(config, f)
                except TypeError:
                    del config["model"]
                    print("Cant pickle the model. Continue without storing the model")


            try:
                # reset model, if the model has reset method (all eptk predictors have reset method)
                model = model.reset()
                print("resetting ...")
            except:
                print("Warning: model cannot be reset. Make sure it has the reset method")
    return scores


# cross validation score function

class CrossValidation():
    """
    Parameters
    ----------
          X : array-like of shape (n_samples, n_features)
              Training data, where n_samples is the number of samples
              and n_features is the number of features.
          y : array-like of shape (n_samples,)
              The target variable for energy prediction problems.

          time_bin_type : str
          ["week" , "month"]
          group the data according to the time bins (weeks, months)

          model: model object.
          A model object with fit() and predict() methods.

          metric: metric method or a list of metric methods
          A metric method to evaluate the model performance.
          It could also be list of metric methods, to simultaneously evaluate the model with multiple metrics at a time.


          test_period: int
          An integer less than the total number of groups.

          verbose: Boolean (Default = False)
          Prints the score at each iteration.

          store: Boolean (Default = False)
          If set to True, the progress made by the crossvalidator is saved after each iteration. Along with the predictions made and a config file.
          The files are stored in the /eptk/temp directory.


          title: str or None (Default = None)
          The title given for storing the progress made by the cross validator.
          Example- If title = linear_reg, then inside the /temp directory,
          the configuration file is stored as linear_reg_config.
          And the kth iteration predictions are stored as linear_reg_Iteration_k.
          Note - Both are stored as pickle objects
    """

    def __init__(self, model=None, metric=None, time_bin_type="month", test_period=2, verbose=False, store=False,
                 title=None, min_train_size=3):

        self.X = None
        self.y = None
        self.model = model
        self.metric = metric
        self.time_bin_type = time_bin_type
        self.test_period = test_period
        self.verbose = verbose
        self.store = store
        self.scores = []
        self.start_iteration = 0
        self.title = title
        self.min_train_size = min_train_size

    def load(self, title):
        """A method to load the cross_validator configuration.

        Parameters
        ------------
        title: str
        A title given while storing the configurations.

        """

        current = os.getcwd()

        try:
            with open(join(current, "temp", f"{title}_config"), "rb") as config:
                configuration = pickle.load(config)
        except:
            print(f"Failed to load config {title}_config. Make sure the title parameter set is appropriate. \n "
                  f"Another possibility is that the config pickle file is empty. In the next iteration the configs "
                  f"will be saved as json files.")
            raise ImportError

        print(f"Configurations loaded : {configuration}")
        try:
         self.model = configuration["model"]
        except:
            print("couldnt open the stored model. Fetching the model if provided while initializing the class")
            print(f"{self.model.__class__()}")
        self.metric = configuration["metric"]
        self.time_bin_type = configuration["time_bin_type"]
        self.test_period = configuration["test_period"]
        self.title = configuration["title"]
        self.start_iteration = configuration["start_iteration"]
        self.scores = configuration["scores"]
        self.min_train_size = configuration["min_train_size"]
        return self

    def evaluate(self, X, y):
        """
        Parameters
        ----------
        X : Pandas dataframe

        y : array-like of shape (n_samples,)
        The target variable for energy prediction problems.

        Returns
        ----------
        mean score over interactions: float or list of float
        returns the avg model evaluation score over each iteration. Thr score is a list if the evaluation metric is
        a list of metrics.
        """
        self.X = X
        self.y = y
        self.groups = absolute_quantity(self.X["timestamp"], self.time_bin_type)

        # remove timestamp column
        if "timestamp" in self.X.columns:
            self.X = self.X.drop(columns=["timestamp"])
        print(f"scores, if previously calculated : {self.scores}")
        self.scores.extend(
            _cross_validation_scores(self.X, self.y, self.groups, self.model, self.metric, self.test_period,
                                     self.verbose, self.start_iteration, self.store, self.title, self.time_bin_type, _scores= self.scores,
                                     min_train_size=self.min_train_size))
        return np.mean(self.scores, axis=0)


