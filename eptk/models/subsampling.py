# -*- coding: utf-8 -*-
from .base import BasePredictor
from sklearn.utils import check_array
import numpy as np
import pandas as pd
from eptk.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from ..utils import get_slice
import numpy as np
import pandas as pd
from .classical import RidgeRegressionPredictor
import copy
import warnings


class SubsamplingPredictor(BasePredictor):
    """
    Ensembler for subsampling the datasets based on categorical features and training multiple predictors of the same type (with same hyper-parameters)  on each of the subsets seperately.
    The final model takes the weighted sum of the predictions made by each of the predictors, which were trained on different subset of the data.

    Parameters
    -----------
    predictors: A list of predictor objects
    A list containing all the unfitted predictors.

    weights: An array of weights or str or None (Default)
    Options:-
             "uniform" : Gives equal importance to the prediction from all the models. (Default)
               None    : No weights are assigned. Weights are to be optimized while fitting the model.

     group by: str or list
     A column name or a list of column names on basis of which the subsampling is done.
     Each combinations of these columns gives mutually exclusive and exhaustive subsets of the dataset. On these subsets,the predictors are trained.
     The final predictor takes the weighted sum of all the predictors.

    technique: str
        Select one from  the techniques provided as options.
        Options : {"sparse", "meta_regressor", "bayesian_opt" }
             For all the techniques the training data is split into 80:20. All indvidual predictors are fitted on the 80% of the data.
             Then the 20% of data is considered for optimizing the weights. The split is not random and it takes into consideration that 80% data of any given subset
             is available for training.
             After the weights are optimized. All the predictors are fitted to the entire training data.
            1. sparse (prop_to_performance): After all the predictors are fitted to the 80% data. Their performance is evaluated over the remaining 20% data. The
            metric used is mean_squard_error. The models are assigned weights based on the performance. A numerically stable softmax function is used to assign weights.
            The negative of mean square error is taken as the input for the softmax.

            2. meta_regressor:  After all the predictors are fitted to the 80% data. The coefficents of the regresion of true value of target over the vector
            of predictons of each indvidual model is taken as the weights.
                            ( Y = W.P + error )

            3. bayesian_opt: The weights are optimized using gp regression. First the black box cost function over the weight space is created.
            All the weights are between 0 and 1. We take negative of mean_square_error of the weighted predictions with the actual target values as the function to maximize.
            After optimization we get the vector of weights.



    """

    def __init__(self, group_by, predictor=None, weights=None, optimize_weights=False,technique="bayesian_opt",optimizer_valid_split = 0.2):
        self.group_by = group_by
        self.predictor = predictor
        self.predictors = {}
        self.weights = weights
        self.optimize_weights = optimize_weights
        self.technique = technique
        self.optimizer_valid_split = optimizer_valid_split
        if self.weights is None:
            print("Proceeding without a weight array.")

    def _optimize_weights(self, X, y):
        # we need to make sure each of the groups is represented in the 80% split
        indices_groups = {}
        g = X.groupby(self.group_by)
        for value, value_df in g:
            indices_groups[value] = value_df.index.tolist()

        train_index = []
        test_index = []
        for group, indices in indices_groups.items():
            train_index.extend(indices[:int(len(indices) * (1 - self.optimizer_valid_split))])
            test_index.extend(indices[int(len(indices) * (1-self.optimizer_valid_split)):])

        X_train = get_slice(X, train_index)
        y_train = get_slice(y, train_index)
        X_test = get_slice(X, test_index)
        y_test = get_slice(y, test_index)

        # training
        X_train.reset_index(drop=True, inplace=True)
        temp = X_train.groupby(self.group_by)
        for value, value_df in temp:
            indices = value_df.index.tolist()
            y_ = y_train[indices]
            # remove the column used for grouping
            value_df = value_df.drop(self.group_by, axis=1)
            predictor = self.predictor.reset()
            predictor.fit(value_df, y_)
            self.predictors[f"group-{str(value)}"] = predictor

            # predictions
        predictions = []
        X_test = X_test.drop(self.group_by, axis=1)
        for i in self.predictors:
            predictions.append(self.predictors[i].predict(X_test))

        if self.technique == "meta_regressor":
            # making a matrix for trainning
            predictions = np.array(predictions)
            predictions = predictions.transpose()
            meta = RidgeRegressionPredictor(fit_intercept=False)
            meta.fit(predictions, y_test)
            self.weights = meta.model.coef_
            print(f"Weights: {self.weights}")


        elif self.technique == "sparse":
            performance = []
            for pred in predictions:
                performance.append(-mean_squared_error(pred, y_test))

                # numerically stable softmax

            def softmax(x):
                """Compute softmax values for each sets of scores in x."""
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0)

            self.weights = softmax(performance)
            print(f"Weights: {self.weights}")

        elif self.technique == "bayesian_opt":
            # setting the bounds
            pbounds = {}
            for i in range(len(self.predictors)):
                pbounds["w" + str(i + 1)] = (0.001, 1)

            # maximize the negative mse
            def cost_function_over_weight_space(**weights):
                p = np.array(predictions)
                w = np.array(list(weights.values()))
                sum = np.sum(w)
                y_hat = 1 / sum * np.dot(w, p)  # normalized weights cost func
                return -(mean_squared_error(y_test, y_hat))

            optimizer = BayesianOptimization(f=cost_function_over_weight_space, pbounds=pbounds, random_state=1)
            optimizer.maximize(init_points=20, n_iter=40)

            negativeloss = [i["target"] for i in optimizer.res]
            index = negativeloss.index(max(negativeloss))
            weights = np.array(list(optimizer.res[index]["params"].values()))
            self.weights = 1 / sum(weights) * weights  # normalize
            print(f"Weights: {self.weights}")


        else:
            raise AttributeError("Invalid value for attribute technique")

        # reset predictors converted to sklearn
        print(self.predictors)
        for i in self.predictors:
            print(self.predictors[i], type(self.predictors[i]))
            self.predictors[i] = self.predictors[i].reset()
        return self.weights

    def fit(self, X, y, **kwargs):

        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
         Training data. The column "timestamp" will be removed if it is found. (When X is a pandas dataframe)

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

        optimize_weights : Boolean (Deafault =False)
        True or False. If set to True, the weights are optimized by using one of the 3 techniques.


        """

        try:  # if X is a pandas object with timestamp column
            if "timestamp" in X.columns:
                X = X.drop("timestamp", axis=1)
        except:
            pass

        X.reset_index(drop=True, inplace=True)
        if self.optimize_weights == True:
            self._optimize_weights(X, y)

        y = check_array(y, ensure_2d=False)
        temp = X.groupby(self.group_by)
        for value, value_df in temp:
            print(f"fitting {self.group_by} : {value}")
            indices = value_df.index.tolist()
            y_ = y[indices]
            # remove the column used for grouping
            value_df = value_df.drop(self.group_by, axis=1)
            predictor = self.predictor.reset()
            self.predictors[f"group-{str(value)}"] = predictor.fit(value_df, y_, **kwargs)

        del temp

        return self.predictors

    def predict(self, X):
        """

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)

        Returns
        --------
        An array of ensembler estimates for input X.

        """
        try:
          X = X.copy()
        except:
            warnings.warn("Couldnt make a copy of the input dataframe. The dataframe would be modified.")

        try:  # if X is a pandas object with timestamp column
            if "timestamp" in X.columns:
                X = X.drop("timestamp", axis=1)
        except:
            pass

        if type(self.group_by) == str:
          if self.group_by in (X.columns):
             X.drop(self.group_by, axis=1, inplace=True)
        else:
            for x in self.group_by:
              if x in (X.columns):
                X.drop(x, axis=1, inplace=True)



        predictions = []
        for m in self.predictors:
            predictions.append(self.predictors[m].predict(X))

        predictions = np.array(predictions)

        if self.weights is "uniform":
            try:
                self.weights = 1 / len(self.predictors) * np.ones(len(self.predictors))
                print(f"uniform weights assigned : {self.weights}")
            except ZeroDivisionError:
                raise ValueError("The list of predictors cannot be empty")

        return np.dot(self.weights, predictions)

    def reset(self):
        """A method to reset the subsampler. It prevents some models from remembering training from previous iteration while cross validation."""
        a = copy.deepcopy(self.__dict__)
        a.pop("predictors")
        new = self.__class__(**a)
        return new








