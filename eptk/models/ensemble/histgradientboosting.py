# -*- coding: utf-8 -*-

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from ..base import BasePredictor

class HistGradientBoostingPredictor(BasePredictor):
    """Histogram-based Gradient Boosting Regression Tree.
    Parameters
    ----------
    loss : {'squared_error', 'absolute_error', 'poisson'}, default='squared_error'
        The loss function to use in the boosting process. Note that the
        "least squares" and "poisson" losses actually implement
        "half least squares loss" and "half poisson deviance" to simplify the
        computation of the gradient. Furthermore, "poisson" loss internally
        uses a log-link and requires ``y >= 0``
    
    learning_rate : float, default=0.1
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.

    max_iter : int, default=100
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees.

    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.

    max_depth : int or None, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.

    min_samples_leaf : int, default=20
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.

    l2_regularization : float, default=0
        The L2 regularization parameter. Use ``0`` for no regularization
        (default).
    
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.
    categorical_features : array-like of {bool, int} of shape (n_features) \
            or shape (n_categorical_features,), default=None.
        Indicates the categorical features.
        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.
        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be in [0, max_bins -1].
    
    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonic constraint to enforce on each feature. -1, 1
        and 0 respectively correspond to a negative constraint, positive
        constraint and no constraint. 

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
        See :term:`the Glossary <warm_start>`.

    early_stopping : 'auto' or bool, default='auto'
        If 'auto', early stopping is enabled if the sample size is larger than
        10000. If True, early stopping is enabled, otherwise early stopping is
        disabled.
    
    scoring : str or callable or None, default='loss'
        Scoring parameter to use for early stopping. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used. If
        ``scoring='loss'``, early stopping is checked w.r.t the loss value.
        Only used if early stopping is performed.

    validation_fraction : int or float or None, default=0.1
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping. If None, early stopping is done on
        the training data. Only used if early stopping is performed.

    n_iter_no_change : int, default=10
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance. Only used if early stopping is performed.

    tol : float, default=1e-7
        The absolute tolerance to use when comparing scores during early
        stopping. The higher the tolerance, the more likely we are to early
        stop: higher tolerance means that it will be harder for subsequent
        iterations to be considered an improvement upon the reference score.

    verbose : int, default=0
        The verbosity level. If not zero, print some information about the
        fitting process.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled.
        Pass an int for reproducible output across multiple function calls.
 


    see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
    for more details.
    """ 

    def __init__(self, l2_regularization = 0.0,
                 learning_rate= 0.1,
                 loss = 'least_squares',
                 max_bins = 255,
                 max_depth = None,
                 max_iter = 100,
                 max_leaf_nodes = 31,
                 min_samples_leaf = 20,
                 n_iter_no_change = 50,
                 random_state = None,
                 scoring = None,
                 tol = 1e-07,
                 validation_fraction =0.1,
                 verbose = 0,
                 warm_start = False):
    
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_bins = max_bins
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.scoring = scoring
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        self.warm_start = warm_start


        self.model = HistGradientBoostingRegressor(l2_regularization = self.l2_regularization,
                 learning_rate= self.learning_rate,
                 loss = self.loss,
                 max_bins = self.max_bins,
                 max_depth = self.max_depth,
                 max_iter = self.max_iter,
                 max_leaf_nodes = self.max_leaf_nodes,
                 min_samples_leaf = self.min_samples_leaf,
                 n_iter_no_change = self.n_iter_no_change,
                 random_state = self.random_state,
                 scoring = self.scoring,
                 tol = self.tol,
                 validation_fraction = self.validation_fraction,
                 verbose = self.verbose,
                 warm_start = self.warm_start )

    def fit(self, X, y):
            
            """
            Fit model. 
        
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
             Training data. The column "timestamp" will be removed if it is found. (When X is a pandas dataframe) 
            
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values."""


        
            self.model.fit(X,y)
            return self.model

    def predict(self, X):
           
            """
            Parameters
            ----------
            X : array-like or sparse matrix, shape (n_samples, n_features)
            
            Returns
            --------
            An array of model estimates for input X.

            """


            return self.model.predict(X)



    def set_params(self,**params):
            
            """
            Set the parameters of this predictor.
        
            Parameters
            ----------
            **params : dict
            Predictor parameters.
            """            


            #first set all the parameters
            super().set_params(**params)
            #remove the parameters from params not require for the sklearn HistGradientBoostingRegressor
            self.model.set_params(**params)
    







 