# -*- coding: utf-8 -*-

from eptk.models.base import BasePredictor
from lightgbm import LGBMRegressor
import copy


class LightGBMPredictor(BasePredictor):
   
   """
   Parameters
   ----------
        boosting_type : string, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'goss', Gradient-based One-Side Sampling.
            'rf', Random Forest.
        
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
        
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        
        objective : string, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (hessian) needed in a child (leaf).
        
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        
        n_jobs : int, optional (default=-1)
            Number of parallel threads.
        
        silent : bool, optional (default=True)
            Whether to print messages while running boosting.
        
        importance_type : string, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        
        **kwargs
            Other parameters for the model.
            
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.
   
   
   
   
   
   """
   
   def __init__(self,*,
        boosting_type = 'gbdt',
        class_weight = None,
        colsample_bytree = 1.0,
        importance_type = 'split',
        learning_rate = 0.1,
        max_depth = -1,
        min_child_samples = 20,
        min_child_weight = 0.001,
        min_split_gain = 0.0,
        n_estimators = 100,
        n_jobs = -1,
        num_leaves = 31,
        objective = None,
        random_state = None,
        reg_alpha = 0.0,
        reg_lambda = 0.0,
        silent = True,
        subsample = 1.0,
        subsample_for_bin = 200000,
        subsample_freq = 0,
        **kwargs
        ):
        
      
        self.boosting_type = boosting_type
        self.class_weight = class_weight
        self.colsample_bytree = colsample_bytree
        self.importance_type = importance_type 
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.num_leaves = num_leaves
        self.objective = objective
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.silent = silent
        self.subsample = subsample
        self.subsample_for_bin = subsample_for_bin
        self.subsample_freq = subsample_freq
        self.kwargs = kwargs




        self.model =  LGBMRegressor(boosting_type = self.boosting_type,
        class_weight = self.class_weight,
        colsample_bytree = self.colsample_bytree,
        importance_type = self.importance_type, 
        learning_rate = self.learning_rate,
        max_depth = self.max_depth,
        min_child_samples = self.min_child_samples,
        min_child_weight = self.min_child_weight,
        min_split_gain = self.min_split_gain,
        n_estimators = self.n_estimators,
        n_jobs = self.n_jobs,
        num_leaves = self.num_leaves,
        objective = self.objective,
        random_state = self.random_state,
        reg_alpha = self.reg_alpha,
        reg_lambda = self.reg_lambda,
        silent = self.silent,
        subsample = self.subsample,
        subsample_for_bin = self.subsample_for_bin,
        subsample_freq = self.subsample_freq,
        **self.kwargs
        )  #remove the kwargs ( in the self.kwargs) not supported by the lgbm model.
        

   def fit(self, X, y):
            
            """
            Fit LightGBM regressor model. 
        
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
             Training data. The column "timestamp" will be removed if it is found. (When X is a pandas dataframe) 
            
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values."""

            try: #if X is a pandas object with timestamp column
                if "timestamp" in X.columns:
                    X = X.drop("timestamp",axis = 1)
            except:
                pass
            return self.model.fit(X,y)

   def predict(self, X):
          
            """
            Parameters
            ----------
            X : array-like or sparse matrix, shape (n_samples, n_features)
            
            Returns
            --------
            An array of model estimates for input X.

            """



            try: #if X is a pandas object with timestamp column
                if "timestamp" in X.columns:
                    X = X.drop("timestamp",axis = 1)
            except:
                pass
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
            for key, value in params.items():
                setattr(self, key, value)
            
            super().set_params(**params)
            #remove the parameters from params not require for the Xgboost model
            self.model.set_params(**params)

   @staticmethod
   def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z        
   

   def get_params(self, deep = True):
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


       return self.merge_two_dicts(super().get_params(deep), self.model.get_params())
