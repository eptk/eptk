# -*- coding: utf-8 -*-

from eptk.models.base import BasePredictor
from catboost import CatBoostRegressor
import copy


class CatBoostPredictor(BasePredictor):
   """
   Parameters
   -----------
   loss_function : str
   The metric to use in training. The specified value also determines the machine learning problem to solve.

   custom_metric: str
   Metric values to output during training. These functions are not optimized and are displayed for informational purposes only.

   eval_metric: str
   The metric used for overfitting detection (if enabled) and best model selection.

   num_boost_round, n_estimators, num_trees: int
   The maximum number of trees that can be built when solving machine learning problems.
   When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.

   learning_rate: float
   The learning rate.
   Used for reducing the gradient step.

   random_seed: The random seed used for training.

   l2_leaf_reg (reg_lamda) : Coefficient at the L2 regularization term of the cost function.
   Any positive value is allowed.

   bootstrap_type: string
   Bootstrap type. Defines the method for sampling the weights of objects.
       Supported methods:
          - Bayesian
          - Bernoulli
          - MVS
          - Poisson (supported for GPU only)
          - No
   
   bagging_temperature: float
   Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.
   Use the Bayesian bootstrap to assign random weights to objects.
   The weights are sampled from exponential distribution if the value of this parameter is set to “1”. All weights are equal to 1 if the value of this parameter is set to “0”.

   subsample: float
   Sample rate for bagging. 

   boosting_type: str
   Boosting scheme.
        Possible values:
        Ordered — Usually provides better quality on small datasets, but it may be slower than the Plain scheme.
        Plain — The classic gradient boosting scheme.


   boost_from_average: Boolean (True or False)

   score_function: The score type used to select the next split during the tree construction.  

   early_stopping_rounds: int
   Sets the overfitting detector type to Iter and stops the training after the specified number of iterations since the iteration with the optimal metric value.

   task_type: The processing unit type to use for training.
          Possible values:
                - CPU (Default)
                - GPU

   see: https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
   for more details.
   """
   def __init__(self, loss_function = 'RMSE', **kwargs):
        self.loss_function = loss_function
        self.kwargs = kwargs
        #remove the kwargs ( in the self.kwargs) not supported by the Catboost model.       
        self.model = CatBoostRegressor(loss_function = self.loss_function,**self.kwargs)
       
        
   def fit(self, X, y):
           
            """
            Fit CatBoost regressor model. 
        
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

