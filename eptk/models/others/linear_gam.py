# -*- coding: utf-8 -*-
from pygam import LinearGAM
from ..base import BasePredictor

class LinearGAMPredictor(BasePredictor):



    """
    Linear GAM
    This is a GAM with a Normal error distribution, and an identity link.
    Parameters
    ----------
    terms : expression specifying terms to model, optional.
        By default a univariate spline term will be allocated for each feature.
        For example:
        >>> GAM(s(0) + l(1) + f(2) + te(3, 4))
        will fit a spline term on feature 0, a linear term on feature 1,
        a factor term on feature 2, and a tensor term on features 3 and 4.
   
    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.
    
    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        Note: the intercept receives no smoothing penalty.
    
    max_iter : int, optional
        Maximum number of iterations allowed for the solver to converge.
    
    tol : float, optional
        Tolerance for stopping criteria.
    
    verbose : bool, optional
        whether to show pyGAM warnings.
    
    see: https://pygam.readthedocs.io/en/latest/
    for more info
    
    """
   
    def __init__(self,
                 callbacks = ['deviance', 'diffs'],
                 fit_intercept = True,
                 max_iter = 100,
                 scale = None,
                 terms = 'auto',
                 tol = 0.0001,
                 verbose = False,
                 **kwargs):
    
        self.callbacks = callbacks
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.scale = scale
        self.terms = terms
        self.tol = tol
        self.verbose = verbose
        self.kwargs = kwargs

        self.model = LinearGAM(callbacks = self.callbacks,
                         fit_intercept = self.fit_intercept,
                         max_iter = self.max_iter,
                         scale = self.scale,
                         terms = self.terms,
                         tol = self.tol,
                         verbose = self.verbose,
                          **self.kwargs)
        
    def fit(self, X, y):
            """
            Fit Generalized additive models. 
        
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
            
            #remove the parameters from params not require for the LinearGAM model
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
 





