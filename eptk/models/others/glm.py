# -*- coding: utf-8 -*-
from statsmodels.api  import families, GLM
from statsmodels.tools.tools import add_constant
from ..base import BasePredictor

#distribution familes and link functions

dist_fam = {"gamma" : families.Gamma, "gaussian" : families.Gaussian, "inverse_gauss" : families.InverseGaussian }
link_fam = {"inverse_power" : families.links.inverse_power(), "inverse_squared" : families.links.inverse_squared(), "identity": families.links.identity(),
            "log":  families.links.log() }



class GLMPredictor(BasePredictor):
    """ Genralized Linear Model.

    Parameters
    ----------
    distribution_family : str
    Set the distribution family of the target (energy readings).
       options : {"gamma" , "gaussian, "inverse_gauss" (inverse gaussian)}
    
    link function: str 
    E(Y|X) = g-1(X'B), where g is the link function.
      options: {"inverse_power", "inverse_squared", "identity", "log"}

    For example: Gaussian distribution with identity link is same as OLS. 

    fit_intercept : bool, default = True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. 


    See: https://www.statsmodels.org/stable/glm.html
    for more details.      

    """
    def __init__(
            self,
            fit_intercept = True,
            distribution_family = "gamma",
            link_function = "inverse_power",
    ):

        self.distribution_family = distribution_family                             
        self.link_function = link_function
        self.fit_intercept = fit_intercept
        self.res = None
        
    def fit(self, X, y):
            
            """
            Fit model. 
        
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
            
            if self.fit_intercept == True:
                X = add_constant(X, prepend=False)

            self.model = GLM(y, X, family = dist_fam[self.distribution_family](link_fam[self.link_function]))
            self.res = self.model.fit()
            
  
            return self.res

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
            
            
            if self.fit_intercept == True:
                X = add_constant(X, prepend=False)
            
            
            return self.res.predict(X)



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
            #remove the parameters from params not required for statsmodel GLM
            self.model.set_params(**params)
 
    def summary(self):

        """ Returns the full summary of the fitted model"""

        return self.res.summary() 
