# -*- coding: utf-8 -*-
from .base import BasePredictor
from sklearn.utils import check_array
import numpy as np
import pandas as pd
from eptk.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from .classical import RidgeRegressionPredictor
import copy

class Ensembler(BasePredictor):
   """ 
   Ensembler for aggregating multiple predictors together. The final prediction is a weighted prediction of all the predictors.

   Parameters
   -----------
   predictors: A list of predictor objects
   A list containing all the unfitted predictors.

   weights: An array of weights or str or None (Default)
   Options:-
            Array     : An array of floats of length equal to the number of predictors.
            "uniform" : Gives equal importance to the prediction from all the models.
            None      : No weights are assigned. Weights are to be optimized while fitting the model.

   optimize_weights : Boolean (Deafault =False)
            True or False. If set to True, the weights are optimized by using one of the 3 techniques.

            technique: str
            Select one from  the techniques provided as options.
            Options : {"sparse", "meta_regressor", "bayesian_opt" }
                 For all the techniques the training data is split into 80:20. All indvidual predictors are fitted on the 80% of the data.
                 Then the 20% of data is considered for optimizing the weights.
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

   def __init__(self, predictors = [], weights = None, optimize_weights = False, technique = "bayesian_opt"):
        self.predictors = predictors
        self.weights = weights
        self.optimize_weights = optimize_weights
        self.technique = technique
        if self.weights is None:
            print("Proceeding without a weight array.")
        elif weights is "uniform":
            try:
              self.weights = 1/len(predictors) * np.ones(len(predictors))
            except ZeroDivisionError:
                 raise ValueError("The list of predictors cannot be empty")

        else:
         self.weights = check_array(weights, ensure_2d = False)
         if len(self.predictors) != len(self.weights):
            raise ValueError (f"The weights array should have size = no. of predictors.\n no. of predictors : {len(predictors)} ")
       
        
   def fit(self, X, y):
            """
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
             Training data. The column "timestamp" will be removed if it is found. (When X is a pandas dataframe) 
            
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

            """

            try: #if X is a pandas object with timestamp column
                if "timestamp" in X.columns:
                    X = X.drop("timestamp", axis = 1)
            except:
                pass

            if  self.optimize_weights == True:
              self._optimize_weights(X, y)

            for predictor in self.predictors:
                predictor.fit(X,y)

            return self.predictors

   
   def _optimize_weights(self, X, y):
        """
        Techniques : meta_regressor, sparse, bayesian_optimization
        """
        technique = self.technique
        X_ = check_array(X) # checks and converts into numpy array
        y_ = check_array(y, ensure_2d = False) 
        total = len(X_)
        split = 80*total//100
        X_train = X_[:split]
        y_train = y_[:split]
        X_test = X_[split:]
        y_test = y_[split:]
          
        for predictor in self.predictors:
            predictor.fit(X_train,y_train)

        predictions = []
        for i in self.predictors:
            predictions.append(i.predict(X_test))
                   

        if  technique == "meta_regressor":
          # making a matrix for training
          predictions = np.array(predictions)
          predictions = predictions.transpose()
          meta = RidgeRegressionPredictor(fit_intercept = False)
          meta.fit(predictions, y_test)
          self.weights = meta.model.coef_
          print(f"Weights: {self.weights}")


        elif technique == "sparse":
         performance = []
         for pred in predictions:
            performance.append(-mean_squared_error(pred,y_test)) 
        
         #numerically stable softmax
         def softmax(x):
           """Compute softmax values for each sets of scores in x."""
           e_x = np.exp(x - np.max(x))
           return e_x / e_x.sum(axis = 0) 
         
         self.weights = softmax(performance)
         print(f"Weights: {self.weights}")  
         print(f"Weights: {self.weights}")  
          
        elif technique == "bayesian_opt":
            # setting the bounds
            pbounds={}
            for i in range(len(self.predictors)):
                     pbounds["w"+str(i+1)] = (0.001,1)
            

            #maximize the negative mse           
            def cost_function_over_weight_space(**weights):
               p= np.array(predictions)
               w = np.array(list(weights.values()))
               sum = np.sum(w)
               y_hat = 1/sum * np.dot(w,p) # normalized weights cost func
               return -(mean_squared_error(y_test,y_hat))
          
            optimizer = BayesianOptimization(f = cost_function_over_weight_space,pbounds=pbounds,random_state=1)
            optimizer.maximize(init_points=20,n_iter=40)
            
            
            negativeloss = [i["target"] for i in optimizer.res] 
            index = negativeloss.index(max(negativeloss))
            weights = np.array(list(optimizer.res[index]["params"].values()))
            self.weights = 1/sum(weights)*weights # normalize
            print(f"Weights: {self.weights}")
                
        
        else: 
            raise AttributeError ("Invalid value for attribute technique")

        #reset predictors
        self.predictors = [predictor.reset() for predictor in self.predictors]    
        return self.weights 
              

   def predict(self, X):
            
            
            """
            Parameters
            ----------
            X : array-like or sparse matrix, shape (n_samples, n_features)
            
            Returns
            --------
            An array of ensembler estimates for input X.

            """ 
            
            
            
            if self.weights is None:
                raise ValueError( "Weights cannot be None")
            
            try: #if X is a pandas object with timestamp column
                if "timestamp" in X.columns:
                    X = X.drop("timestamp", axis = 1)
            except:
                pass
             

            predictions = []
            for m in self.predictors:
                predictions.append(m.predict(X))
            
            predictions = np.array(predictions)

            return  np.dot(self.weights, predictions)      
  
   def reset(self):

      """A method to reset the ensembler. It prevents some models from remembering training from previous iteration while cross validation."""
      a = copy.deepcopy(self.__dict__)
      a["predictors"] = [predictor.reset() for predictor in a["predictors"]]
      new = self.__class__(**a)
      return new
  


   