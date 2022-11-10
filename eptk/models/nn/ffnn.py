# -*- coding: utf-8 -*-

from ..base import BasePredictor
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from sklearn.utils import check_array

class FFNNPredictor(BasePredictor):

   """
   Parameters
   -----------

   hidden_layers : an array of integers  (Default = [])
   Initialize the hidden layers of the neural network by passing a list of neurons per hidden layer.
   example: 
   If hidden_layers = [10, 10 ,2], then the neural network will have 1st hidden layer with 10 neurons, second with 10 neurons and the third hidden layer with 2 neurons.
   
   activation: an array of integers  (Default = [])
   Set the type of activation function for all the neurons present in each of the hidden layers.
   example: activation = ["relu", "relu", "relu"] will set all the three layers to have relu activation function.
   Note: The  size of the activation array should be same as the hidden_layers.
   
   dropout: float (between 0 and 1, Default = 0)
   randomly sets input units to 0 with a frequency of dropout at each step during training time, which helps prevent overfitting.
   The dropout layers are present in between all the subsequent layers if the model and has the same dropout rate given by dropout.
 
   training parameters:
   
  
   epochs: Integer (Default = 1)
   Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. 
   Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". 
   The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.

   batch_size: Integer or None. 
   Number of samples per gradient update. If unspecified, batch_size will default to 32.
   The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters. 

   Note: The final layer has only one neuron with identity activation. (For regression)
 
   """
   def __init__(self, hidden_layers = [512,256,32], activation = ["relu","relu","relu"], dropout = 0.1, epochs = 10, batch_size = None):
        
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.dropout = dropout
        for layer in range(len(hidden_layers)):
            #sequentially add layers to the model
            self.model.add(Dense(self.hidden_layers[layer], self.activation[layer], kernel_initializer = keras.initializers.glorot_uniform()))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout))
        
        #final regression layer
        self.model.add(Dense(1))
        # setting up the type of gradient descent
        self.model.compile(loss = "mse" , optimizer="adam")
       
        
   def fit(self, X, y, **kwargs):
           
            """
            Fit FFNN model. 
        
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
             Training data. The column "timestamp" will be removed if it is found. (When X is a pandas dataframe) 
            
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values."""

            try: #if X is a pandas object with timestamp column
                if "timestamp" in X.columns:
                    X = X.drop("timestamp", axis = 1)
            except:
                pass
            X = check_array(X)
            y = check_array(y, ensure_2d = False)         
            return self.model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size, **kwargs) 

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
            #converting into numpy array
            X = check_array(X) 

            return  self.model.predict(X)    
   
   def summary(self):
       """Once a model is "built", you can call its summary() method to display its contents"""

       return self.model.summary()
   
    

