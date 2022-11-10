# -*- coding: utf-8 -*-

from ..base import BasePredictor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._coordinate_descent import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

class LinearRegressionPredictor(BasePredictor):
        """
        Ordinary least squares Linear Regression.
        LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
        to minimize the residual sum of squares between the observed targets in
        the dataset, and the targets predicted by the linear approximation.
        
        Parameters
        ----------
        fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
        
        normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
        .. deprecated:: 1.0
           `normalize` was deprecated in version 1.0 and will be
           removed in 1.2.
        
        copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    
        n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup for n_targets > 1 and sufficient large problems.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
         
        See : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        for more details.
        
        
        
        
        """      
        
        
        
        
        
        
        
        def __init__(
                self,
                fit_intercept = True,
                normalize = "deprecated",
                copy_X = True,
                n_jobs = None,
                ):

            self.fit_intercept = fit_intercept
            self.normalize = normalize
            self.copy_X = copy_X
            self.n_jobs = n_jobs
            

            self.model = LinearRegression(
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                copy_X=self.copy_X,
                n_jobs=self.n_jobs)

        def fit(self, X, y):
            """
            Fit linear model. 
        
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
            
            super().set_params(**params)
            #remove the parameters from params not require for the sklearn regression model.
            
            self.model.set_params(**params)


class LassoRegressionPredictor(BasePredictor):
    
    """
    Linear Model trained with L1 prior as regularizer (aka the Lasso).
    
    The optimization objective for Lasso is::
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    
    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
        .. deprecated:: 1.0
            ``normalize`` was deprecated in version 1.0 and will be removed in
            1.2.
    precompute : 'auto', bool or array-like of shape (n_features, n_features),\
    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
 
    
    See : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    for more details.
          
    
    """

    def __init__(
            self,
            alpha=1.0,
            fit_intercept=True,
            normalize="deprecated",
            precompute=False,
            copy_X=True,
            max_iter=1000,
            tol=1e-4,
            warm_start=False,
            positive=False,
            random_state=None,
            selection="cyclic",
    ):

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

        self.model=Lasso(alpha = self.alpha,
        fit_intercept =self.fit_intercept,
        normalize = self.normalize,
        precompute = self.precompute,
        copy_X = self.copy_X,
        max_iter = self.max_iter,
        tol = self.tol,
        warm_start = self.warm_start,
        positive = self.positive,
        random_state = self.random_state,
        selection = self.selection)


    def fit(self, X, y):
            
            """
            Fit linear model. 
        
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
            super().set_params(**params)
            #remove the parameters from params not require for the sklearn Lasso model
            self.model.set_params(**params)





class RidgeRegressionPredictor(BasePredictor):

    """Linear least squares with l2 regularization.
    Minimizes the objective function::

    ||y - Xw||^2_2 + alpha * ||w||^2_2

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.


    Parameters
    ----------
    alpha : {float, ndarray of shape (n_targets,)}, default=1.0
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        
    fit_intercept : bool, default=True
        Whether to fit the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. ``X`` and ``y`` are expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
    
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    max_iter : int, default=None
        Maximum number of iterations for conjugate gradient solver.
        For 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
        For 'lbfgs' solver, the default value is 15000.

    tol : float, default=1e-3
        Precision of the solution.

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', \
            'sag', 'saga', 'lbfgs'}, default='auto'
        Solver to use in the computational routines:
        - 'auto' chooses the solver automatically based on the type of data.
        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. More stable for singular matrices than 'cholesky'.
        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution.
        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).
        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.
        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its improved, unbiased version named SAGA. Both methods also use an
          iterative procedure, and are often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.
        - 'lbfgs' uses L-BFGS-B algorithm implemented in
          `scipy.optimize.minimize`. It can be used only when `positive`
          is True.
        All last six solvers support both dense and sparse data. However, only
        'sag', 'sparse_cg', and 'lbfgs' support sparse input when `fit_intercept`
        is True.
           Stochastic Average Gradient descent solver.
           SAGA solver.
    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.
        Only 'lbfgs' solver is supported in this case.
    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag' or 'saga' to shuffle the data.
        See :term:`Glossary <random_state>` for details.
           `random_state` to support Stochastic Average Gradient.
           
           
    See : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge
    for more details.

    """


    def __init__(
            self,
            alpha=1.0,
            fit_intercept=True,
            normalize="deprecated",
            copy_X=True,
            max_iter=None,
            tol=1e-3,
            solver="auto",
            random_state=None,
    ):
        self.alpha=alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

        self.model = Ridge(
                alpha = self.alpha,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                copy_X=self.copy_X,
                max_iter=self.max_iter,
                tol=self.tol,
                solver=self.solver,
                random_state=self.random_state)


    def fit(self, X, y):
            
            """
            Fit linear model. 
        
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
            #remove the parameters from params not require for the sklearn ridge model
            self.model.set_params(**params)






class ElasticNetPredictor(BasePredictor):

    """Linear regression with combined L1 and L2 priors as regularizer.
            Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    
    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::
            a * ||w||_1 + 0.5 * b * ||w||_2^2
    
    where::
            alpha = a + b and l1_ratio = a / (a + b)
    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.

    Parameters
    ----------
    
    alpha : float, default=1.0
        Constant that multiplies the penalty terms. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter. ``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.
    
    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.
    
    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.
    
    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
        .. deprecated:: 1.0
            ``normalize`` was deprecated in version 1.0 and will be removed in
            1.2.
    
    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.
    
    max_iter : int, default=1000
        The maximum number of iterations.
    
    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.
    
    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.
    
    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.
    
    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    
    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
    
   

    see: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    for more details.

        """        
        
        
        
        
    def __init__(self,
                 alpha = 1.0,
                 copy_X = True,
                 fit_intercept = True,
                 l1_ratio =  0.5,
                 max_iter = 1000,
                 normalize =  False,
                 positive =  False,
                 precompute = False,
                 random_state = None,
                 selection =  'cyclic',
                 tol = 0.0001,
                 warm_start = False
                ):
 
             self.alpha = alpha
             self.fit_intercept = fit_intercept
             self.l1_ratio = l1_ratio
             self.normalize = normalize
             self.precompute = precompute
             self.copy_X = copy_X
             self.max_iter = max_iter
             self.tol = tol
             self.warm_start = warm_start
             self.positive = positive
             self.random_state = random_state
             self.selection = selection

             self.model =  ElasticNet(alpha = self.alpha,
                             fit_intercept =self.fit_intercept,
                             normalize = self.normalize,
                             l1_ratio = self.l1_ratio,
                             precompute = self.precompute,
                             copy_X = self.copy_X,
                             max_iter = self.max_iter,
                             tol = self.tol,
                             warm_start = self.warm_start,
                             positive = self.positive,
                             random_state = self.random_state,
                             selection = self.selection)
       

        
                                       

    def fit(self, X, y):
            """
            Fit linear model. 
        
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
            
            super().set_params(**params)
            #remove the parameters from params not require for the sklearn ElasticNet model.
            
            self.model.set_params(**params)


class SGDRPredictor(BasePredictor):
    
    
    """Linear model fitted by minimizing a regularized empirical loss with SGD
    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).
    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.
    This implementation works with data represented as dense numpy arrays of
    floating point values for the features.

    Parameters
    ----------
    loss : str, default='squared_loss'
        The loss function to be used. The possible values are 'squared_loss',
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'
        The 'squared_error' refers to the ordinary least squares fit.
        'huber' modifies 'squared_error' to focus less on getting outliers
        correct by switching from squared to linear loss past a distance of
        epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is
        linear past that; this is the loss function used in SVR.
        'squared_epsilon_insensitive' is the same but becomes squared loss past
        a tolerance of epsilon.
      
    penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term. The higher the
        value, the stronger the regularization.
        Also used to compute the learning rate when set to `learning_rate` is
        set to 'optimal'.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        Only used if `penalty` is 'elasticnet'.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

    tol : float, default=1e-3
        The stopping criterion. If it is not None, training will stop
        when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
        epochs.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss functions; only if `loss` is
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
        For 'huber', determines the threshold at which it becomes less
        important to get the prediction exactly right.

        For epsilon-insensitive, any differences between the current prediction
        and the correct label are ignored if they are less than this threshold.

    random_state : int, RandomState instance, default=None
        Used for shuffling the data, when ``shuffle`` is set to ``True``.
        Pass an int for reproducible output across multiple function calls.
        
    learning_rate : string, default='invscaling'
        The learning rate schedule:
        - 'constant': `eta = eta0`
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          where t0 is chosen by a heuristic proposed by Leon Bottou.
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        - 'adaptive': eta = eta0, as long as the training keeps decreasing.
          Each time n_iter_no_change consecutive epochs fail to decrease the
          training loss by tol or fail to increase validation score by tol if
          early_stopping is True, the current learning rate is divided by 5.


    eta0 : double, default=0.01
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.01.

    power_t : double, default=0.25
        The exponent for inverse scaling learning rate.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score returned by the `score` method is not
        improving by at least `tol` for `n_iter_no_change` consecutive
        epochs.

    
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if `early_stopping` is True.

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before stopping
        fitting.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.
      
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.
        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit``  will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights accross all
        updates and stores the result in the ``coef_`` attribute. If set to
        an int greater than 1, averaging will begin once the total number of
        samples seen reaches `average`. So ``average=10`` will begin
        averaging after seeing 10 samples.
   
    
    """

    def __init__(self, alpha = 0.0001,
                 average = False,
                 early_stopping = False,
                 epsilon = 0.1,
                 eta0 = 0.01,
                 fit_intercept = True,
                 l1_ratio = 0.15,
                 learning_rate = 'invscaling',
                 loss = 'squared_loss',
                 max_iter = 1000,
                 n_iter_no_change = 5,
                 penalty = 'l2',
                 power_t = 0.25,
                 random_state = None,
                 shuffle = True,
                 tol = 0.001,
                 validation_fraction = 0.1,
                 verbose = 0,
                 warm_start = False):
    
                 self.alpha = alpha
                 self.average = average
                 self.early_stopping = early_stopping
                 self.epsilon = epsilon
                 self.eta0 = eta0
                 self.fit_intercept = fit_intercept
                 self.l1_ratio = l1_ratio
                 self.learning_rate = learning_rate
                 self.loss = loss
                 self.max_iter = max_iter
                 self.n_iter_no_change = n_iter_no_change
                 self.penalty = penalty
                 self.power_t = power_t
                 self.random_state = random_state
                 self.shuffle = shuffle
                 self.tol = tol
                 self.validation_fraction = validation_fraction
                 self.verbose = verbose
                 self.warm_start =  warm_start
                 
                 self.model = SGDRegressor(alpha = self.alpha,
                 average = self.average,
                 early_stopping = self.early_stopping,
                 epsilon = self.epsilon,
                 eta0 = self.eta0,
                 fit_intercept = self.fit_intercept ,
                 l1_ratio = self.l1_ratio,
                 learning_rate = self.learning_rate,
                 loss = self.loss,
                 max_iter = self.max_iter,
                 n_iter_no_change = self.n_iter_no_change,
                 penalty = self.penalty,
                 power_t = self.power_t,
                 random_state = self.random_state,
                 shuffle = self.shuffle,
                 tol = self.tol,
                 validation_fraction = self.validation_fraction,
                 verbose = self.verbose,
                 warm_start = self.warm_start)

    def fit(self, X, y):
            
            """
            Fit linear model. 
        
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
            #remove the parameters from params not require for the sklearn  SDGRegression model
            self.model.set_params(**params)




                   


