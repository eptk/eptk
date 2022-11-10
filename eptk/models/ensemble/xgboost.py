# -*- coding: utf-8 -*-

from eptk.models.base import BasePredictor
from xgboost import XGBRegressor 
import copy


class XGBoostPredictor(BasePredictor):

   """
   General parameters
   -----------------------------------------
   booster: [default= gbtree ]
   Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
   
   verbosity: [default=1]
   Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message. If there’s unexpected behaviour, please try to increase value of verbosity.

   validate_parameters: [default to false, except for Python, R and CLI interface]
   When set to True, XGBoost will perform validation of input parameters to check whether a parameter is used or not. The feature is still experimental. It’s expected to have some false positives.
   
   nthread: [default to maximum number of threads available if not set]
   Number of parallel threads used to run XGBoost. When choosing it, please keep thread contention and hyperthreading in mind.
   disable_default_eval_metric [default=``false``]
   Flag to disable default metric. Set to 1 or true to disable.

   num_pbuffer: [set automatically by XGBoost, no need to be set by user]
   Size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step.

   num_feature: [set automatically by XGBoost, no need to be set by user]
   Feature dimension used in boosting, set to maximum dimension of the feature

   Parameters for Tree Booster
   ------------------------------
   eta: [default=0.3, alias: learning_rate]
   Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
   range: [0,1]
   
   gamma: [default=0, alias: min_split_loss]
   Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
   range: [0,inf]

   max_depth: [default=6]
   Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
   range: [0,inf] (0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist)

   min_child_weight: [default=1]
   Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
   range: [0,inf]

   max_delta_step: [default=0]
   Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
   range: [0,inf]

   subsample: [default=1]
   Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
   range: (0,1]

   sampling_method: [default= uniform]
   The method to use to sample the training instances.
   uniform: each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
   gradient_based: the selection probability for each training instance is proportional to the regularized absolute value of gradients .
   subsample may be set to as low as 0.1 without loss of model accuracy. Note that this sampling method is only supported when tree_method is set to gpu_hist; other tree methods only support uniform sampling.

   colsample_bytree, colsample_bylevel, colsample_bynode [default=1]
   This is a family of parameters for subsampling of columns.
   All colsample_by* parameters have a range of (0, 1], the default value of 1, and specify the fraction of columns to be subsampled.
   colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
   colsample_bylevel is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
   colsample_bynode is the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
   colsample_by* parameters work cumulatively. For instance, the combination {'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5} with 64 features will leave 8 features to choose from at each split.
   On Python interface, when using hist, gpu_hist or exact tree method, one can set the feature_weights for DMatrix to define the probability of each feature being selected when using column sampling. There’s a similar parameter for fit method in sklearn interface.
   
   lambda: [default=1, alias: reg_lambda]
   L2 regularization term on weights. Increasing this value will make model more conservative.

   alpha: [default=0, alias: reg_alpha]
   L1 regularization term on weights. Increasing this value will make model more conservative.

   tree_method: string [default= auto]
   The tree construction algorithm used in XGBoost. See description in the reference paper and XGBoost Tree Methods.
   XGBoost supports approx, hist and gpu_hist for distributed training. Experimental support for external memory is available for approx and gpu_hist.
   Choices: auto, exact, approx, hist, gpu_hist, this is a combination of commonly used updaters. For other updaters like refresh, set the parameter updater directly.
       auto: Use heuristic to choose the fastest method.
       For small dataset, exact greedy (exact) will be used.
       For larger dataset, approximate algorithm (approx) will be chosen. It’s recommended to try hist and gpu_hist for higher performance with large dataset. (gpu_hist)has support for external memory.
       Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is chosen to notify this choice.
       exact: Exact greedy algorithm. Enumerates all split candidates.
       approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
       hist: Faster histogram optimized approximate greedy algorithm.
       gpu_hist: GPU implementation of hist algorithm.

   sketch_eps: [default=0.03]
   Only used for tree_method=approx.
   This roughly translates into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy.
   Usually user does not have to tune this. But consider setting to a lower number for more accurate enumeration of split candidates.
   range: (0, 1)

   scale_pos_weight: [default=1]
   Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances).
   
   updater: [default= grow_colmaker,prune]
   A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters. However, it could be also set explicitly by a user. The following updaters exist:
   grow_colmaker: non-distributed column-based construction of trees.
   grow_histmaker: distributed tree construction with row-based data splitting based on global proposal of histogram counting.
   grow_local_histmaker: based on local histogram counting.
   grow_quantile_histmaker: Grow tree using quantized histogram.
   grow_gpu_hist: Grow tree with GPU.

   sync: synchronizes trees in all distributed nodes.

   refresh: refreshes tree’s statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.

   prune: prunes the splits where loss < min_split_loss (or gamma).
   In a distributed setting, the implicit updater sequence value would be adjusted to grow_histmaker,prune by default, and you can set tree_method as hist to use grow_histmaker.
   
   refresh_leaf: [default=1]
   This is a parameter of the refresh updater. When this flag is 1, tree leafs as well as tree nodes’ stats are updated. When it is 0, only node stats are updated.
   
   process_type: [default= default]
   A type of boosting process to run.
   Choices: default, update
      default: The normal boosting process which creates new trees.
      update: Starts from an existing model and only updates its trees. In each boosting iteration, a tree from the initial model is taken, a specified sequence of updaters is run for that tree, and a modified tree is added to the new model. The new model would have either the same or smaller number of trees, depending on the number of boosting iterations performed. Currently, the following built-in updaters could be meaningfully used with this process type: refresh, prune. With process_type=update, one cannot use updaters that create new trees.
  
   grow_policy: [default= depthwise]
    Controls a way new nodes are added to the tree.
    Currently supported only if tree_method is set to hist or gpu_hist.
    Choices: depthwise, lossguide
          depthwise: split at nodes closest to the root.
          lossguide: split at nodes with highest loss change.

   max_leaves: [default=0]
   Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.

   max_bin: [default=256]
   Only used if tree_method is set to hist or gpu_hist.
   Maximum number of discrete bins to bucket continuous features.
   Increasing this number improves the optimality of splits at the cost of higher computation time.

   predictor: [default=``auto``]
   The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
          auto: Configure predictor based on heuristics.
          cpu_predictor: Multicore CPU prediction algorithm.
          gpu_predictor: Prediction using GPU. Used when tree_method is gpu_hist. When predictor is set to default value auto, the gpu_hist tree method is able to provide GPU based prediction without copying training data to GPU memory. If gpu_predictor is explicitly specified, then all data is copied into GPU, only recommended for performing prediction tasks.
          num_parallel_tree, [default=1] - Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.

   monotone_constraints:
   Constraint of variable monotonicity. 
   
   interaction_constraints:
   Constraints for interaction representing permitted interactions. The constraints must be specified in the form of a nest list, e.g. [[0, 1], [2, 3, 4]], where each inner list is a group of indices of features that are allowed to interact with each other. 

      
   see: 
   https://xgboost.readthedocs.io/en/latest/parameter.html
   """

   def __init__(
        self,
        *,
        base_score = 0.5,
        booster = 'gbtree',
        colsample_bylevel = 1,
        colsample_bynode = 1,
        colsample_bytree = 1,
        gamma = 0,
        importance_type = 'gain',
        learning_rate = 0.1,
        max_delta_step = 0,
        max_depth = 3,
        min_child_weight = 1,
        missing = None,
        n_estimators = 100,
        n_jobs = 1,
        nthread = None,
        objective = 'reg:linear',
        random_state = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        scale_pos_weight = 1,
        seed = None,
        silent = None,
        subsample = 1,
        verbosity = 1,
        **kwargs
        ):

      
        self.base_score = base_score 
        self.booster = booster
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.importance_type = importance_type
        self.learning_rate = learning_rate
        self.max_delta_step = max_delta_step
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.missing = missing
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.nthread = nthread
        self.objective = objective
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.seed = seed
        self.silent = silent
        self.subsample = subsample
        self.verbosity =verbosity
        self.kwargs = kwargs 
       






        self.model= XGBRegressor(base_score = self.base_score ,
        booster = self.booster,
        colsample_bylevel = self.colsample_bylevel,
        colsample_bynode = self.colsample_bynode,
        colsample_bytree = self.colsample_bytree,
        gamma = self.gamma,
        importance_type = self.importance_type ,
        learning_rate = self.learning_rate,
        max_delta_step = self.max_delta_step,
        max_depth = self.max_depth,
        min_child_weight = self.min_child_weight,
        missing = self.missing,
        n_estimators = self.n_estimators,
        n_jobs = self.n_jobs,
        nthread = self.nthread,
        objective = self.objective ,
        random_state = self.random_state,
        reg_alpha = self.reg_alpha,
        reg_lambda = self.reg_lambda ,
        scale_pos_weight = self.scale_pos_weight,
        seed = self.seed,
        silent = self.silent,
        subsample = self.subsample,
        verbosity = self.verbosity,
        **self.kwargs
        )  #remove the kwargs ( in the self.kwargs) not supported by the Xgboost model.
        
   def fit(self, X, y):
            """
            Fit XGBOOST regressor model. 
        
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
 

   
 