import numpy as np

from tpot.config import *

config_tpot_no_estimators = {
  # Preprocesssors
  'sklearn.preprocessing.Binarizer': {
    'threshold': np.arange(0.0, 1.01, 0.05)
  },

  'sklearn.decomposition.FastICA': {
    'tol': np.arange(0.0, 1.01, 0.05)
  },

  'sklearn.cluster.FeatureAgglomeration': {
    'linkage': ['ward', 'complete', 'average'],
    'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
  },

  'sklearn.preprocessing.MaxAbsScaler': {
  },

  'sklearn.preprocessing.MinMaxScaler': {
  },

  'sklearn.preprocessing.Normalizer': {
    'norm': ['l1', 'l2', 'max']
  },

  'sklearn.kernel_approximation.Nystroem': {
    'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
    'gamma': np.arange(0.0, 1.01, 0.05),
    'n_components': range(1, 11)
  },

  'sklearn.decomposition.PCA': {
    'svd_solver': ['randomized'],
    'iterated_power': range(1, 11)
  },

  'sklearn.preprocessing.PolynomialFeatures': {
    'degree': [2],
    'include_bias': [False],
    'interaction_only': [False]
  },

  'sklearn.kernel_approximation.RBFSampler': {
    'gamma': np.arange(0.0, 1.01, 0.05)
  },

  'sklearn.preprocessing.RobustScaler': {
  },

  'sklearn.preprocessing.StandardScaler': {
  },

  'tpot.builtins.ZeroCount': {
  },

  'tpot.builtins.OneHotEncoder': {
    'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
    'sparse': [False],
    'threshold': [10]
  },

  # Selectors
  'sklearn.feature_selection.SelectFwe': {
    'alpha': np.arange(0, 0.05, 0.001),
    'score_func': {
      'sklearn.feature_selection.f_classif': None
    }
  },

  'sklearn.feature_selection.SelectPercentile': {
    'percentile': range(1, 100),
    'score_func': {
      'sklearn.feature_selection.f_classif': None
    }
  },

  'sklearn.feature_selection.VarianceThreshold': {
    'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
  },

  'sklearn.feature_selection.RFE': {
    'step': np.arange(0.05, 1.01, 0.05),
    'estimator': {
      'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ['gini', 'entropy'],
        'max_features': np.arange(0.05, 1.01, 0.05)
      }
    }
  },

  'sklearn.feature_selection.SelectFromModel': {
    'threshold': np.arange(0, 1.01, 0.05),
    'estimator': {
      'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ['gini', 'entropy'],
        'max_features': np.arange(0.05, 1.01, 0.05)
      }
    }
  }
}

###############################################################################
# TPOT-NN logistic regression alone
config_lr_nn = {
  'tpot.builtins.PytorchLRClassifier': {
    'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    'batch_size': [4, 8, 16, 32],
    'num_epochs': [5, 10, 15],
    'weight_decay': [0, 1e-4, 1e-3, 1e-2]
  }
}
# TPOT-NN logistic regression with selectors and transformers
config_lr_nn_tpot = {
  **config_lr_nn,
  **config_tpot_no_estimators
}

###############################################################################
# TPOT-NN MLP alone
config_mlp_nn = {
  'tpot.builtins.PytorchMLPClassifier': {
    'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    'batch_size': [4, 8, 16, 32],
    'num_epochs': [5, 10, 15],
    'weight_decay': [0, 1e-4, 1e-3, 1e-2]
  }
}
# TPOT-NN MLP with selectors and transformers
config_mlp_nn_tpot = {
  **config_mlp_nn,
  **config_tpot_no_estimators
}

###############################################################################
# Scikit-learn logistic regression alone
config_lr_sk = {
  'sklearn.linear_model.LogisticRegression': classifier_config_dict['sklearn.linear_model.LogisticRegression']
}
# Scikit-learn logistic regression with selectors and transformers
config_lr_sk_tpot = {
  **config_lr_sk,
  **config_tpot_no_estimators
}

###############################################################################
# Scikit-learn MLP alone
config_mlp_sk = {
  'sklearn.neural_network.MLPClassifier': {
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
  }
}
# Scikit-learn MLP with selectors and transformers
config_mlp_sk_tpot = {
  **config_mlp_sk,
  **config_tpot_no_estimators
}

###############################################################################
# All of TPOT (including NN estimators)
config_nn = {
  **config_lr_nn,
  **config_mlp_nn,
  **classifier_config_dict
}