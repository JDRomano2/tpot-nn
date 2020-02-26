
# coding: utf-8

# In[1]:


from tpot import TPOTRegressor, TPOTClassifier
from tpot.export_utils import generate_import_code, generate_export_pipeline_code
from tpot.export_utils import export_pipeline, expr_to_tree
from sklearn.model_selection import train_test_split, cross_val_score
# from tpot.config.classifier_nn import classifier_config_nn
from sklearn.pipeline import make_pipeline
from tpot.config import regressor_config_dict_light
from sklearn.metrics.scorer import make_scorer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_digits
from utils import *
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import os

import glob
import itertools
from time import process_time
# from ipywidgets import IntProgress


# In[2]:


# set global variables
n_gen = 50
n_pop = 100

my_datasets = pd.read_csv('qsar_datasets.txt', header = None)[0].values.tolist()
heavy_sets = ['3A4', 'LOGD']
for i in heavy_sets:
    my_datasets.remove(i)

make_func = 'def opt_pipe(training_features, testing_features):\n'
import_impute = 'from sklearn.impute import SimpleImputer\n\n'
impute_text = '\timputer = SimpleImputer(strategy="median")\n\timputer.fit(training_features)\n\ttraining_features = imputer.transform(training_features)\n\ttesting_features = imputer.transform(testing_features)\n'

def write_pipes(name, tpot):
    """Write TPOT pipelines out to subdirectories."""
    import_codes = generate_import_code(tpot._optimized_pipeline, tpot.operators)
    pipeline_codes = generate_export_pipeline_code(expr_to_tree(tpot._optimized_pipeline,tpot._pset), tpot.operators)
    pipe_text = import_codes.replace('import numpy as np\nimport pandas as pd\n', 'from sklearn.preprocessing import FunctionTransformer\nfrom copy import copy\n')
    if tpot._imputed: # add impute code when there is missing data
        pipe_text += import_impute + make_func + impute_text
    else:
        pipe_text += make_func
    pipe_text += '\texported_pipeline = ' + pipeline_codes + "\n\treturn({'train_feat': training_features, 'test_feat': testing_features, 'pipe': exported_pipeline})"
    f = open(name + '.py', 'w')
    f.write(pipe_text)
    f.close()


# In[3]:


personal_config = regressor_config_dict_light

personal_config['sklearn.neural_network.MLPRegressor'] = {
    # MLPClassifier for neural networks
    # TODO: revisit/tweak: alpha, momentum, learning rate_init
    # separater paras based on activation
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'hidden_layer_sizes': [(1024, 256, 64, )],
    'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
    'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 0.75, 0.9],
    'momentum': [0.1, 0.5, 0.75, 0.9]
}

personal_config['sklearn.feature_selection.VarianceThreshold'] = {
        'threshold': [0.0005]
}

# In[4]:


scoring_fn = 'r2'
# def my_r2(y, y_hat):
#     '''Calculates the R^2 for set of observations y and the predictions y_hat.
#     Same metric in DNN paper: squared Pearson correlation coefficient
#     between predicted and observed activities in the test set.
#     Assumes these two are np arrays.
#     '''
#     r2 = np.corrcoef(y_hat, y)[0,1]**2
#     return(r2)
# scoring_fn = make_scorer(my_r2, greater_is_better = True)

random_state = 1618
path = ''
extension = 'csv'
data_dir = '/project/moore/users/weixuanf/TPOT-QSAR-project/raw_data/preprocess/'
label = 'Act'


# In[5]:


def build_tpot_structure(outcome = 'quant'):
    if outcome == 'binary':
        tpot = TPOTClassifier(generations = n_gen,
                         population_size = n_pop,
                         verbosity = 2,
                         config_dict = personal_config,
                         scoring = scoring_fn,
                         random_state = random_state,
                         cv = TimeSeriesSplit(n_splits=5),
                         n_jobs = 8,
                         template = 'VarianceThreshold-Selector-Transformer-MLPClassifier')
    else: # quantitative trait
        tpot = TPOTRegressor(generations = n_gen,
                         population_size = n_pop,
                         verbosity = 2,
                         config_dict = personal_config,
                         scoring = scoring_fn,
                         random_state = random_state,
                         cv = TimeSeriesSplit(n_splits=5),
                         n_jobs = 8,
                         template = 'VarianceThreshold-Selector-Transformer-MLPRegressor')
    return tpot

def run_tpot(dat_name, outcome = 'quant'):
    tpot = build_tpot_structure(outcome)

    # Read in the data:
    train_data = pd.read_csv(data_dir + dat_name + '_training_preprocessed.csv', index_col = 'MOLECULE')
    X_train = train_data.drop(label, axis=1)
    y_train = train_data[label]
    del train_data

    ### Run TPOT with `template`:
    t_start = process_time() # start timing
    tpot.fit(X_train.values, y_train)
    t_stop = process_time() # end timing

    print('Total elapsed process time:', t_stop - t_start, 'seconds')
    write_pipes('pipelines/3layers_' + dat_name + '_' + str(random_state), tpot)
    CV_score = tpot._optimized_pipeline_score
    delta_t = t_stop - t_start

    return {'CV_R2_score': CV_score, 'Elapsed time': delta_t}


# In[ ]:


mtypes = {'datasets': my_datasets,
          'outcomes':['quant']}
mtype_grid = expand_grid(mtypes) # data type grid

results = mtype_grid.apply(
    lambda r: run_tpot(r.datasets, r.outcomes),
    axis = 1, result_type = 'expand')
results
final_results = pd.concat([mtype_grid, results], axis = 1)
final_results.to_csv('accuracies/3layers_qsar_' + str(random_state) + ".csv", sep = "\t")
