import argparse
import os
import time
import warnings

# We want clean output, without sklearn deprecation warnings
warnings.filterwarnings("ignore")

import tpot
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from config_dicts_local import *

parser = argparse.ArgumentParser(description='Run a single TPOT-NN evaluation job.')

parser.add_argument(
  '--dataset',
  choices=[
    'Hill_Valley_with_noise',
    'Hill_Valley_without_noise',
    'breast-cancer-wisconsin',
    'car-evaluation',
    'glass',
    'ionosphere',
    'spambase',
    'wine-quality-red',
    'wine-quality-white'
  ],
  help='PMLB dataset name'
)
parser.add_argument(
  '--use_template',
  help='Train TPOT using a template string (if False, run TPOT from config dict instead',
)
parser.add_argument(
  '--use_nn',
  help='Include TPOT-NN extension estimators',
  action='store_true'
)
parser.add_argument(
  '--estimator_select',
  choices=[
    'lr',
    'mlp',
    'all'
  ],
  help='Choose estimator architecture (lr or mlp). If not specified, all estimators are used.'
)
parser.add_argument(
  '--jobname',
  help='Name of the LFS job (for setting the pipeline file\'s name).'
)
parser.add_argument(
  '--solo',
  help='Run ONLY the estimator (don\'t use with --estimator_select all)',
  action='store_true'
)

args = parser.parse_args()

dataset = args.dataset
estimator_select = args.estimator_select
# tpot_all = args.tpot_all
tpot_all = True if estimator_select == 'all' else False
use_nn = args.use_nn

print(">> TRAINING TPOT NN EVALUATION MODEL")
print(">> JOB START TIME:        {0:.2f}".format(time.time()))
print(">> DATASET:               {0}".format(args.dataset))
print(">> USING SOLO ESTIMATOR:  {0}".format(args.solo))
print(">> USING TPOT-NN:         {0}".format(args.use_nn))
conf_type = 'template' if args.use_template else 'config_dict'
print(">> CONFIGURATION TYPE:    {0}".format(conf_type))

X, y = fetch_data(args.dataset, return_X_y=True, local_cache_dir="pmlb_data_cache/")

template_str = None

if conf_type == 'template':
  if tpot_all:
    template_str = 'Selector-Transformer-Estimator'
  elif use_nn:
    if estimator_select == 'lr':
      template_str = 'Selector-Transformer-PytorchLRClassifier'
    elif estimator_select == 'mlp':
      template_str = 'Selector-Transformer-PytorchMLPClassifier'
  else:
    if estimator_select == 'lr':
      template_str = 'Selector-Transformer-LogisticRegression'
    elif estimator_select == 'mlp':
      template_str = 'Selector-Transformer-MLPClassifier'
else:
  if tpot_all:
    if use_nn:
      config = config_nn
    else:
      config = classifier_config_dict
  else:
    # Build the variable name for the config dict as a string then eval it
    config_var_name = 'config_'
    config_var_name += estimator_select
    config_var_name += '_'
    config_var_name += 'nn' if use_nn else 'sk'
    if (not args.solo):
      config_var_name += '_tpot'
    config = eval(config_var_name)
    print(">> CONFIG DICT NAME:     ", config_var_name)

if template_str:
  print(">> TEMPLATE STRING:      ", template_str)

X_train, X_test, y_train, y_test = train_test_split(
  X, y.astype(np.float64), train_size=0.8, test_size=0.2
)

if conf_type == 'template':
  # USE TEMPLATE
  # (No stacking)
  clf_t = tpot.TPOTClassifier(
    generations=100,
    population_size=100,
    verbosity=2,
    config_dict=config_nn,  # We can be permissive when template_str is set
    template=template_str
  )
else:
  # USE CONFIG DICT
  # (Stacking allowed)
  clf_t = tpot.TPOTClassifier(
    generations=100,
    population_size=100,
    verbosity=2,
    config_dict=config,
  )

start_t = time.time()
print(">> BEGIN TRAINING AT:     {0:.2f}".format(start_t))
clf_t.fit(X_train, y_train)
end_t = time.time()
print(">> END TRAINING AT:       {0:.2f}".format(end_t))
print(">> TRAINING TIME ELAPSED: {0:.2f}".format(end_t - start_t))

print(">> ACCURACY SCORE:       ", clf_t.score(X_test, y_test))

estimator = estimator_select
use_nn_str = 'nn' if use_nn else 'no-nn'

pipeline_fname = os.path.join('pipelines', '{0}.py'.format(args.jobname))
clf_t.export(pipeline_fname)

print(">> PIPELINE SAVED TO:    ", pipeline_fname)
