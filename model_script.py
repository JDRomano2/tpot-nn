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
  '--tpot_all',
  help='Run TPOT using all selectors, transformers, and estimators (can be combined with \'--use_nn\').',
  action='store_true'
)
parser.add_argument(
  '--use_template',
  help='Train TPOT using a template string (if False, run TPOT from config dict instead',
  action='store_true'
)
parser.add_argument(
  '--use_nn',
  help='Include TPOT-NN extension estimators',
  action='store_true'
)
parser.add_argument(
  '--use_classic',
  help='Include \'classic\' TPOT estimators',
  action='store_true'
)
parser.add_argument(
  '--estimator_select',
  choices=[
    'lr',
    'mlp'
  ],
  help='Choose estimator architecture (lr or mlp). If not specified, all estimators are used.'
)

args = parser.parse_args()

dataset = args.dataset
tpot_all = args.tpot_all
use_nn = args.use_nn
use_classic = args.use_classic
estimator_select = args.estimator_select

if tpot_all and use_classic:
  raise RuntimeError("Cannot use 'tpot_all' and 'use_classic' simultaneously")
if tpot_all and estimator_select:
  raise RuntimeError("Cannot use 'tpot_all' and set 'estimator_select' simultaneously")

print(">> TRAINING TPOT NN EVALUATION MODEL")
print(">> JOB START TIME:        {0:.2f}".format(time.time()))
print(">> DATASET:               {0}".format(args.dataset))
print(">> USING CLASSIC TPOT:    {0}".format(args.use_classic))
print(">> USING TPOT-NN:         {0}".format(args.use_nn))
conf_type = 'template' if args.use_template else 'config_dict'
print(">> CONFIGURATION TYPE:    {0}".format(conf_type))

X, y = fetch_data(args.dataset, return_X_y=True, local_cache_dir="pmlb_data_cache/")

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
  template_str = None
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
    if use_classic:
      config_var_name += '_tpot'
    config = eval(config_var_name)

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

pipeline_fname = os.path.join(
  'pipelines', 
  '{0}-{1}-{2}.py'.format(
    args.dataset, args.model_template, int(time.time())
  )
)
clf_t.export(pipeline_fname)

print(">> PIPELINE SAVED TO:    ", pipeline_fname)