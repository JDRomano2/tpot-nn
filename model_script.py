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

parser = argparse.ArgumentParser(description='Run a single TPOT-NN evaluation job.')
parser.add_argument(
  '--dataset',
  choices=[
    'Hill_Valley_with_noise',
    'breast-cancer-wisconsin',
    'spambase',
    'ionosphere'
  ],
  help='PMLB dataset name'
)
parser.add_argument(
  '--model_template',
  choices=[
    'tpot_lr',
    'tpot_mlp',
    'tpot_all',
    'tpotnn_lr',
    'tpotnn_mlp',
    'tpotnn_all'
  ],
  help='Type(s) of model to include for TPOT'
)
args = parser.parse_args()

print(">> TRAINING TPOT NN EVALUATION MODEL")
print(">> JOB START TIME:        {0:.2f}".format(time.time()))
print(">> DATASET:              ", args.dataset)
print(">> MODEL TEMPLATE:       ", args.model_template)

X, y = fetch_data(args.dataset, return_X_y=True, local_cache_dir="pmlb_data_cache/")

template = args.model_template
t0, t1 = tuple(template.split('_'))

if t0 == 'tpot':
  config = None
  if t1 == 'lr':
    template_str = 'Selector-Transformer-LogisticRegression'
  elif t1 == 'mlp':
    template_str = 'Selector-Transformer-MLPClassifier'
  else:
    template_str = None
else:
  config = 'TPOT NN'
  if t1 == 'lr':
    template_str = 'Selector-Transformer-tpot.builtins.PytorchLRClassifier'
  elif t1 == 'mlp':
    template_str = 'Selector-Transformer-tpot.builtins.PytorchMLPClassifier'
  else:
    template_str = None

print(">> TEMPLATE STRING:      ", template_str)

X_train, X_test, y_train, y_test = train_test_split(
  X, y.astype(np.float64), train_size=0.8, test_size=0.2
)

clf_t = tpot.TPOTClassifier(
  generations=5,
  population_size=30,
  verbosity=2,
  config_dict=config,
  template=template_str
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