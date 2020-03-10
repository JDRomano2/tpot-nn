import time
import string
import os, sys
from pathlib import Path

dsets = [
  'Hill_Valley_with_noise',
  'Hill_Valley_without_noise',
  'breast-cancer-wisconsin',
  'car-evaluation',
  'glass',
  'ionosphere'
  'spambase',
  'wine-quality-red',
  'wine-quality-white'
]

#jobname_template = 'eval-{0}-{1}_{2}_{3}'
jobname_prefix = 'eval_tpotnn'

# Create directories if they don't already exist
Path("./job_files").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./pipelines").mkdir(parents=True, exist_ok=True)
Path("./pmlb_data_cache").mkdir(parents=True, exist_ok=True)

def make_jobfile(dataset, tpot_all, use_template, use_nn, use_classic, estimator, n_reps=5):
  py_cmd = 'python model_script.py --dataset={0}'.format(dataset)
  if use_nn:
    py_cmd += ' --use_nn'
  if use_template:
    py_cmd += ' --use_template'
  if tpot_all:
    py_cmd += ' --tpot_all'
  else:
    # Set options that aren't compatible with '--tpot_all'
    if use_classic:
      py_cmd += ' --use_classic'
    py_cmd += ' --estimator_select {0}'.format(estimator)

  use_nn_str = 'nn' if use_nn else 'no-nn'
  type_str = 'template' if use_template else 'config'
  dset_str = dataset.lower().translate(str.maketrans('', '', string.punctuation))

  for rep in range(1, n_reps+1):
    if tpot_all:
      jobname = "{0}_all_{1}_{2}_{3}_rep{4}_{5}".format(
        jobname_prefix, use_nn_str, type_str, dset_str, rep, int(time.time())
      )
    else:
      jobname = "{0}_{1}_{2}_{3}_{4}_rep{5}_{6}".format(
        jobname_prefix, estimator, use_nn_str, type_str, dset_str, rep, int(time.time())
      )

    jobfile_path = 'job_files/{0}.sh'.format(jobname)
    jobfile = open(jobfile_path, 'w')
    jobfile.writelines([
      '#!/bin/bash\n',
      '#BSUB -J {0}\n'.format(jobname),
      '#BSUB -o logs/{0}.out\n'.format(jobname),
      '#BSUB -e logs/{0}.err\n'.format(jobname),
      '#BSUB -M 10000\n',
      '\n',
      '{0}\n'.format(py_cmd)
    ])
    jobfile.close()

    #os.system('bsub < ' + jobfile_path)

for dset in dsets:
  for nn in [True, False]:
    for classic_tpot in [True, False]:
      for model in ['lr', 'mlp']:
        # Allow stacking
        make_jobfile(dataset=dset, tpot_all=False, use_template=False,
                     use_nn=nn, use_classic=classic_tpot, estimator=model)

        # Don't allow stacking
        make_jobfile(dataset=dset, tpot_all=False, use_template=True,
                     use_nn=nn, use_classic=classic_tpot, estimator=model)

    # See what TPOT does if we give it access to every estimator
    make_jobfile(dataset=dset, tpot_all=True, use_template=False,
                 use_nn=nn, use_classic=classic_tpot, estimator=model)
