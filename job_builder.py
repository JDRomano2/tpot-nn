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
  'ionosphere',
  'spambase',
  'wine-quality-red',
  'wine-quality-white'
]

#jobname_template = 'eval-{0}-{1}_{2}_{3}'
jobname_prefix = 'tpot-nn'

# Create directories if they don't already exist
Path("./job_files").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./pipelines").mkdir(parents=True, exist_ok=True)
Path("./pmlb_data_cache").mkdir(parents=True, exist_ok=True)

def make_jobfile(dataset, use_template, use_nn, estimator, solo, n_reps=5):
  
  py_cmd = 'python model_script.py --dataset={0}'.format(dataset)
  if use_nn:
    py_cmd += ' --use_nn'
  if use_template:
    py_cmd += ' --use_template'
  if solo:
    py_cmd += ' --solo'
  py_cmd += ' --estimator_select {0}'.format(estimator)
  
  use_nn_str = 'nn' if use_nn else 'no-nn'
  type_str = 'template' if use_template else 'config'
  dset_str = dataset.lower().translate(str.maketrans('', '', string.punctuation))
  solo_str = 'solo' if solo else 'nosolo'

  for rep in range(1, n_reps+1):
    jobname = "{0}_{1}_{2}_{3}_{4}_{5}_rep{6}_{7}".format(
      jobname_prefix, estimator, use_nn_str, solo_str, type_str, dset_str, rep, int(time.time())
    )

    py_cmd_rep = py_cmd + ' --jobname {0}'.format(jobname)

    jobfile_path = 'job_files/{0}.sh'.format(jobname)
    jobfile = open(jobfile_path, 'w')
    jobfile.writelines([
      '#!/bin/bash\n',
      '#BSUB -J {0}\n'.format(jobname),
      '#BSUB -q "epistasis_long"'
      '#BSUB -o logs/{0}.out\n'.format(jobname),
      '#BSUB -e logs/{0}.err\n'.format(jobname),
      '#BSUB -M 10000\n',
      '\n',
      '{0}\n'.format(py_cmd_rep)
    ])
    jobfile.close()

    os.system('bsub < ' + jobfile_path)

for dset in dsets:
  for nn in [True, False]:
    for solo_estimator in [True, False]:
      for use_template in [True, False]:
        for model in ['lr', 'mlp', 'all']:
          if (model == 'all') and (solo_estimator == True):
            # It doesn't make sense to do 'all estimators' and NOT search for the best estimator
            continue
          if (use_template) and (solo_estimator == True):
            # Likewise, template option invalidates solo estimator
            continue
          
          # Allow stacking
          make_jobfile(dataset=dset, use_template=use_template,
                      use_nn=nn, solo=solo_estimator, estimator=model)

          # Don't allow stacking
          make_jobfile(dataset=dset, use_template=use_template,
                      use_nn=nn, solo=solo_estimator, estimator=model)
