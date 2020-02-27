import argparse
import tpot
from pmlb import fetch_data

parser = argparse.ArgumentParser(description='Run a single TPOT-NN evaluation job.')
parser.add_argument(
  'dataset',
  choices=[
    'Hill_Valley_with_noise'
    'breast-cancer-wisconsin',
    'spambase',
    'ionosphere'
  ],
  help='PMLB dataset name'
)
parser.add_argument(
  'model_template',
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

data = fetch_data(args.dataset)

template = args.model_template
t0, t1 = tuple(template.split('_'))

if t0 == 'tpot':
  if t1 == 'lr':
    template_str = 'Selector-Transformer-sklearn.linear_model.LogisticRegression'
  elif t1 == 'mlp':
    template_str = 'Selector-Transformer-sklearn.neural_network.MLPClassifier'
  else:
    template_str = 'Selector-Transformer-Classifier'
else:
  if t1 == 'lr':
    template_str = 'Selector-Transformer-sklearn.linear_model.LogisticRegression'
  elif t1 == 'mlp':
    template_str = 'Selector-Transformer-sklearn.neural_network.MLPClassifier'
  else:
    template_str = 'Selector-Transformer-Classifier'



# Example:
# def submitJob(uniquejobname,scratchpath,logpath,runpath,dataset,outfile,algorithm,discthresh,outcomelabel,neighbors):
#     """ Submit Job to the cluster. """
#     jobName = scratchpath+'/'+uniquejobname+'_'+str(time.time())+'_run.sh'
#     shFile = open(jobName, 'w')
#     shFile.write('#!/bin/bash\n')
#     shFile.write('#BSUB -J '+uniquejobname+'_'+str(time.time())+'\n')
#     shFile.write('#BSUB -o ' + logpath+'/'+uniquejobname+'.o\n')
#     shFile.write('#BSUB -M ' + 10000'+'\n')
#     shFile.write('#BSUB -e ' + logpath+'/'+uniquejobname+'.e\n\n')
#     shFile.write('python '+runpath+'/'+'run_scikit-rebate.py '+str(dataset)+' '+str(outfile)+' '+str(algorithm)+' '+str(discthresh)+' '+str(outcomelabel)+' '+str(neighbors)+'\n')
#     shFile.close()
#     os.system('bsub < '+jobName)
