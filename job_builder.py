import time
import os, sys

dsets = [
  'Hill_Valley_with_noise'
  'breast-cancer-wisconsin',
  'spambase',
  'ionosphere'
]

models = [
  'tpot_lr',
  'tpot_mlp',
  'tpot_all',
  'tpotnn_lr',
  'tpotnn_mlp',
  'tpotnn_all'
]

jobname_template = 'eval-{0}-{1}_{2}_{3}'

for dset in dsets:
  for model in models:
    py_cmd = 'python model_script.py --dataset={0} --model_template={1}'.format(dset, model)
    for rep in range(1,6):  # 5 reps for each dset/model combo
      jobname = jobname_template.format(dset, model, rep, str(time.time()))

      jobfile_path = 'job_files/{0}.sh'.format(jobname)
      jobfile = open(jobname, 'w')
      jobfile.writelines([
        '#!/bin/bash',
        '#BSUB -J {0}'.format(jobname),
        '#BSUB -o logs/{0}.out'.format(jobname),
        '#BSUB -e logs/{0}.err'.format(jobname),
        '#BSUB -M 10000',
        '',
        '{0}'.format(py_cmd)
      ])
      jobfile.close()

      #os.system('bsub < ' + jobfile_path)

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
