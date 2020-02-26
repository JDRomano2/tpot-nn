## Datasets


## Jobs

For each dataset, submit the following jobs with 5 replicates each:

- tpot: LR
- tpot: MLP
- tpot: All modules enabled
- tpot-nn: LR
- tpot-nn: MLP
- tpot-nn: All modules (including NN) enabled

5 replicates on 6 TPOT pipelines for 4 datasets means 120 jobs need to be run.

## Submitting

To submit the entire array of jobs via LSF, use the following command in the repository root directory:

```
$ bsub -J "tpotNnArray[1:120]" < submit_job.sh
```