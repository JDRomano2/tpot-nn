#!/bin/bash
#BSUB -J tpot_nn_eval
#BSUB -o tpot_nn_eval.%J.%I.out
#BSUB -e tpot_nn_eval.%J.%I.error

jobn=${LSB_JOBINDEX}

echo ${jobn}
sleep 5