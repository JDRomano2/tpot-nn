#!/bin/bash
#BSUB -J tpot_nn_eval
#BSUB -o tpot_nn_eval.%J.out
#BSUB -e tpot_nn_eval.%J.error

jobn=%I

echo ${jobn}
sleep 5