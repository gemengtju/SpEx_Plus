#!/usr/bin/env bash

set -eu

cpt_dir=exp/conv_tasnet
epochs=100
# constrainted by GPU number & memory
batch_size=20
cache_size=16

#[ $# -ne 2 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1

./nnet/train.py --gpu "0,1,2" --epochs $epochs --batch-size $batch_size --checkpoint $cpt_dir/conv-net 
