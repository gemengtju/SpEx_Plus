#!/usr/bin/env bash

set -eu

#cpt_dir=exp/conv_tasnet
#epochs=100
# constrainted by GPU number & memory
#batch_size=20
#cache_size=16

#[ $# -ne 2 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1

#./nnet/train.py --gpu "1,2,3,4,5,7" --epochs $epochs --batch-size $batch_size --checkpoint $cpt_dir/conv-net
./nnet/separate.py
