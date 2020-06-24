#!/usr/bin/env bash

set -eu

cpt_dir=exp/conv_tasnet
epochs=100
# constrainted by GPU number & memory
<<<<<<< HEAD
batch_size=4
=======
batch_size=2
>>>>>>> aeacd7d903e14f03861e90433a1cd1ea49a21b4e
cache_size=16

#[ $# -ne 2 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1

<<<<<<< HEAD
./nnet/train.py --gpu "1,2" --epochs $epochs --batch-size $batch_size --checkpoint $cpt_dir/conv-net 
=======
./nnet/train.py --gpu "3" --epochs $epochs --batch-size $batch_size --checkpoint $cpt_dir/conv-net 
>>>>>>> aeacd7d903e14f03861e90433a1cd1ea49a21b4e
