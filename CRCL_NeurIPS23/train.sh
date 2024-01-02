#!/bin/bash

# More recommended hyperparameter settings can be found the in the Table 1 at https://openreview.net/attachment?id=UBBeUjTja8&name=supplementary_material

filename=f30k
module_name=VSEinfty
# VSEinfty SAF SGR
gpus=3
# schedules=30
# schedules='2,2,2,20'
# lr_update=10
# schedules='5,5,5,40'
schedules='5,5,5,30'
lr_update=15
noise_rate=0.8
warm_epoch=2
tau=0.05
alpha=0.8
 
folder_name=./NCR_logs/${filename}_${module_name}_${noise_rate} 

noise_file=./noise_index/f30k_precomp_0.8.npy

data_path='/home_bak/hupeng/data/data'
vocab_path='/home_bak/hupeng/data/vocab'


CUDA_VISIBLE_DEVICES=$gpus python train.py --val_step 1000 --gpu $gpus --alpha $alpha   --data_name ${filename}_precomp --tau $tau --data_path $data_path --vocab_path $vocab_path   --warm_epoch $warm_epoch\
    --schedules $schedules --lr_update $lr_update --noise_file $noise_file --module_name $module_name --folder_name $folder_name --noise_ratio $noise_rate   \
 