PyTorch implementation for [Cross-modal Active Complementary Learning with Self-refining Correspondence](https://openreview.net/pdf?id=UBBeUjTja8) (NeurIPS 2023).

## Introduction
 
## Datasets

Our directory structure of ```data```.
```
data
├── f30k_precomp # pre-computed BUTD region features for Flickr30K, provided by SCAN
│     ├── train_ids.txt
│     ├── train_caps.txt
│     ├── ......
│
├── coco_precomp # pre-computed BUTD region features for COCO, provided by SCAN
│     ├── train_ids.txt
│     ├── train_caps.txt
│     ├── ......
│
├── cc152k_precomp # pre-computed BUTD region features for cc152k, provided by NCR
│     ├── train_ids.txt
│     ├── train_caps.tsv
│     ├── ......
│
└── vocab  # vocab files provided by SCAN and NCR
      ├── f30k_precomp_vocab.json
      ├── coco_precomp_vocab.json
      └── cc152k_precomp_vocab.json
```

### MS-COCO and Flickr30K
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies.

### CC152K
Following [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR), we use a subset of [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions) (CC), named CC152K. CC152K contains training 150,000 samples from the CC training split, 1,000 validation samples and 1,000 testing samples from the CC validation split.

[Download Dataset](https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar)

## Training and Evaluation

### Training new models
```
sh train.sh


#!/bin/bash

# More recommended hyperparameter settings can be found the in the Table 1 at https://openreview.net/attachment?id=UBBeUjTja8&name=supplementary_material

filename=f30k
module_name=SGR
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


CUDA_VISIBLE_DEVICES=$gpus python train.py --val_step 1000 --gpu $gpus --alpha $alpha   --data_name ${filename}_precomp \
    --tau $tau --data_path $data_path --vocab_path $vocab_path   --warm_epoch $warm_epoch\
    --schedules $schedules --lr_update $lr_update --noise_file $noise_file --module_name $module_name --folder_name $folder_name --noise_ratio $noise_rate  
 

```

### Evaluation
```
python eval.py
```
## Citation
If CRCL is useful for your research, please cite the following paper:
```
@inproceedings{
  qin2023crossmodal,
  title={Cross-modal Active Complementary Learning with Self-refining Correspondence},
  author={Yang Qin and Yuan Sun and Dezhong Peng and Joey Tianyi Zhou and Xi Peng and Peng Hu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=UBBeUjTja8}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 
