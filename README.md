PyTorch implementation for [Cross-modal Active Complementary Learning with Self-refining Correspondence](https://openreview.net/pdf?id=UBBeUjTja8) (NeurIPS 2023).

## Introduction

## Requirements

- Python 3.8
- PyTorch (>=1.10.0)
- numpy
- scikit-learn
- TensorBoard
- Punkt Sentence Tokenizer:
  
```
import nltk
nltk.download()
> d punkt
```
  
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
 
