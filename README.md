# __[T-PAMI' 2022]  Meta-DETR__ <br> (Official PyTorch Implementation)

[![arXiv](https://img.shields.io/badge/arXiv-2208.00219-b31b1b.svg)](https://arxiv.org/abs/2208.00219)
[![Survey](https://github.com/sindresorhus/awesome/blob/main/media/mentioned-badge.svg)](https://github.com/dk-liang/Awesome-Visual-Transformer)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
[![GitHub license](https://badgen.net/github/license/ZhangGongjie/Meta-DETR)](https://github.com/ZhangGongjie/Meta-DETR/blob/master/LICENSE)

-------

This repository is the official PyTorch implementation of the
T-PAMI 2022 paper "[Meta-DETR: Image-Level Few-Shot Detection with Inter-Class Correlation Exploitation](https://doi.org/10.1109/TPAMI.2022.3195735)" by _Gongjie Zhang, Zhipeng Luo, Kaiwen Cui, Shijian Lu, and Eric P. Xing_. 

<b> [ Important Notice ] </b> &nbsp;&nbsp; Meta-DETR first appeared as a tech report on arXiv.org (https://arxiv.org/abs/2103.11731v2) in 2021. Since its release, we have made substantial improvements to the original version. This repository corresponds to [the final published version accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI) in 2022](https://doi.org/10.1109/TPAMI.2022.3195735). Please kindly be advised to refer to the latest version of the paper.


-------
&nbsp;
## Brief Introduction

Meta-DETR is a state-of-the-art few-shot object detector that performs image-level meta-learning-based prediction and effectively exploits the inter-class correlation to enhance generalization from old knowledge to new classes. Meta-DETR entirely bypasses the proposal quality gap between base and novel classes, thus achieving superior performance than R-CNN-based few-shot object detectors. In addition, Meta-DETR performs meta-learning on a set of support classes at one go, thus effectively leveraging the inter-class correlation for better generalization.

<div align=center>  
<img src='.assets/motivation.jpg' width="60%">
</div>


<div align=center>  
<img src='.assets/MetaDETR_architecture.jpg' width="95%">
</div>

Please check [our T-PAMI paper](https://doi.org/10.1109/TPAMI.2022.3195735) or [its preprint version](https://arxiv.org/abs/2208.00219) for more details.


-------
&nbsp;

## Installation

### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- Ubuntu LTS 18.04
- 8x NVIDIA V100 GPUs (32GB)
- CUDA 10.2
- Python == 3.7
- PyTorch == 1.7.1+cu102, TorchVision == 0.8.2+cu102
- GCC == 7.5.0
- cython, pycocotools, tqdm, scipy

We recommend using the exact setups above. However, other environments (Linux, Python>=3.7, CUDA>=9.2, GCC>=5.4, PyTorch>=1.5.1, TorchVision>=0.6.1) should also work properly.

&nbsp;

### Code Installation

First, clone the repository locally:
```shell
git clone https://github.com/ZhangGongjie/Meta-DETR.git
```

We recommend you to use [Anaconda](https://www.anaconda.com/) to create a conda environment:
```bash
conda create -n meta_detr python=3.7 pip
```

Then, activate the environment:
```bash
conda activate meta_detr
```

Then, install PyTorch and TorchVision:

(preferably using our recommended setups; CUDA version should match your own local environment)
```bash
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.2 -c pytorch
```

After that, install other requirements:
```bash
conda install cython scipy tqdm
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

As Meta-DETR is developed upon Deformable DETR, you need to compile [*Deformable Attention*](https://github.com/fundamentalvision/Deformable-DETR).
```bash
# compile CUDA operators of Deformable Attention
cd Meta-DETR
cd ./models/ops
sh ./make.sh
python test.py  # unit test (should see all checking is True)
```

&nbsp;

### Data Preparation

#### MS-COCO for Few-Shot Object Detection

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    ├── coco_fewshot/        # Few-shot dataset 
    └── coco/                # MS-COCO dataset
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
```

The [`coco_fewshot`](data/coco_fewshot) folder (_already provided in this repo_) contains randomly sampled few-shot datasets as described in [the paper](https://doi.org/10.1109/TPAMI.2022.3195735), including the five data setups with different random seeds. In each K-shot (K=1,3,5,10,30) data setup, we ensure that there are exactly K object instances for each novel class. The numbers of base-class object instances vary.


#### Pascal VOC for Few-Shot Object Detection

We transform the original Pascal VOC dataset format into MS-COCO format for parsing. The transformed Pascal VOC dataset is available for download at [GoogleDrive](https://drive.google.com/file/d/1JCxJ2lmNX5E4YsvAZnngVZ5hQeJU67tj/view?usp=sharing).


After downloading MS-COCO-style Pascal VOC, please organize them as following:

```
code_root/
└── data/
    ├── voc_fewshot_split1/     # VOC Few-shot dataset
    ├── voc_fewshot_split2/     # VOC Few-shot dataset
    ├── voc_fewshot_split3/     # VOC Few-shot dataset
    └── voc/                    # MS-COCO-Style Pascal VOC dataset
        ├── images/
        └── annotations/
            ├── xxxxx.json
            ├── yyyyy.json
            └── zzzzz.json
```

Similarly, the few-shot datasets for Pascal VOC are also provided in this repo ([`voc_fewshot_split1`](data/voc_fewshot_split1), [`voc_fewshot_split2`](data/voc_fewshot_split2), and [`voc_fewshot_split3`](data/voc_fewshot_split3)). For each class split, there are 10 data setups with different random seeds. In each K-shot (K=1,2,3,5,10) data setup, we ensure that there are exactly K object instances for each novel class. The numbers of base-class object instances vary.

----------
&nbsp;

## Usage

### Reproducing Paper Results

All scripts to reproduce results reported in [our T-PAMI paper](https://doi.org/10.1109/TPAMI.2022.3195735)
are stored in [`./scripts`](scripts). The arguments are pretty easy and straightforward to understand. 

Taking MS-COCO as an example, run the following commands to reproduce paper results:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./scripts/run_experiments_coco.sh
```


### To Perform _**Base Training**_

We take MS-COCO as an example. First, create `basetrain.sh` and copy the following commands into it.
```bash
EXP_DIR=exps/coco
BASE_TRAIN_DIR=${EXP_DIR}/base_train
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

python -u main.py \
    --dataset_file coco_base \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 4 \
    --category_codes_cls_loss \
    --epoch 25 \
    --lr_drop_milestones 20 \
    --save_every_epoch 5 \
    --eval_every_epoch 5 \
    --output_dir ${BASE_TRAIN_DIR} \
2>&1 | tee ${BASE_TRAIN_DIR}/log.txt
```
Then, run the commands below to start base training.
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8  ./basetrain.sh
```

### To Perform _**Few-Shot Finetuning**_
We take MS-COCO as an example. First, create `fsfinetune.sh` and copy the following commands into it.
```bash
EXP_DIR=exps/coco
BASE_TRAIN_DIR=${EXP_DIR}/base_train
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

fewshot_seed=01
num_shot=10
epoch=500
lr_drop1=300
lr_drop2=450
FS_FT_DIR=${EXP_DIR}/seed${fewshot_seed}_${num_shot}shot
mkdir ${FS_FT_DIR}

python -u main.py \
    --dataset_file coco_base \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 2 \
    --category_codes_cls_loss \
    --resume ${BASE_TRAIN_DIR}/checkpoint.pth \
    --fewshot_finetune \
    --fewshot_seed ${fewshot_seed} \
    --num_shots ${num_shot} \
    --epoch ${epoch} \
    --lr_drop_milestones ${lr_drop1} ${lr_drop2} \
    --warmup_epochs 50 \
    --save_every_epoch ${epoch} \
    --eval_every_epoch ${epoch} \
    --output_dir ${FS_FT_DIR} \
2>&1 | tee ${FS_FT_DIR}/log.txt
```
Note that you need to add `--fewshot_finetune` to indicate that the training and inference should be conducted on few-shot setups. You also need to specify the number of shots, few-shot random seed, training epoch setups, and the checkpoint file path after base training.
Then, run the commands below to start few-shot finetuning. After finetuning, the program will automatically perform inference on novel classes.
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8  ./fsfinetune.sh
```


### To Perform _**Only Inference**_ After Few-Shot Finetuning

We take MS-COCO as an example. Simply run:
```bash
python -u main.py \
--dataset_file coco_base \
--backbone resnet101 \
--num_feature_levels 1 \
--enc_layers 6 \
--dec_layers 6 \
--hidden_dim 256 \
--num_queries 300 \
--batch_size 2 \
--category_codes_cls_loss \
--resume path/to/checkpoint.pth/generated/by/few-shot-fintuning \
--fewshot_finetune \
--fewshot_seed ${fewshot_seed} \
--num_shots ${num_shot} \
--eval \
2>&1 | tee ./log_inference.txt
```
Note that user should set `--eval` and `--resume path/to/checkpoint.pth/generated/by/few-shot-fintuning` correctly.



-----------
&nbsp;
## Pre-Trained Model Weights

We provide trained model weights after __the base training stage__ for users to finetune.

*All pre-trained model weights are stored in __Google Drive__.*

- __MS-COCO__ after base training:&nbsp;&nbsp; click [here](https://drive.google.com/file/d/19tfI_XNZolDId_G5s45YTgcFKt8Ji7c8/view?usp=sharing) to download.

- __Pascal VOC Split 1__ after base training:&nbsp;&nbsp; click [here](https://drive.google.com/file/d/1e3xHnVVsS3JFNGTfh51xjUPPZVtwTGOq/view?usp=sharing) to download.

- __Pascal VOC Split 2__ after base training:&nbsp;&nbsp; click [here](https://drive.google.com/file/d/1SMOQP-ZKnuIrg3R32a-6FYtA3zkWeNF2/view?usp=sharing) to download.

- __Pascal VOC Split 3__ after base training:&nbsp;&nbsp; click [here](https://drive.google.com/file/d/1EJ6uP3yAequS5Wl3gEDtyxKx8ZfgPhAi/view?usp=sharing) to download.



----------

&nbsp;
## License

The implementation codes of Meta-DETR are released under the MIT license.

Please see the [LICENSE](LICENSE) file for more information.

However, prior works' licenses also apply. It is the users' responsibility to ensure compliance with all license requirements.


------------

&nbsp;
## Citation

If you find Meta-DETR useful or inspiring, please consider citing:

```bibtex
@article{Meta-DETR-2022,
  author={Zhang, Gongjie and Luo, Zhipeng and Cui, Kaiwen and Lu, Shijian and Xing, Eric P.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{Meta-DETR}: Image-Level Few-Shot Detection with Inter-Class Correlation Exploitation}, 
  year={2022},
  doi={10.1109/TPAMI.2022.3195735},
}
```

----------
&nbsp;
## Acknowledgement

Our proposed Meta-DETR is heavily inspired by many outstanding prior works, including [DETR](https://github.com/facebookresearch/detr) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Thank the authors of above projects for open-sourcing their implementation codes!
