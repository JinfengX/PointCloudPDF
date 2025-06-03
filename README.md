<div align="center">

# PDF: A Probability-Driven Framework for Open World 3D Point Cloud Semantic Segmentation

by [Jinfeng Xu](https://jinfengx.github.io/), Siyuan Yang, [Xianzhi Li](https://nini-lxz.github.io/), Yuan Tang, Yixue Hao, Long Hu, Min Chen

[![Conference](https://img.shields.io/badge/CVPR-2024-blue)](https://cvpr.thecvf.com/Conferences/2024/)
[![arXiv](https://img.shields.io/badge/arXiv-2404.00979-b31b1b.svg)](https://arxiv.org/abs/2404.00979)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://github.com/Pointcept/Pointcept"><img alt="Pointcept" src="https://img.shields.io/badge/Forked_from-Pointcept-rgb(72,180,97)?logo=github&style=flat"></a>
<br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.2211.13702-B31B1B.svg)](https://arxiv.org/abs/2211.13702) -->

</div>


<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    width: 98%;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=./docs/teaser.png alt="">
    <br>
</div>


## Introduction
**PointCloudPDF** is the repository for our Computer Vision and Pattern Recognition (CVPR) 2024 paper 'PDF: A Probability-Driven Framework for Open World 3D Point Cloud Semantic Segmentation'.
In this paper, we propose a Probability-Driven Framework (PDF)
for open world semantic segmentation that includes (i) a lightweight U-decoder branch to identify unknown classes by estimating the uncertainties, (ii) a flexible pseudo-labeling scheme to supply geometry features along with probability distribution features of unknown classes by generating pseudo labels, and (iii) an incremental knowledge distillation strategy to incorporate novel classes into the existing knowledge base gradually.
Our framework enables the model to behave like human beings, which could recognize unknown objects and incrementally learn them with the corresponding knowledge.
Experimental results on the S3DIS and ScanNetv2 datasets demonstrate that the proposed PDF outperforms other methods by a large margin in both important tasks of open world semantic segmentation.


## Code structure
We organize our code based on the [Pointcept](https://github.com/Pointcept/Pointcept) which is a powerful and flexible codebase for point cloud perception research.
The directory structure of our project looks like this:
```
│
├── configs                  <- Experiment configs
│   ├── _base_               <- Base configs
│   ├── s3dis                   <- configs for s3dis dataset
│   │   ├── openseg-pt-v1-0-msp    <- open-set segmentation configs for msp method based on the pointTransformer
│   │   └── ... 
│   └── ...
│
├── data                     <- Project data
│   └── ...
│
├── docs                     <- Project documents
│
├── libs                     <- Third party libraries
│
├── pointcept                <- Code of framework 
│   ├── datasets                <- Datasets processing
│   ├── engines                 <- Main procedures of training and evaluation
│   ├── models                  <- Model zoo
│   ├── recognizers             <- recognizers for open-set semantic segmentation
│   └── utils                   <- Utilities of framework 
│
├── scripts                  <- Scripts for training and test
│
├── tools                    <- Entry for program launch
│
├── .gitignore          
└── README.md
```

# Installation

## Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.

## Environment
``` bash
conda create -n pointpdf python=3.8
conda activate pointpdf
conda install ninja==1.11.1 -c conda-forge 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 
conda install scipy==1.9.1 scikit-learn==1.1.2 numpy==1.19.5 mkl==2024.0 -c conda-forge 
conda install pyg pytorch-cluster pytorch-scatter pytorch-sparse -c pyg 
conda install sharedarray tensorboard tensorboardx yapf addict einops plyfile termcolor timm -c conda-forge --no-update-deps 
conda install h5py pyyaml -c anaconda --no-update-deps 
pip install spconv-cu113
pip install torch-points3d
pip install open3d  # for visualization
pip uninstall sharedarray
pip install sharedarray==3.2.1
cd libs/pointops
python setup.py install
cd ../..
cd libs/pointops2
python setup.py install
cd ../..
```

# Data Preparation

### ScanNet v2

The preprocessing supports semantic and instance segmentation for both `ScanNet20`.

- Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
- Run preprocessing code for raw ScanNet as follows:

```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset (output dir).
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```

- (Alternative) The preprocess data can also be downloaded [[here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EREuB1If2DNEjz43-rdaVf4B5toMaIViXv8gEbxr9ydeYA?e=ffXeG4)], please agree the official license before download it.

- (Optional) Download ScanNet Data Efficient files:
```bash
# download-scannet.py is the official download script
# or follow instructions here: https://kaldir.vc.in.tum.de/scannet_benchmark/data_efficient/documentation#download
python download-scannet.py --data_efficient -o ${RAW_SCANNET_DIR}
# unzip downloads
cd ${RAW_SCANNET_DIR}/tasks
unzip limited-annotation-points.zip
unzip limited-bboxes.zip
unzip limited-reconstruction-scenes.zip
# copy files to processed dataset folder
cp -r ${RAW_SCANNET_DIR}/tasks ${PROCESSED_SCANNET_DIR}
```

- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset.
mkdir data
ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

### S3DIS

- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2.zip` file and unzip it.
- Run preprocessing code for S3DIS as follows:

```bash
# S3DIS_DIR: the directory of downloaded Stanford3dDataset_v1.2 dataset.
# RAW_S3DIS_DIR: the directory of Stanford2d3dDataset_noXYZ dataset. (optional, for parsing normal)
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset (output dir).

# S3DIS without aligned angle
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR}
# S3DIS with aligned angle
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --align_angle
# S3DIS with normal vector (recommended, normal is helpful)
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --parse_normal
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --align_angle --parse_normal
```

- (Alternative) The preprocess data can also be downloaded [[here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/ERtd0QAyLGNMs6vsM4XnebcBseQ8YTL0UTrMmp11PmQF3g?e=MsER95
)] (with normal vector and aligned angle), please agree with the official license before downloading it.

- Link processed dataset to codebase.
```bash
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
mkdir data
ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
```

# Quick Start

## Training

- **PointTransformer on S3DIS dataset**
``` bash
export PYTHONPATH=./ && export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# open-set segmentation with msp method
python tools/train.py --config-file configs/s3dis/openseg-pt-v1-0-msp.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
# open-set segmentation with our method (training from scratch)
python tools/train.py --config-file configs/s3dis/openseg-pt-v1-0-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```
The msp method does not make changes to the backbone, which only differs from semantic segmentation in the evaluation process. Our method trains open-set segmentation model by finetuning the semantic segmentation model. Therefore, our method can resume training from msp checkpoint directly:
```bash
# open-set segmentation with our method (resume training from msp checkpoint)
python tools/train.py --config-file configs/s3dis/openseg-pt-v1-0-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${MSP_CHECKPOINT_PATH}
```

- **PointTransformer on ScannetV2 dataset**
``` bash
export PYTHONPATH=./ && export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# open-set segmentation with msp method
python tools/train.py --config-file configs/scannet/openseg-pt-v1-0-msp.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
# open-set segmentation with our method (training from scratch)
python tools/train.py --config-file configs/scannet/openseg-pt-v1-0-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
# open-set segmentation with our method (resume training from msp checkpoint)
python tools/train.py --config-file configs/scannet/openseg-pt-v1-0-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${MSP_CHECKPOINT_PATH}
```

- **StratifiedTransformer on S3DIS dataset**

``` bash
export PYTHONPATH=./ && export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# open-set segmentation with msp method
python tools/train.py --config-file configs/s3dis/openseg-st-v1m1-0-origin-msp.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
# open-set segmentation with our method (training from scratch)
python tools/train.py --config-file configs/s3dis/openseg-st-v1m1-0-origin-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

- **StratifiedTransformer on ScannetV2 dataset**
``` bash
export PYTHONPATH=./ && export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# open-set segmentation with msp method
python tools/train.py --config-file configs/scannet/openseg-st-v1m1-0-origin-msp.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
# open-set segmentation with our method (training from scratch)
python tools/train.py --config-file configs/scannet/openseg-st-v1m1-0-origin-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
# open-set segmentation with our method (resume training from msp checkpoint)
python tools/train.py --config-file configs/scannet/openseg-st-v1m1-0-origin-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${MSP_CHECKPOINT_PATH}
```

## Evaluation
The evaluation results can be obtained by appending parameter `eval_only` to training command. For example:
```bash
export PYTHONPATH=./ && export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# evaluate the msp method
python tools/train.py --config-file configs/scannet/openseg-pt-v1-0-msp.py --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH} eval_only=True
# evaluate our method
python tools/train.py --config-file configs/scannet/openseg-pt-v1-0-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options weight=${MSP_CHECKPOINT_PATH} save_path=${SAVE_PATH} eval_only=True
```
Note that, these are not precise evaluation results. To test the model, please refer to [Test](#test).

## Test
To perform precise test:
```bash
# test our method on s3dis dataset
python tools/test.py --config-file configs/s3dis/openseg-pt-v1-0-pointpdf-v1m1-base.py --num-gpus ${NUM_GPU} --options weight=${MSP_CHECKPOINT_PATH} save_path=${SAVE_PATH}
```

## Trained checkpoints
| Dataset | Model |Checkpoints | AUPR | AUROC | mIoU |
| :---: |:-----------:|:-----------:|:----------------:|:----:|:----:|
| ScanNetv2 | StratifiedTransformer | [Google Drive](https://drive.google.com/file/d/15XuqMKJy4A625E63Cazc_a9I9LViCgMP/view?usp=sharing) | 68.9 | 91.3 | 64.5 |

# Acknowledgements

```
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```
If you have any questions, please contact <a href="mailto:jinfengxu.edu@gmail.com">jinfengxu.edu@gmail.com</a>.
