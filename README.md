# APES: Attention-based Point Cloud Edge Sampling

<p>
<a href="https://arxiv.org/pdf/2302.14673.pdf">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a>
<a href="https://junweizheng93.github.io/publications/APES/APES.html">
    <img src="https://img.shields.io/badge/Project-Homepage-red" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" /></a>
<a href="https://mmengine.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Framework-MMEngine-ff69b4" /></a>
<a href="https://github.com/JunweiZheng93/APES/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
</p>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention-based-point-cloud-edge-sampling/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=attention-based-point-cloud-edge-sampling) <br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention-based-point-cloud-edge-sampling/3d-part-segmentation-on-shapenet-part)](https://paperswithcode.com/sota/3d-part-segmentation-on-shapenet-part?p=attention-based-point-cloud-edge-sampling)

## Homepage

This project is selected as a Highlight at CVPR 2023! For more information about the project, please refer to our [project homepage](https://junweizheng93.github.io/publications/APES/APES.html).


## Prerequisites

Install all necessary packages using:

```shell
conda create -n APES python=3.9 -y
conda activate APES
conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```


## Data

Download and preprocess the data using:

```shell
python utils/download_modelnet.py  # for classification
python utils/download_shapenet.py  # for segmentation
```


## Train

Train models from scratch using:

```shell
# using single GPU
# command: bash utils/single_gpu_train.sh cfg_file
bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py  # for classification using local-based downsampling
bash utils/single_gpu_train.sh configs/apes/apes_cls_global-modelnet-200epochs.py  # for classification using global-based downsampling
bash utils/single_gpu_train.sh configs/apes/apes_seg_local-shapenet-200epochs.py  # for segmentation using local-based downsampling
bash utils/single_gpu_train.sh configs/apes/apes_seg_global-shapenet-200epochs.py  # for segmentation using global-based downsampling

# using multiple GPUs 
# command: bash utils/dist_train.sh cfg_file num_gpus
bash utils/dist_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py 2  # for classification using local-based downsampling
bash utils/dist_train.sh configs/apes/apes_cls_global-modelnet-200epochs.py 2  # for classification using global-based downsampling
bash utils/dist_train.sh configs/apes/apes_seg_local-shapenet-200epochs.py 2  # for segmentation using local-based downsampling
bash utils/dist_train.sh configs/apes/apes_seg_global-shapenet-200epochs.py 2  # for segmentation using global-based downsampling
```


## Test

Test model with checkpoint using:

```shell
# using single GPU
# command: bash utils/single_gpu_test.sh cfg_file ckpt_path
bash utils/single_gpu_test.sh configs/apes/apes_cls_local-modelnet-200epochs.py ckpt_path  # for classification using local-based downsampling
bash utils/single_gpu_test.sh configs/apes/apes_cls_global-modelnet-200epochs.py ckpt_path # for classification using global-based downsampling
bash utils/single_gpu_test.sh configs/apes/apes_seg_local-shapenet-200epochs.py ckpt_path  # for segmentation using local-based downsampling
bash utils/single_gpu_test.sh configs/apes/apes_seg_global-shapenet-200epochs.py ckpt_path # for segmentation using global-based downsampling

# using multiple GPUs 
# command: bash utils/dist_test.sh cfg_file ckpt_path num_gpus
bash utils/dist_test.sh configs/apes/apes_cls_local-modelnet-200epochs.py ckpt_path 2  # for classification using local-based downsampling
bash utils/dist_test.sh configs/apes/apes_cls_global-modelnet-200epochs.py ckpt_path 2  # for classification using global-based downsampling
bash utils/dist_test.sh configs/apes/apes_seg_local-shapenet-200epochs.py ckpt_path 2  # for segmentation using local-based downsampling
bash utils/dist_test.sh configs/apes/apes_seg_global-shapenet-200epochs.py ckpt_path 2  # for segmentation using global-based downsampling
```


## Visualization

Visualize results with checkpoint using:

```shell
# using single GPU
# command: bash utils/single_gpu_test.sh cfg_file ckpt_path -vis
bash utils/single_gpu_test.sh configs/apes/apes_cls_local-modelnet-200epochs.py ckpt_path -vis  # for classification using local-based downsampling
bash utils/single_gpu_test.sh configs/apes/apes_cls_global-modelnet-200epochs.py ckpt_path -vis  # for classification using global-based downsampling
bash utils/single_gpu_test.sh configs/apes/apes_seg_local-shapenet-200epochs.py ckpt_path -vis  # for segmentation using local-based downsampling
bash utils/single_gpu_test.sh configs/apes/apes_seg_global-shapenet-200epochs.py ckpt_path -vis  # for segmentation using global-based downsampling

# using multiple GPUs 
# command: bash utils/dist_test.sh cfg_file ckpt_path num_gpus -vis
bash utils/dist_test.sh configs/apes/apes_cls_local-modelnet-200epochs.py ckpt_path 2 -vis  # for classification using local-based downsampling
bash utils/dist_test.sh configs/apes/apes_cls_global-modelnet-200epochs.py ckpt_path 2 -vis  # for classification using global-based downsampling
bash utils/dist_test.sh configs/apes/apes_seg_local-shapenet-200epochs.py ckpt_path 2 -vis  # for segmentation using local-based downsampling
bash utils/dist_test.sh configs/apes/apes_seg_global-shapenet-200epochs.py ckpt_path 2 -vis  # for segmentation using global-based downsampling
```


## Citation

If you are interested in this work, please cite as below:

```text
@InProceedings{Wu_2023_CVPR,
    author    = {Wu, Chengzhi and Zheng, Junwei and Pfrommer, Julius and Beyerer, J\"urgen},
    title     = {Attention-Based Point Cloud Edge Sampling},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {5333-5343}
}
```
