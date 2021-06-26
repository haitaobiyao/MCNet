# SCUT-Seg Dataset

A new benchmark dataset for thermal image semantic segmentation

### Description

<img src=".\example\example.png" alt="example"  />

The SCUT-Seg Datasets is a large semantic segmentation dataset for thermal image.  The image sequences are mainly from [SCUT FIR Pedestrian Dataset](https://github.com/SCUT-CV/SCUT_FIR_Pedestrian_Dataset). To ensure the diversity of the data, we further collected some images captured in the summer. Finally, we picked out 2,010 images with large scene gap as our final label images, and divide the objects in the image into 10 categories, including background, road, person, rider, car, truck, fence, tree, bus and pole.

### Download

Image sequences and annotations

- [BaiduYun](https://pan.baidu.com/s/1QvHukmTTm0kNiroKK-72uQ)  code: 1234

- GoogleDrive (Coming soon)

### Annotations

The public [annotation tool](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor) for driving scenes provided by hitachi automotive and industry lab is used to annotate our dataset. The label details of each category are as follows:

| Name       | Ground Truth value | Color           |
| ---------- | ------------------ | --------------- |
| background | 0                  | [0, 0, 0]       |
| road       | 7                  | [128, 64, 128]  |
| person     | 24                 | [60, 20, 220]   |
| rider      | 25                 | [0, 0, 255]     |
| car        | 26                 | [142, 0, 0]     |
| truck      | 27                 | [70, 0, 0]      |
| fence      | 13                 | [153, 153, 190] |
| tree       | 21                 | [35, 142, 107]  |
| bus        | 28                 | [100, 60, 0]    |
| pole       | 17                 | [153, 153, 153] |



### Citing SCUT-Seg Dataset

Please consider citing our paper in your publications if you find SCUT-Seg Dataset helps your research:
```
@article{XIONG2021103628,
title = {MCNet: Multi-level Correction Network for thermal image semantic segmentation of nighttime driving scene},
journal = {Infrared Physics & Technology},
pages = {103628},
year = {2021},
issn = {1350-4495},
doi = {https://doi.org/10.1016/j.infrared.2020.103628},
author = {Haitao Xiong and Wenjie Cai and Qiong Liu},
}
```
# MCNet
### Background

MCNet is a network architecture proposed in our paper [MCNet: Multi-level Correction Network for thermal image semantic segmentation of nighttime driving scene](https://www.sciencedirect.com/science/article/pii/S1350449520306769),

by Haitao Xiong, Wenjie Cai, Qiong Liu from South China University of Technology.

### Prerequisites

- PyTorch 1.0
  - `pip3 install torch torchvision`
- Easydict
  - `pip3 install easydict`
- [Apex](https://nvidia.github.io/apex/index.html)
- tqdm
  - `pip3 install tqdm`

## Training

1. modify the `config.py` according to your requirements

2. train a network:

   ```
   export NGPUS=8
   python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
   ```

### Inference

1. evaluate a trained network on the validation set:

   ```
   python eval.py
   ```

2. input arguments:

   ```
   usage: -e epoch_idx -d device_idx [--verbose ] 
   [--show_image] [--save_path Pred_Save_Path]
   ```

### Comparisons with SOTAs

The following results are referred from our paper. Recently, we found that training with [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) can achieve higher performance on the SCUT-Seg dataset. Therefore, we recommend researchers  to train the SCUT-Seg dataset with [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

| Method     | Backbone   | mIoU(%)   |
| ---------- | ---------- | --------- |
| U-Net      | ----       | 41.53     |
| FCN-32s    | ResNet-101 | 48.43     |
| FCN-16s    | ResNet-101 | 59.65     |
| DFN        | ResNet-101 | 64.55     |
| BiseNet    | ResNet-101 | 63.03     |
| PSPNet     | ResNet-101 | 67.04     |
| PSANet     | ResNet-101 | 67.32     |
| DeepLabV3  | ResNet-101 | 64.64     |
| DeepLabV3+ | ResNet-101 | 68.00     |
| **MCNet**  | ResNet-101 | **69.79** |

### Download Model

[MCNet-ResNet101](https://pan.baidu.com/s/1wPrpyJrj-0y576bGsCe0Ag) code:1234

### Code Borrowed From

Our code mainly refers from [torchseg](https://github.com/ycszen/TorchSeg).

### Contact

Please contact me if you get stuck anywhere.
