# SCUT-Seg Dataset

A new benchmark dataset for thermal image semantic segmentation

### Description

<img src=".\example\example.png" alt="example"  />

The SCUT-Seg Datasets is a large semantic segmentation for thermal image.  The image sequences are mainly from [SCUT FIR Pedestrian Dataset](https://github.com/SCUT-CV/SCUT_FIR_Pedestrian_Dataset). To ensure the diversity of the data, we further collected some images captured in the summer. Finally, we picked out 2,010 images with large scene gap as our final label images, and divide the objects in the image into 10 categories, including background, road, person, rider, car, truck, fence, tree, bus and pole.

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
# Code
Coming soon
