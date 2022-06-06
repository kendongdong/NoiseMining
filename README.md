## **Pixel-Level Noise Mining for Weakly Supervised Salient Object Prediction**



### This is a PyTorch implementation of our paper.

## Overview

![avatar](https://github.com/kendongdong/NoiseMining/blob/main/overview.png)



## Prerequisites

- Python 3.8.12

- torch 1.9.0+cu111

- torchvision  0.10.0+cu111

  

## Usage

### 1. Clone the repository
```shell
git https://github.com/kendongdong/NoiseMining.git
cd NoiseMining/
```
### 2. Download the datasets
Download the following datasets and unzip them.

* [Train dataset](https://pan.baidu.com/share/init?surl=hq135pTjbwuda0VMocOsxw) dataset in google drive.
* [Test dataset](https://drive.google.com/drive/folders/1oYPAQzl6-1AeVGP8dJ0IRd6jeVt_T4hK) dataset in google drive.
* The .txt file link for testing and training is [here](https://drive.google.com/drive/folders/1oYPAQzl6-1AeVGP8dJ0IRd6jeVt_T4hK).
### 3. Train
1. Set the  parameters  in`demo_train.py`  correctly, including path and  your training settings. Run train file as follow:

   ```shell
   python demo_train.py
   ```

2. We demo using VGG-19 as network backbone and train with a initial lr of 3e-4 for 30 epoches.

3. After training the result model will be stored under `checkpoint/exp_noiseMining` folder.

### 4. Test
For single dataset testing:  you should set  data path in `demo_test.py` as yours , and run test file as follow :
```shell
python demo_test.py 
```
All results saliency maps will be stored under `'noiseMining/results/'` folders in .png formats.

Thanks to [MOLF](https://github.com/jiwei0921/MoLF), [CORES](https://github.com/UCSC-REAL/cores) and [ProSelfLC](https://github.com/XinshaoAmosWang/ProSelfLC-2021) .

