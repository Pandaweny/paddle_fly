# [AI训练营]paddleclas实现图像分类baseline大作业

多彩缤纷的世界，相信大家的相册里都有着许许多多的美景，这次我们要做的是对美景图片进行场景分类。

# 一、项目背景

由于我本人比较喜欢拍摄风景图片，但懒于分类，便想到运用paddleclas进行场景分类，来解决这一问题。



# 二、数据集简介


项目使用了paddle的场景5分类数据集，一共有3499条数据，都是风景图片。
## 数据集介绍

<font size="3px" color="red">本次大数据集有五个可供大家选择。分别是：</font>   
1. 猫12分类
2. 垃圾40分类

<font size="3px" color="red"> 3. 场景5分类</font> 


4. 食物5分类
5. 蝴蝶20分类


**我要做的是场景分类，所以选择的data是“场景5分类”**



```python
# 先导入库
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
import random
```


```python
# 忽略（垃圾）警告信息
# 在python中运行代码经常会遇到的情况是——代码可以正常运行但是会提示警告，有时特别讨厌。
# 那么如何来控制警告输出呢？其实很简单，python通过调用warnings模块中定义的warn()函数来发出警告。我们可以通过警告过滤器进行控制是否发出警告消息。
import warnings
warnings.filterwarnings("ignore")
```

### 2.1 解压数据集，查看数据的结构


```python
# 项目挂载的数据集先解压出来，待解压完毕，刷新后可发现左侧文件夹根目录出现五个zip
!unzip -oq /home/aistudio/data/data103736/五种图像分类数据集.zip
```

左侧可以看到如图所示五个zip    
![](https://ai-studio-static-online.cdn.bcebos.com/f8bc5b21a0ba49b4b78b6e7b18ac0341dfb14cf545b14c83b1f597b6ee8109bb)



```python
# 本次项目选择场景5分类作为数据集
# (此处需要你根据自己的选择进行解压对应的文件)
# 解压完毕左侧出现文件夹，即为需要分类的文件
!unzip -oq /home/aistudio/场景5分类.zip
```


```python
# 查看结构，正为一个类别下有一系列对应的图片
!tree scenes/
```

    5 directories, 3498 files

**五类场景图片**  
1. river
2. lawn
3. church
4. ice
5. desert    

具体结构如下：
```
scenes/
├── river
│   ├── 1005649.jpg
│   ├── 1011328.jpg
│   ├── 101251.jpg
```

### 2.2 拿到总的训练数据txt


```python
import os
# -*- coding: utf-8 -*-
# 根据官方paddleclas的提示，我们需要把图像变为两个txt文件
# train_list.txt（训练集）
# val_list.txt（验证集）
# 先把路径搞定 比如：foods/beef_carpaccio/855780.jpg ,读取到并写入txt 

# 根据左侧生成的文件夹名字来写根目录
dirpath = "scenes"
# 先得到总的txt后续再进行划分，因为要划分出验证集，所以要先打乱，因为原本是有序的
def get_all_txt():
    all_list = []
    i = 0 # 标记总文件数量
    j = 0 # 标记文件类别
    for root,dirs,files in os.walk(dirpath): # 分别代表根目录、文件夹、文件
        for file in files:
            i = i + 1 
            # 文件中每行格式： 图像相对路径      图像的label_id（数字类别）（注意：中间有空格）。              
            imgpath = os.path.join(root,file)
            all_list.append(imgpath+" "+str(j)+"\n")

        j = j + 1

    allstr = ''.join(all_list)
    f = open('all_list.txt','w',encoding='utf-8')
    f.write(allstr)
    return all_list , i
all_list,all_lenth = get_all_txt()
print("数据集大小：",all_lenth)
```

    数据集大小： 3499


### 2.3 数据打乱


```python
# 把数据打乱
all_list = shuffle(all_list)
allstr = ''.join(all_list)
f = open('all_list.txt','w',encoding='utf-8')
f.write(allstr)
print("打乱成功，并重新写入文本")
```

    打乱成功，并重新写入文本


### 2.4 数据划分


```python
# 按照比例划分数据集 食品的数据有3499张图片，不算大数据，一般9:1即可
train_size = int(all_lenth * 0.9)
train_list = all_list[:train_size]
val_list = all_list[train_size:]

print(len(train_list))
print(len(val_list))
```

    3149
    350



```python
# 运行cell，生成训练集txt 
train_txt = ''.join(train_list)
f_train = open('train_list.txt','w',encoding='utf-8')
f_train.write(train_txt)
f_train.close()
print("train_list.txt 生成成功！")

# 运行cell，生成验证集txt
val_txt = ''.join(val_list)
f_val = open('val_list.txt','w',encoding='utf-8')
f_val.write(val_txt)
f_val.close()
print("val_list.txt 生成成功！")
```

    train_list.txt 生成成功！
    val_list.txt 生成成功！


## 3 安装paddleclas

数据集核实完搞定成功的前提下，可以准备更改原文档的参数进行实现自己的图片分类了！

这里采用paddleclas的2.2版本，好用！


```python
# 先把paddleclas安装上再说
# 安装paddleclas以及相关三方包(好像studio自带的已经够用了，无需安装了)
!git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.2
# 我这里安装相关包时，花了30几分钟还有错误提示，不管他即可
#!pip install --upgrade -r PaddleClas/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

    fatal: destination path 'PaddleClas' already exists and is not an empty directory.



```python
#因为后续paddleclas的命令需要在PaddleClas目录下，所以进入PaddleClas根目录，执行此命令
%cd PaddleClas
!ls
```

    /home/aistudio/PaddleClas
    dataset  hubconf.py   MANIFEST.in    README_ch.md  requirements.txt
    deploy	 __init__.py  paddleclas.py  README_en.md  setup.py
    docs	 LICENSE      ppcls	     README.md	   tools



```python
# 将图片移动到paddleclas下面的数据集里面
# 至于为什么现在移动，也是我的一点小技巧，防止之前移动的话，生成的txt的路径是全路径，反而需要去掉路径的一部分
!mv ../scenes/ dataset/
```


```python
# 挪动文件到对应目录
!mv ../all_list.txt dataset/scenes
!mv ../train_list.txt dataset/scenes
!mv ../val_list.txt dataset/scenes
```

### 3.1 修改配置文件
#### 3.1.1
主要是以下几点：分类数、图片总量、训练和验证的路径、图像尺寸、数据预处理、训练和预测的num_workers: 0
```
分类数：6类（包含0类）
图片总量：3499
训练路径：./dataset/scenes/train_list.txt
验证路径：./dataset/scenes/val_list.txt
图像尺寸：[3, 224, 224] 
```
路径如下：
>PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml

<font size="3px" color="red">（主要的参数已经进行注释，一定要过一遍）</font>
```
调参修改的参数：
训练轮次epochs: 40 
使用网络name: ResNet50_vc
学习率learning_rate: 0.0125
相关路径path: ./dataset/scenes
```
```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  output_dir: ./output/
  # 使用GPU训练
  device: gpu
  # 每几个轮次保存一次
  save_interval: 1 
  eval_during_train: True
  # 每几个轮次验证一次
  eval_interval: 1 
  # 训练轮次
  epochs: 40 
  print_batch_step: 1
  use_visualdl: True #开启可视化（目前平台不可用）
  # used for static mode and model export
  # 图像大小
  image_shape: [3, 224, 224] 
  save_inference_dir: ./inference
  # training model under @to_static
  to_static: False

# model architecture
Arch:
  # 采用的网络
  name: ResNet50_vc
  # 类别数 多了个0类 0-5 (0无用) 
  class_num: 6 
 
# loss function config for traing/eval process
Loss:
  Train:

    - CELoss: 
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0


Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Piecewise
    learning_rate: 0.0125
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
  regularizer:
    name: 'L2'
    coeff: 0.0005


# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的训练集文本路径
      cls_label_path: ./dataset/scenes/train_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

  Eval:
    dataset: 
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的验证集文本路径
      cls_label_path: ./dataset/scenes/val_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

Infer:
  infer_imgs: data/photography-1628649773318-753.jpg
  batch_size: 10
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    # 输出的可能性最高的前topk个
    topk: 5
    # 标签文件 需要自己新建文件
    class_id_map_file: ./dataset/label_list.txt

Metric:
  Train:
    - TopkAcc:
        topk: [1, 5]
  Eval:
    - TopkAcc:
        topk: [1, 5]
```
#### 3.1.2 标签文件
这个是在预测时生成对照的依据，在上个文件有提到这个
```
# 标签文件 需要自己新建文件
    class_id_map_file: dataset/label_list.txt
```
按照对应的进行编写：   

![](https://ai-studio-static-online.cdn.bcebos.com/0e40a0afaa824ba9b70778aa7931a3baf2a421bcb81b4b0f83632da4e4ddc0ef)  

如食品分类(要对照之前的txt的类别确认无误) 
```
1 church
2 lawn
3 river
4 ice
5 desert
```
![](pictures/J1%EXE95`T]L]YDHW382DOI.png)




![](https://ai-studio-static-online.cdn.bcebos.com/bd764090d0b547afb04dd7bce6a07afa1311241145d740c48452eab09eaab0ad)


### 3.2 开始训练


```python
# 提示，运行过程中可能存在坏图的情况，但是不用担心，训练过程不受影响。
# epochs=40
!python3 tools/train.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```

    [2021/08/11 11:30:37] root INFO: Already save model in ./output/ResNet50/latest

### 3.3 预测一张


```python
# 更换为你训练的网络，需要预测的文件，上面训练所得到的的最优模型文件
# 我这里是不严谨的，直接使用训练集的图片进行验证，大家可以去百度搜一些相关的图片传上来，进行预测
!python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml \
    -o Infer.infer_imgs=data/photography-1628649773318-753.jpg \
    -o Global.pretrained_model=output/ResNet50/best_model
```

    /home/aistudio/PaddleClas/ppcls/arch/backbone/model_zoo/vision_transformer.py:15: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Callable
    [2021/08/11 11:33:36] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2021/08/11 11:33:36] root INFO: Arch : 
    [2021/08/11 11:33:36] root INFO:     class_num : 6
    [2021/08/11 11:33:36] root INFO:     name : ResNet50
    [2021/08/11 11:33:36] root INFO: DataLoader : 
    [2021/08/11 11:33:36] root INFO:     Eval : 
    [2021/08/11 11:33:36] root INFO:         dataset : 
    [2021/08/11 11:33:36] root INFO:             cls_label_path : ./dataset/scenes/val_list.txt
    [2021/08/11 11:33:36] root INFO:             image_root : ./dataset/
    [2021/08/11 11:33:36] root INFO:             name : ImageNetDataset
    [2021/08/11 11:33:36] root INFO:             transform_ops : 
    [2021/08/11 11:33:36] root INFO:                 DecodeImage : 
    [2021/08/11 11:33:36] root INFO:                     channel_first : False
    [2021/08/11 11:33:36] root INFO:                     to_rgb : True
    [2021/08/11 11:33:36] root INFO:                 ResizeImage : 
    [2021/08/11 11:33:36] root INFO:                     resize_short : 256
    [2021/08/11 11:33:36] root INFO:                 CropImage : 
    [2021/08/11 11:33:36] root INFO:                     size : 224
    [2021/08/11 11:33:36] root INFO:                 NormalizeImage : 
    [2021/08/11 11:33:36] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/11 11:33:36] root INFO:                     order : 
    [2021/08/11 11:33:36] root INFO:                     scale : 1.0/255.0
    [2021/08/11 11:33:36] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/11 11:33:36] root INFO:         loader : 
    [2021/08/11 11:33:36] root INFO:             num_workers : 0
    [2021/08/11 11:33:36] root INFO:             use_shared_memory : True
    [2021/08/11 11:33:36] root INFO:         sampler : 
    [2021/08/11 11:33:36] root INFO:             batch_size : 128
    [2021/08/11 11:33:36] root INFO:             drop_last : False
    [2021/08/11 11:33:36] root INFO:             name : DistributedBatchSampler
    [2021/08/11 11:33:36] root INFO:             shuffle : True
    [2021/08/11 11:33:36] root INFO:     Train : 
    [2021/08/11 11:33:36] root INFO:         dataset : 
    [2021/08/11 11:33:36] root INFO:             cls_label_path : ./dataset/scenes/train_list.txt
    [2021/08/11 11:33:36] root INFO:             image_root : ./dataset/
    [2021/08/11 11:33:36] root INFO:             name : ImageNetDataset
    [2021/08/11 11:33:36] root INFO:             transform_ops : 
    [2021/08/11 11:33:36] root INFO:                 DecodeImage : 
    [2021/08/11 11:33:36] root INFO:                     channel_first : False
    [2021/08/11 11:33:36] root INFO:                     to_rgb : True
    [2021/08/11 11:33:36] root INFO:                 ResizeImage : 
    [2021/08/11 11:33:36] root INFO:                     resize_short : 256
    [2021/08/11 11:33:36] root INFO:                 CropImage : 
    [2021/08/11 11:33:36] root INFO:                     size : 224
    [2021/08/11 11:33:36] root INFO:                 RandFlipImage : 
    [2021/08/11 11:33:36] root INFO:                     flip_code : 1
    [2021/08/11 11:33:36] root INFO:                 NormalizeImage : 
    [2021/08/11 11:33:36] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/11 11:33:36] root INFO:                     order : 
    [2021/08/11 11:33:36] root INFO:                     scale : 1.0/255.0
    [2021/08/11 11:33:36] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/11 11:33:36] root INFO:         loader : 
    [2021/08/11 11:33:36] root INFO:             num_workers : 0
    [2021/08/11 11:33:36] root INFO:             use_shared_memory : True
    [2021/08/11 11:33:36] root INFO:         sampler : 
    [2021/08/11 11:33:36] root INFO:             batch_size : 128
    [2021/08/11 11:33:36] root INFO:             drop_last : False
    [2021/08/11 11:33:36] root INFO:             name : DistributedBatchSampler
    [2021/08/11 11:33:36] root INFO:             shuffle : True
    [2021/08/11 11:33:36] root INFO: Global : 
    [2021/08/11 11:33:36] root INFO:     checkpoints : None
    [2021/08/11 11:33:36] root INFO:     device : gpu
    [2021/08/11 11:33:36] root INFO:     epochs : 40
    [2021/08/11 11:33:36] root INFO:     eval_during_train : True
    [2021/08/11 11:33:36] root INFO:     eval_interval : 1
    [2021/08/11 11:33:36] root INFO:     image_shape : [3, 224, 224]
    [2021/08/11 11:33:36] root INFO:     output_dir : ./output/
    [2021/08/11 11:33:36] root INFO:     pretrained_model : output/ResNet50/best_model
    [2021/08/11 11:33:36] root INFO:     print_batch_step : 1
    [2021/08/11 11:33:36] root INFO:     save_inference_dir : ./inference
    [2021/08/11 11:33:36] root INFO:     save_interval : 1
    [2021/08/11 11:33:36] root INFO:     to_static : False
    [2021/08/11 11:33:36] root INFO:     use_visualdl : True
    [2021/08/11 11:33:36] root INFO: Infer : 
    [2021/08/11 11:33:36] root INFO:     PostProcess : 
    [2021/08/11 11:33:36] root INFO:         class_id_map_file : ./dataset/label_list.txt
    [2021/08/11 11:33:36] root INFO:         name : Topk
    [2021/08/11 11:33:36] root INFO:         topk : 5
    [2021/08/11 11:33:36] root INFO:     batch_size : 10
    [2021/08/11 11:33:36] root INFO:     infer_imgs : dataset/photography-1628649773318-753.jpg
    [2021/08/11 11:33:36] root INFO:     transforms : 
    [2021/08/11 11:33:36] root INFO:         DecodeImage : 
    [2021/08/11 11:33:36] root INFO:             channel_first : False
    [2021/08/11 11:33:36] root INFO:             to_rgb : True
    [2021/08/11 11:33:36] root INFO:         ResizeImage : 
    [2021/08/11 11:33:36] root INFO:             resize_short : 256
    [2021/08/11 11:33:36] root INFO:         CropImage : 
    [2021/08/11 11:33:36] root INFO:             size : 224
    [2021/08/11 11:33:36] root INFO:         NormalizeImage : 
    [2021/08/11 11:33:36] root INFO:             mean : [0.485, 0.456, 0.406]
    [2021/08/11 11:33:36] root INFO:             order : 
    [2021/08/11 11:33:36] root INFO:             scale : 1.0/255.0
    [2021/08/11 11:33:36] root INFO:             std : [0.229, 0.224, 0.225]
    [2021/08/11 11:33:36] root INFO:         ToCHWImage : None
    [2021/08/11 11:33:36] root INFO: Loss : 
    [2021/08/11 11:33:36] root INFO:     Eval : 
    [2021/08/11 11:33:36] root INFO:         CELoss : 
    [2021/08/11 11:33:36] root INFO:             weight : 1.0
    [2021/08/11 11:33:36] root INFO:     Train : 
    [2021/08/11 11:33:36] root INFO:         CELoss : 
    [2021/08/11 11:33:36] root INFO:             weight : 1.0
    [2021/08/11 11:33:36] root INFO: Metric : 
    [2021/08/11 11:33:36] root INFO:     Eval : 
    [2021/08/11 11:33:36] root INFO:         TopkAcc : 
    [2021/08/11 11:33:36] root INFO:             topk : [1, 5]
    [2021/08/11 11:33:36] root INFO:     Train : 
    [2021/08/11 11:33:36] root INFO:         TopkAcc : 
    [2021/08/11 11:33:36] root INFO:             topk : [1, 5]
    [2021/08/11 11:33:36] root INFO: Optimizer : 
    [2021/08/11 11:33:36] root INFO:     lr : 
    [2021/08/11 11:33:36] root INFO:         decay_epochs : [30, 60, 90]
    [2021/08/11 11:33:36] root INFO:         learning_rate : 0.0125
    [2021/08/11 11:33:36] root INFO:         name : Piecewise
    [2021/08/11 11:33:36] root INFO:         values : [0.1, 0.01, 0.001, 0.0001]
    [2021/08/11 11:33:36] root INFO:     momentum : 0.9
    [2021/08/11 11:33:36] root INFO:     name : Momentum
    [2021/08/11 11:33:36] root INFO:     regularizer : 
    [2021/08/11 11:33:36] root INFO:         coeff : 0.0005
    [2021/08/11 11:33:36] root INFO:         name : L2
    W0811 11:33:36.707304  4336 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0811 11:33:36.713097  4336 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    [2021/08/11 11:33:42] root INFO: train with paddle 2.1.0 and device CUDAPlace(0)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    [{'class_ids': [4, 5, 3, 1, 2], 'scores': [0.81915, 0.14509, 0.02771, 0.00714, 0.00091], 'file_name': 'dataset/photography-1628649773318-753.jpg', 'label_names': ['ice', 'desert', 'river', 'church', 'lawn']}]

## 4 效果展示
运行完成，最后几行会得到结果如下形式：
```
[{'class_ids': [4, 5, 3, 1, 2],
'scores': [0.81915, 0.14509, 0.02771, 0.00714, 0.00091], 
'file_name': 'data/photography-1628649773318-753.jpg', 
'label_names': ['ice', 'desert', 'river', 'church', 'lawn']}]
```
可以发现，预测结果不对，准确率很低，但整体的项目流程你已经掌握了！    
训练轮数还有很大提升空间，自行变动参数直到预测正确为止~    

## 5 总结与升华

通过做图像分类这个项目，我大致了解了项目的整体流程，虽然在实现的过程中遇到了许多问题，比如：模型的选择，参数的优化等等。
不过，最后也是完成了项目，达到了预期的效果。

## 个人简介
Ai Studio个人链接：我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/760853
