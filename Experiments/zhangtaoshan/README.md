# 基于目标检测的火灾预防和检测

利用火焰数据集训练一个检测火焰的目标检测模型，基于该检测模型可以进行火灾的监控与报警。

# 一、项目背景

火灾成为了不可忽视的多发性问题，火灾一次次展示了其对于人员伤亡和财产的巨大破坏性，火灾的预防和检测一直是人类与火灾斗争过程中的焦点。传统的火灾检测通过人员不断巡逻，不仅浪费了人力也无法有效地处理大面积的预警情况。随着图像处理技术的发展，可以通过实时回传摄像头的图像数据并对图像作分析与处理，对出现火焰的场景进行快速报警，然后通过人工干预以及处理火灾险情。本项目基于PaddleX实现。

# 二、数据集简介

数据集使用了AI Studio开源的火焰数据集，总共有492幅图像。[数据集地址](https://aistudio.baidu.com/aistudio/datasetdetail/103743)

## 1.数据加载和预处理

首先安装PaddleX并对数据集划分训练集，验证集和测试集。
```shell
# 安装PaddleX
pip install paddlex

# 使用PaddleX将数据集划分为训练集，验证集和测试集
paddlex --split_dataset --format VOC --dataset_dir work/fire --val_value 0.2 --test_value 0.1
```

训练集样本量: 345，验证集样本量: 98，测试集样本量: 49


## 2.数据预处理
在训练模型之前，对目标检测任务的数据进行操作，从而提升模型效果。对训练集和验证集的数据预处理包括：
```python
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])
```
对数据集做完预处理后，开始加载数据集，这里使用PaddleX自带的datasets类。
```python
train_dataset = pdx.datasets.VOCDetection(
    data_dir='work/fire',
    file_list='work/fire/train_list.txt',
    label_list='work/fire/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='work/fire',
    file_list='work/fire/val_list.txt',
    label_list='work/fire/labels.txt',
    transforms=eval_transforms)
```


# 三、模型选择和开发

PaddleX目前内置了Faster RCNN和YOLOv3两种检测算法，及其配套的多种骨干网络。本项目使用基于MobileNetV1的YOLOv3算法。

## 1. 利用PaddleX加载YOLOv3模型
```python
model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='MobileNetV1')
```
## 2.模型训练

```python
model.train(
    num_epochs=100,				
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    learning_rate=0.00025,
    lr_decay_epochs=[60, 80],
    save_dir='output/yolov3_mobilenetv1')
```
最终的输出模型保存在当前目录下的output/yolov3_mobilenetv1文件夹下，bbox_map=52.19973077270447。

## 3. 预测结果可视化
```python
model = pdx.load_model("output/yolov3_mobilenetv1/best_model")
result = model.predict("output/yolov3_mobilenetv1/10.jpg")
pdx.det.visualize("output/yolov3_mobilenetv1/10.jpg", result, threshold=0.1, save_dir="output/yolov3_mobilenetv1")
```
<img src="work/visualize_10.jpg">
由结果可知，该模型没有没得得到充分的训练，可以通过迭代更多的epoch和数据增强等调优操作得到性能更好的模型。

# 四、总结与升华

PaddleX是一个具有高级API的深度学习套件，但同时在我们遇到问题的时候也无法快速有效地定位问题所在。所以，对于这种高级API的使用，最好能够有众多低级API供使用者参考以便于调试。
总之，PaddleX是一个对初学者友好的，可以快速入门深度学习的优秀套件，期待其更好的发展。
