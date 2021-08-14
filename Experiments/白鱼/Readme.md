# AI入门小白之简单障碍物检测


这个项目应该算是一个简单的入门项目吧 做一个简单的障碍物识别 体会一下整个流程 应该是怎么样的

# 一、项目背景

身为一个想学AI的小白　从此刻开始入门　希望不久之后不会入土　做这个项目主要想入个门　把一个完整的AI项目的流程顺下来

这个项目主要是障碍物的识别 在生活中障碍物的识别 可以运用到很多的地方 比如自动驾驶 但这个项目不足以达到应用的地步 

这个项目仅供小白了解AI项目的流程是怎么样的

# 二、数据集简介

　数据集是我在AI　Studio的数据集大厅找的一个障碍物数据集 数据集里面有很多的不同的障碍物文件 这个数据集里面的图片都已经完成了批注 使用起来很方便 在这个项目 我对这个数据集进行了一些删减　取其中的一部分　进行模型训练及模型预测 如果你使用的是自制数据集 一定要对图片文件进行一个批注 即把你要识别的东西 框出来

 那么接下来我们来开始这个简单的小项目吧

# 三、简单障碍物检测的数据集

## 3.1 查看并解压数据集


```python
%cd /home/aistudio/data/data103919/
!unzip -oq /home/aistudio/data/data103919/简单障碍物.zip -d /home/aistudio/
```

 接下来 我们导入PaddleX模型库 此项目是基于PaddleX完成的 同时后续将数据集处理成VOC格式 用PaddleX里面的工具很好用 一下子就可以处理所有 得到我们想要的VOC格式文件 不用一个一个处理 很方便


```python
!pip install paddlex 
```


```python
#使用PaddleX中的工具可以快速将数据集处理成VOC格式 
!paddlex --split_dataset --format VOC --dataset_dir  /home/aistudio/stop/ --val_value 0.15  --test_value 0.05
```

## 3.2数据的预处理


```python
#数据预处理 这里使用了图像混合、随机像素变换、随机膨胀、随即裁剪、随机水平翻转等数据增强方法。
from paddlex.det import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32)
])
 
eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32),
])
```

## 3.3 定义测试集和验证集


```python
#定义测试集 和验证集 将前边划分好的是数据集用变量来定义，方便后边训练使用

base = '/home/aistudio/work2/mot02'
from paddlex.det import transforms
from random import shuffle
import paddlex as pdx
import os 

base = '/home/aistudio/stop'

train_dataset = pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=os.path.join(base, 'test_list.txt'),
    label_list='/home/aistudio/stop/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=os.path.join(base, 'val_list.txt'),
    label_list='/home/aistudio/stop/labels.txt',
    transforms=eval_transforms)
```

# 四、模型训练

在这里的模型训练 我选择了骨干网络为DarkNet53的YOLO-V3模型


```python
#训练模型

import matplotlib
matplotlib.use('Agg') 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

#num_classes有些模型需要加1 比如faster_rcnn
num_classes = len(train_dataset.labels)

model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')

model.get_model_info()

model.train(
    num_epochs=70,#训练轮数
    train_dataset=train_dataset,
    train_batch_size=1,#一次加载 多少图片
    eval_dataset=eval_dataset,
    learning_rate=3e-5,
    warmup_steps=90,
    warmup_start_lr=0.0,
    save_interval_epochs=7,#每隔七轮保存一次
    lr_decay_epochs=[42, 70],
    save_dir='output/yolov3_darknet53',
    use_vdl=True)
```

# 五、模型预测


```python
#导出训练模型
!paddlex --export_inference --model_dir=/home/aistudio/data/data103919/output/yolov3_darknet53/best_model --save_dir=./inference_model
```

进行单个图片的预测


```python
import paddlex as pdx
predictor = pdx.deploy.Predictor('./inference_model')
result = predictor.predict(image='/home/aistudio/Test/6.jpg')
```


```python
%matplotlib inline
import matplotlib.pyplot as plt # plt 用于显示图片
import numpy as np
import cv2

# 读取原始图片
origin_pic = cv2.imread('/home/aistudio/Test/6.jpg')
origin_pic = cv2.cvtColor(origin_pic, cv2.COLOR_BGR2RGB)
plt.imshow(origin_pic)
plt.axis('off') # 不显示坐标轴
plt.show()
```


```python
result
```

# 六、部署模型准备

## HubServing轻量级服务化部署
部署模型的格式均为目录下包含__model__，__params__和model.yml三个文件，也就是inference_model目录下的文件格式。


```python
!pip install paddlehub -U
```

## 模型转换
将PaddleX的Inference Model转换成PaddleHub的预训练模型，使用命令hub convert即可一键转换，对此命令的说明如下：



| 参数  | 用途|  
| -------- | -------- | 
| --model_dir/-m    | PaddleX Inference Model所在的目录 |  
|--module_name/-n	|生成预训练模型的名称|
|--module_version/-v	|生成预训练模型的版本，默认为1.0.0|
|--output_dir/-o	|生成预训练模型的存放位置，默认为{module_name}_{timestamp}  |


```python
!hub convert --model_dir data/data103919/inference_model \
              --module_name hatdet \
              --module_version 1.0 
```

## 模型安装
将模型转换得到的.tar.gz格式的预训练模型压缩包，在进行部署之前需要先安装到本机，使用命令hub install一键安装


```python
!hub install hatdet_1628773541.146559/hatdet.tar.gz
```

## 模型部署
打开终端1，输入hub serving start -m hatdet完成安全帽检测模型的一键部署

![](https://ai-studio-static-online.cdn.bcebos.com/6923b4e5bda246dd927dec53f1df9b627f30016bc19a4f27a38f0d38f5609a77)



这样算是部署好了 下面通过POST请求实现预测


```python
import requests
import json
import cv2
import base64
import numpy as np
import colorsys
import warnings
import matplotlib.pyplot as plt # plt 用于显示图片

import paddlex as pdx
warnings.filterwarnings("ignore")
plt.figure(figsize=(12,8))

def cv2_to_base64(image):
    data = cv2.imencode('.png', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


if __name__ == '__main__':
    # 获取图片的base64编码格式
    img1 = cv2_to_base64(cv2.imread("Test/6.jpg"))
 
    data = {'images': [img1]}
    # 指定content-type
    headers = {"Content-type": "application/json"}
    # 发送HTTP请求
    url = "http://127.0.0.1:8866/predict/hatdet"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果，注意，r.json()["results"]本身就是一个数组，要取到对应图片的预测结果，需指定元素位置，如r.json()["results"][0]
    print(r.json()["results"])
      
    # 使用重写的visualize()方法完成预测结果后处理
    # 显示第一张图片的预测效果
    image = pdx.det.visualize(cv2.imread('Test/6.jpg'),r.json()["results"][0], save_dir=None)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off') # 不显示坐标轴
    plt.show()
```

好了 至此 整个项目运行完毕 最后的预测 结果 貌似有点多 比想象中复杂 但还是有结果 一个AI项目的大致流程 就是如此了 最后的部署只是个最简单的部署 一个好的项目要实际应用 肯定要部署到现实的开发板或者其他的服务器上去 

以上就是整个的大致流程 谢谢观看

# 七、总结

作为一个对AI感兴趣的小白 我还有好多的东西不懂 这也是我首次接触AI Studio这个平台 真的啥也不会 我这个项目落地实现的话 目前看来是不行的 需要更多的数据 这里只用很小很小的一部分数据 同时本人目前的实力也有限 等日后 羽翼丰满了 再回过头来 重新做一遍。这里真的很感谢达人创造营的各位老师的讲解 做这个项目最开始时 真的一头雾水 不知道怎么搞 反复观看老师的视频以及阅读老师的文档 想法一步步清晰 在做的过程中 借鉴很多大佬的代码 真的很感谢 同时也谢谢交流群里的各位小伙伴 在我产生疑惑时 及时给我解答 最后能做完这个项目 真的很不容易 虽然项目一般的很 但我能完成它 真的很棒 为自己点赞 我还有很多的东西不懂 还有很多的东西需要学习 最后小白加油 小白要努力变强

# 作者介绍

湖北经济学院 2019级 计算机科学与技术专业 本科生 余卓凡（纯种小白一枚）

我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/879971

最后给我点个赞吧
