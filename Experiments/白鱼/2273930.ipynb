{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "# AI入门小白之简单障碍物检测\n",
    "\n",
    "\n",
    "这个项目应该算是一个简单的入门项目吧 做一个简单的障碍物识别 体会一下整个流程 应该是怎么样的\n",
    "\n",
    "# 一、项目背景\n",
    "\n",
    "身为一个想学AI的小白　从此刻开始入门　希望不久之后不会入土　做这个项目主要想入个门　把一个完整的AI项目的流程顺下来\n",
    "\n",
    "这个项目主要是障碍物的识别 在生活中障碍物的识别 可以运用到很多的地方 比如自动驾驶 但这个项目不足以达到应用的地步 \n",
    "\n",
    "这个项目仅供小白了解AI项目的流程是怎么样的\n",
    "\n",
    "# 二、数据集简介\n",
    "\n",
    "　数据集是我在AI　Studio的数据集大厅找的一个障碍物数据集 数据集里面有很多的不同的障碍物文件 这个数据集里面的图片都已经完成了批注 使用起来很方便 在这个项目 我对这个数据集进行了一些删减　取其中的一部分　进行模型训练及模型预测 如果你使用的是自制数据集 一定要对图片文件进行一个批注 即把你要识别的东西 框出来\n",
    "\n",
    " 那么接下来我们来开始这个简单的小项目吧"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "# 三、简单障碍物检测的数据集"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "## 3.1 查看并解压数据集"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "%cd /home/aistudio/data/data103919/\r\n",
    "!unzip -oq /home/aistudio/data/data103919/简单障碍物.zip -d /home/aistudio/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    " 接下来 我们导入PaddleX模型库 此项目是基于PaddleX完成的 同时后续将数据集处理成VOC格式 用PaddleX里面的工具很好用 一下子就可以处理所有 得到我们想要的VOC格式文件 不用一个一个处理 很方便"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "!pip install paddlex "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#使用PaddleX中的工具可以快速将数据集处理成VOC格式 \r\n",
    "!paddlex --split_dataset --format VOC --dataset_dir  /home/aistudio/stop/ --val_value 0.15  --test_value 0.05"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "## 3.2数据的预处理"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#数据预处理 这里使用了图像混合、随机像素变换、随机膨胀、随即裁剪、随机水平翻转等数据增强方法。\r\n",
    "from paddlex.det import transforms\r\n",
    "\r\n",
    "train_transforms = transforms.Compose([\r\n",
    "    transforms.RandomHorizontalFlip(),\r\n",
    "    transforms.Normalize(),\r\n",
    "    transforms.ResizeByShort(short_size=800, max_size=1333),\r\n",
    "    transforms.Padding(coarsest_stride=32)\r\n",
    "])\r\n",
    " \r\n",
    "eval_transforms = transforms.Compose([\r\n",
    "    transforms.Normalize(),\r\n",
    "    transforms.ResizeByShort(short_size=800, max_size=1333),\r\n",
    "    transforms.Padding(coarsest_stride=32),\r\n",
    "])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "## 3.3 定义测试集和验证集"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#定义测试集 和验证集 将前边划分好的是数据集用变量来定义，方便后边训练使用\r\n",
    "\r\n",
    "base = '/home/aistudio/work2/mot02'\r\n",
    "from paddlex.det import transforms\r\n",
    "from random import shuffle\r\n",
    "import paddlex as pdx\r\n",
    "import os \r\n",
    "\r\n",
    "base = '/home/aistudio/stop'\r\n",
    "\r\n",
    "train_dataset = pdx.datasets.VOCDetection(\r\n",
    "    data_dir=base,\r\n",
    "    file_list=os.path.join(base, 'test_list.txt'),\r\n",
    "    label_list='/home/aistudio/stop/labels.txt',\r\n",
    "    transforms=train_transforms,\r\n",
    "    shuffle=True)\r\n",
    "eval_dataset = pdx.datasets.VOCDetection(\r\n",
    "    data_dir=base,\r\n",
    "    file_list=os.path.join(base, 'val_list.txt'),\r\n",
    "    label_list='/home/aistudio/stop/labels.txt',\r\n",
    "    transforms=eval_transforms)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "# 四、模型训练"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "在这里的模型训练 我选择了骨干网络为DarkNet53的YOLO-V3模型"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#训练模型\r\n",
    "\r\n",
    "import matplotlib\r\n",
    "matplotlib.use('Agg') \r\n",
    "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\r\n",
    "%matplotlib inline\r\n",
    "import warnings\r\n",
    "warnings.filterwarnings(\"ignore\")\r\n",
    "\r\n",
    "#num_classes有些模型需要加1 比如faster_rcnn\r\n",
    "num_classes = len(train_dataset.labels)\r\n",
    "\r\n",
    "model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')\r\n",
    "\r\n",
    "model.get_model_info()\r\n",
    "\r\n",
    "model.train(\r\n",
    "    num_epochs=70,#训练轮数\r\n",
    "    train_dataset=train_dataset,\r\n",
    "    train_batch_size=1,#一次加载 多少图片\r\n",
    "    eval_dataset=eval_dataset,\r\n",
    "    learning_rate=3e-5,\r\n",
    "    warmup_steps=90,\r\n",
    "    warmup_start_lr=0.0,\r\n",
    "    save_interval_epochs=7,#每隔七轮保存一次\r\n",
    "    lr_decay_epochs=[42, 70],\r\n",
    "    save_dir='output/yolov3_darknet53',\r\n",
    "    use_vdl=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "# 五、模型预测"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#导出训练模型\r\n",
    "!paddlex --export_inference --model_dir=/home/aistudio/data/data103919/output/yolov3_darknet53/best_model --save_dir=./inference_model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "进行单个图片的预测"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import paddlex as pdx\r\n",
    "predictor = pdx.deploy.Predictor('./inference_model')\r\n",
    "result = predictor.predict(image='/home/aistudio/Test/6.jpg')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "%matplotlib inline\r\n",
    "import matplotlib.pyplot as plt # plt 用于显示图片\r\n",
    "import numpy as np\r\n",
    "import cv2\r\n",
    "\r\n",
    "# 读取原始图片\r\n",
    "origin_pic = cv2.imread('/home/aistudio/Test/6.jpg')\r\n",
    "origin_pic = cv2.cvtColor(origin_pic, cv2.COLOR_BGR2RGB)\r\n",
    "plt.imshow(origin_pic)\r\n",
    "plt.axis('off') # 不显示坐标轴\r\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "result"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "# 六、部署模型准备"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "## HubServing轻量级服务化部署\n",
    "部署模型的格式均为目录下包含__model__，__params__和model.yml三个文件，也就是inference_model目录下的文件格式。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "!pip install paddlehub -U"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "## 模型转换\n",
    "将PaddleX的Inference Model转换成PaddleHub的预训练模型，使用命令hub convert即可一键转换，对此命令的说明如下：\n",
    "\n",
    "\n",
    "\n",
    "| 参数  | 用途|  \n",
    "| -------- | -------- | \n",
    "| --model_dir/-m    | PaddleX Inference Model所在的目录 |  \n",
    "|--module_name/-n\t|生成预训练模型的名称|\n",
    "|--module_version/-v\t|生成预训练模型的版本，默认为1.0.0|\n",
    "|--output_dir/-o\t|生成预训练模型的存放位置，默认为{module_name}_{timestamp}  |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "!hub convert --model_dir data/data103919/inference_model \\\r\n",
    "              --module_name hatdet \\\r\n",
    "              --module_version 1.0 "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "## 模型安装\n",
    "将模型转换得到的.tar.gz格式的预训练模型压缩包，在进行部署之前需要先安装到本机，使用命令hub install一键安装"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "!hub install hatdet_1628773541.146559/hatdet.tar.gz"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "## 模型部署\n",
    "打开终端1，输入hub serving start -m hatdet完成安全帽检测模型的一键部署"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "![](https://ai-studio-static-online.cdn.bcebos.com/6923b4e5bda246dd927dec53f1df9b627f30016bc19a4f27a38f0d38f5609a77)\n",
    "\n",
    "\n",
    "\n",
    "这样算是部署好了 下面通过POST请求实现预测"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import requests\r\n",
    "import json\r\n",
    "import cv2\r\n",
    "import base64\r\n",
    "import numpy as np\r\n",
    "import colorsys\r\n",
    "import warnings\r\n",
    "import matplotlib.pyplot as plt # plt 用于显示图片\r\n",
    "\r\n",
    "import paddlex as pdx\r\n",
    "warnings.filterwarnings(\"ignore\")\r\n",
    "plt.figure(figsize=(12,8))\r\n",
    "\r\n",
    "def cv2_to_base64(image):\r\n",
    "    data = cv2.imencode('.png', image)[1]\r\n",
    "    return base64.b64encode(data.tostring()).decode('utf8')\r\n",
    "\r\n",
    "\r\n",
    "if __name__ == '__main__':\r\n",
    "    # 获取图片的base64编码格式\r\n",
    "    img1 = cv2_to_base64(cv2.imread(\"Test/6.jpg\"))\r\n",
    " \r\n",
    "    data = {'images': [img1]}\r\n",
    "    # 指定content-type\r\n",
    "    headers = {\"Content-type\": \"application/json\"}\r\n",
    "    # 发送HTTP请求\r\n",
    "    url = \"http://127.0.0.1:8866/predict/hatdet\"\r\n",
    "    r = requests.post(url=url, headers=headers, data=json.dumps(data))\r\n",
    "\r\n",
    "    # 打印预测结果，注意，r.json()[\"results\"]本身就是一个数组，要取到对应图片的预测结果，需指定元素位置，如r.json()[\"results\"][0]\r\n",
    "    print(r.json()[\"results\"])\r\n",
    "      \r\n",
    "    # 使用重写的visualize()方法完成预测结果后处理\r\n",
    "    # 显示第一张图片的预测效果\r\n",
    "    image = pdx.det.visualize(cv2.imread('Test/6.jpg'),r.json()[\"results\"][0], save_dir=None)\r\n",
    "    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\r\n",
    "    plt.imshow(image)\r\n",
    "    plt.axis('off') # 不显示坐标轴\r\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "好了 至此 整个项目运行完毕 最后的预测 结果 貌似有点多 比想象中复杂 但还是有结果 一个AI项目的大致流程 就是如此了 最后的部署只是个最简单的部署 一个好的项目要实际应用 肯定要部署到现实的开发板或者其他的服务器上去 \n",
    "\n",
    "以上就是整个的大致流程 谢谢观看"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "# 七、总结"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "作为一个对AI感兴趣的小白 我还有好多的东西不懂 这也是我首次接触AI Studio这个平台 真的啥也不会 我这个项目落地实现的话 目前看来是不行的 需要更多的数据 这里只用很小很小的一部分数据 同时本人目前的实力也有限 等日后 羽翼丰满了 再回过头来 重新做一遍。这里真的很感谢达人创造营的各位老师的讲解 做这个项目最开始时 真的一头雾水 不知道怎么搞 反复观看老师的视频以及阅读老师的文档 想法一步步清晰 在做的过程中 借鉴很多大佬的代码 真的很感谢 同时也谢谢交流群里的各位小伙伴 在我产生疑惑时 及时给我解答 最后能做完这个项目 真的很不容易 虽然项目一般的很 但我能完成它 真的很棒 为自己点赞 我还有很多的东西不懂 还有很多的东西需要学习 最后小白加油 小白要努力变强"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "# 作者介绍\n",
    "\n",
    "湖北经济学院 2019级 计算机科学与技术专业 本科生 余卓凡（纯种小白一枚）\n",
    "\n",
    "我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/879971\n",
    "\n",
    "最后给我点个赞吧"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "PaddlePaddle 2.1.2 (Python 3.5)",
   "language": "python",
   "name": "py35-paddle1.2.0"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
