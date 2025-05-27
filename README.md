# 微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类

## 仓库介绍

本项目为课程神经网络和深度学习作业的代码仓库

* 基本要求：
  - (1) 训练集测试集按照 [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 标准；
  - (2) 修改现有的 CNN 架构（如AlexNet，ResNet-18）用于 Caltech-101 识别，通过将其输出层大小设置为 101 以适应数据集中的类别数量，其余层使用在ImageNet上预训练得到的网络参数进行初始化；
  - (3) 在 Caltech-101 数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调；
  - (4) 观察不同的超参数，如训练步数、学习率，及其不同组合带来的影响，并尽可能提升模型性能；
  - (5) 与仅使用 Caltech-101 数据集从随机初始化的网络参数开始训练得到的结果 进行对比，观察预训练带来的提升。

## 文件说明
```bash
- ImageNet_Caltech-101
  - data_process.py  # 数据集处理代码，划分训练集和测试集
  - pr_train.py # 使用预训练模型进行训练
  - re_train.py # 使用随机初始化模型进行训练
  - test.py # 模型测试主程序
  - para_search.py # 模型参数探索主程序
  - visualize.py # 可视化数据增强处理后的图片
  - visualise_param.py # 可视化不同参数下准确率的箱线图
```

## 一、 模型的训练与测试

### 数据下载

从[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02)网页下载Caltech-101数据集caltech-101，存在在仓库根目录同级即可

### 模型训练

* 进入仓库根目录，在命令行中运行：
```bash
python train.py 
```

生成的模型权重会以`pth`的形式自动保存；训练中产生的loss和Accuracy信息会记录在tensorboard中，保存在runs文件夹下
### 模型测试

* 模型权重地址：[https://pan.baidu.com/s/1EE0prDMZ16BCECfs4cGTFA?pwd=2vrf](https://pan.baidu.com/s/1EE0prDMZ16BCECfs4cGTFA?pwd=2vrf)
* 将模型权重文件放至根目录下；
* 运行：
```bash
python test.py
```
推荐使用的模型为`best_pr_model_15_0.01_0.001_3_0.5.pth`，在测试集上准确率可达 94.12%。

## 二、模型参数搜索与可视化

### 1. 模型参数搜索
* 在命令行中运行：
```bash
python para_search.py
```
最后会返回给定候选参数中，最优的参数组合以及在验证集上的最高准确率，并记录在hyperparameter_search_results.csv文件中，方便后续的可视化。
* 在命令行中运行：
```bash
python visualize_param.py
```
会绘制不同超参数下模型准确率的箱线图。

### 2. 训练信息以及模型参数可视化
* 在命令行中运行：
```bash
tensorboard --logdir=runs/  --host=127.0.0.1 --port=8008
```
会在网页端显可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的accuracy变化
