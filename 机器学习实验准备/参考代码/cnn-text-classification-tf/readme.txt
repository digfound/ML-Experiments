文件结构：
所有文件均在cnn-text-classification-tf下
该文件夹下data文件夹包含正负性数据
data_helpers.py清洗数据
eval.py处理其他事项
text_cnn.py文本分类
train.py训练程序

开始程序只需要使用命令下面命令即可：
python train.py


参数含义：
ALLOW_SOFT_PLACEMENT=TRUE:是否需要自动分配
BATCH_SIZE=64:批处理大小
CHECKPOINT_EVERY=100:保存一次模型
DEV_SAMPLE_PERCENTAGE=0.1:测试数据集比例
DROPOUT_KEP_PROB=0.5：正则化CNN，获取神经元激活概率
EMBEDDING_DIM=128：每个单词的词向量的长度
EVALUATE_EVERY=100：每训练一百次，测试一次
FILTER_SIZES=3,4,5：卷积核覆盖的单词
L2_REG_LAMBDA=0.0：正则化参数
LOG_DEVICE_PLACEMENT=False:是否打印日志
NUM_EPOCHS=200：总训练次数
NUM_FILTERS=128：卷积核的数量



实验环境：
1.ubuntu16.04
2.python2.7（默认安装版本）
3.tensorflow1.0版本以上
安装命令：sudo pip install  tensorflow
4.运行文件安装python库
sudo pip install  numpy
sudo pip install  collections





其他：
pip可以用apt-get替换
pip安装设置：
cd 到主目录下
wget https://bootstrap.pypa.io/get-pip.py
sudo python  get-pip.py
mkdir ~/.pip
然后在该目录下创建pip.conf文件编写如下内容：
命令：vim ~/.pip/pip.conf

[global]
trusted-host =  pypi.douban.com
index-url = http://pypi.douban.com/simple