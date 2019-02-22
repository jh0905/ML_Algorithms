# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 2/22/19 4:50 PM
 @desc:
"""
from math import log

"""
 引子:
    机器学习算法其实很古老，作为一个码农经常会不停的敲if, else if, else,其实就已经在用到决策树的思想了。
    只是你有没有想过，有这么多条件，用哪个条件特征先做if，哪个条件特征后做if会比较优呢？
    怎么准确的定量选择这个标准就是决策树机器学习算法的关键了。
    
 本文的架构:
    1.决策树算法原理(信息熵entropy)
    2.用Python实现决策树的构造
    3.使用Matplotlib绘制树形图

 1.决策树算法原理:
    (1) 在构造决策树时,我们首先考虑的就是在当前数据集中的众多特征中,哪一个特征能够划分出最好的结果呢？划分数据集的大原则是：
    将无序的数据集变得更加有序。如何量化数据集的“序”,自然而然的就想起了香农熵或者简称熵;
    (2) 熵度量了事物的不确定性,越不确定的事物,它的熵就越大;随机变量X的熵的表达式H(X)如下：
        H(X) = -sum(pi * log pi)   假设X共有n个类别,i的取值则为1~n,pi表示X属于第i类的概率, 最后计算累加和再求相反数;
        函数实现为: calc_shannon_entropy()
    
"""


# 计算给定数据集的香农熵
def calc_shannon_entropy(data_set):
    num_entries = len(data_set)
    label_count = {}
    for entry in data_set:
        current_label = entry[-1]  # 每一条记录的最后一个元素,为它的label
        label_count[current_label] = label_count.get(current_label, 0) + 1
    shannon_entropy = 0.0
    for label in label_count:
        prob = label_count[label] / num_entries
        shannon_entropy += -prob * log(prob)
    return shannon_entropy


# 构建一个简单的数据集
def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels
