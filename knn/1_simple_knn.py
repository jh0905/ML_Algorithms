# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 2/20/19 11:28 AM
 @desc:以一个简单的数据集，来实现KNN的算法，比较暴力的方法，计算欧氏距离，选则最邻近的k个样本的出现频率最高的标签
"""
import numpy as np
import operator

"""
 k近邻算法思想：
    1.计算出当前未知类型样本点与训练集中所有点的距离（欧氏距离）;
    2.按照距离大小，将训练集中的点递增排序;
    3.选取前k个样本点，计算出前k个样本每个类别出现的频率;
    4.频率最高的类别即为当前未知样本的预测分类;
"""


# 构建一个简单的训练集,X为训练样本,y为训练样本的lable
def create_dataset():
    # 创建四组二维特征
    X = np.array([[1, 101],
                  [5, 89],
                  [108, 5],
                  [115, 8]])
    # 四组特征分别对应的lable
    y = ['爱情片', '爱情片', '动作片', '动作片']
    return X, y


"""
【knn函数注释】
 参数说明：
    test为测试样本(一个向量);
    train为训练集(训练样本的特征组成的矩阵);
    lables为标签(训练集中每一个样本对应的类型组成的向量);
    k为参数,表示选取距离最近的k个样本
    
 函数说明：train为[[1,101],[5,89],[108,5],[115,8]],test为1个样本[10,45],
    这里的操作是将test平铺成[[10,45],[10,45],[10,45],[10,45]],
    这样一来,train和test相减得到的矩阵，就可以用于我们计算测试样本到每一个训练样本的距离
"""


def knn(test, train, lables, k):
    num_train_samples = train.shape[0]
    test = np.tile(test, (num_train_samples, 1))  # np.tile 将test复制num_train_samples行,复制1列
    diff_set = train - test
    distance = np.sqrt((diff_set ** 2).sum(axis=1))  # sum(axis=1),表示每一行的所有列求和
    sorted_distance_indicies = np.argsort(distance)  # 返回数组,如[1, 0, 2, 3],表示下标为1的元素值最小,其次是0,以此类推
    class_count = {}  # 该字典用来存前k个元素的类别，以及出现的次数
    for i in range(k):
        lable = lables[sorted_distance_indicies[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值0
        class_count[lable] = class_count.get(lable, 0) + 1
    # 字典按照value值,降序排列
    # key=operator.itemgetter(1) 根据字典的值进行排序
    # key=operator.itemgetter(0) 根据字典的键进行排序
    # reverse 降序排序字典
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1),
                                reverse=True)  # 如[('爱情片', 2), ('动作片', 1)]
    return sorted_class_count[0][0]


if __name__ == '__main__':
    train, lables = create_dataset()
    test = [50, 10]
    k = 3
    print("预测该样本标签为：", knn(test, train, lables, k))
    # test
