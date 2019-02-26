# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 2/22/19 4:50 PM
 @desc: 手写决策树算法，判断一个具有两个特征('no surfacing','flippers')的样本，是否属于鱼类
"""
from math import log
import operator
import copy
import pickle

"""

 引子:
    机器学习算法其实很古老，作为一个码农经常会不停的敲if, else if, else,其实就已经在用到决策树的思想了。
    只是你有没有想过，有这么多条件，用哪个条件特征先做if，哪个条件特征后做if会比较优呢？
    怎么准确的定量选择这个标准就是决策树机器学习算法的关键了。
    
 本文的架构:
    1.决策树算法原理(信息熵entropy)
    2.用Python实现决策树的构造
    3.测试算法:使用决策树执行分类
    4.使用算法:决策树的存储

 1.决策树算法原理:
    (1) 在构造决策树时,我们首先考虑的就是在当前数据集中的众多特征中,哪一个特征能够划分出最好的结果呢？划分数据集的大原则是：
    将无序的数据集变得更加有序。如何量化数据集的“序”,自然而然的就想起了香农熵或者简称熵;
    (2) 熵度量了事物的不确定性,越不确定的事物,它的熵就越大;随机变量X的熵的表达式H(X)如下：
        H(X) = -sum(pi * log pi)   假设X共有n个类别,i的取值则为1~n,pi表示X属于第i类的概率, 最后计算累加和再求相反数;
        函数实现为: calc_shannon_entropy()
    (3) 在第二步的基础上,我们开始学习如何划分数据集,做法是对每个特征划分数据集,再计算划分之后的信息熵,计算方法如下：
        假如原始数据集共10条数据,特征A共有三种取值,按照特征A划分的话,得到三个子集,样本数分别是3、4、3,那么划分之后的信息熵为
        3/10 * H(A1) + 4/10 * H(A2) + 3/10 * H(A3)
        依次计算每一个特征划分数据集之后的信息熵,信息增益最大的特征作为该数据集上的划分特征
    (4) 递归构造决策树:
        递归过程：得到原始数据集,基于最好的属性划分数据集,由于特征值可能有多个,因此可能存在多个分支将数据集划分,对每一个分支数据集进行分割操作;
        递归结束条件：
            第一个停止条件是所有的类标签完全相同,则直接返回该类标签；
            第二个停止条件是使用完了所有特征,仍然不能将数据划分仅包含唯一类别的分组,即决策树构建失败,特征不够用,
        此时说明数据维度不够,由于第二个停止条件无法简单地返回唯一的类标签,这里挑选出现数量最多的类别作为返回值;
 
 2.用Python实现决策树的构造:
    采用递归的思想，实现函数见下面的 creat_tree(),有详细描述，
    
 3.测试算法 --- 使用决策树执行分类:
    仍然采用递归的思想，遍历tree(字典结构)的keys(),一直读到对应的value不再是dict结构的key为止，
    函数实现见下面的classify()
 4.使用算法 --- 决策树的存储:
    构造决策树是一个很耗时的任务，如果数据集很大，将会耗费很多计算时间，对应地，决策树执行分类任务时，时间是很快的，
    因此，我们要做的是，存储构建好的决策树，从而提高效率；
    我们需要使用Python模块pickle序列化对象，因为序列化对象可以保存在磁盘上，在需要的时候再读取出来，任何对象都可以执行序列化操作
    
"""


# 构建一个简单的数据集
def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']  # 0,1所对应的特征名
    return data_set, labels


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
        shannon_entropy += -prob * log(prob, 2)  # 默认的底为自然指数e,计算机中，我们一般用2为底的log函数
    return shannon_entropy


# 参数说明：
# data_set为原始数据集,axis为进行划分的维度（特征）,value为该维度的特征值,返回满足该特征值的所有列表的集合
# 如：原始数据集为[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# 函数参数值为split_data_set(data, 0, 0) ; 返回:[[1, 'no'], [1, 'no']]
def split_data_set(data_set, axis, value):
    result_set = []
    for entry in data_set:
        if entry[axis] == value:
            new_entry = entry[:axis]  # 像extend,append,remove等方法，都是直接在列表上进行修改,返回值为None,别用a=a.append(b)的形式
            new_entry.extend(entry[axis + 1:])
            result_set.append(new_entry)
    return result_set


"""
 choose_best_feat_to_split 函数思想：
    1.统计数据集中的特征数,逐个特征计算分裂后的信息增益;
    2.计算信息增益过程如下(以特征A为例):
        a.统计特征A的取值(去重)
        b.根据特征值将样本分为几类,A=a1的样本为1类,A=a2的样本为1类,...
        c.统计划分之后，每一类的样本数与总样本数的比值，再计算每一类的信息熵，最终加权求和
    3.信息增益 = 原始信息熵 - 分类后的信息熵，取最大信息增益的那一个特征作为分裂特征
"""


def choose_best_feat_to_split(data_set):
    num_of_feature = len(data_set[0]) - 1  # 最后一列为label,所以特征数-1
    base_entropy = calc_shannon_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_of_feature):
        feat_value_list = [entry[i] for entry in data_set]  # 相当于是取出第i列的所有元素值,组成列表
        unique_feat_value = set(feat_value_list)
        new_entropy = 0.0
        for feat_value in unique_feat_value:
            sub_data_set = split_data_set(data_set, i, feat_value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# 函数功能：统计 class_list中，出现次数最多的类别
def majority_vote(class_list):
    class_count = {}
    for element in class_list:
        class_count[element] = class_count.get(element, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# create_tree函数说明：
# 输入示例：data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']] ,
#         labels = ['no surfacing', 'flippers']
# 输出示例：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# 算法步骤:
# 1.提取传入的数据集中的所有类别元素,如果所有元素类别相同，直接返回该类别值作为这个分支的结果
# 2.如果传进来的数据集中，每一行只剩一个元素，即所有特征都划分完了，只剩下类标签，那么用多数投票原则，返回该最多的类别值，作为这个分支的结果
# 3.在两步没有返回的话，下面是递归的主要过程：
#   a.选择该数据集的最佳分割特征，找到该特征对应的特征名(主要是便于最终生成的树，看着易懂)
#   b.创建树，字典结果，键名为最佳分割特征，键值为{} (递归到最后是类别名)
#   c.删除已分割的特征名，找到最佳分割特征所对应的特征值，将数据集分为几个部分，每个部分都不再包含此特征字段
#   d.递归地对分割的每一个数据集进行create_tree操作
# 4.返回my_tree, my_tree是一个字典结构
def create_tree(data_set, labels):
    class_list = [entry[-1] for entry in data_set]
    # 如果class_list中，第一个元素的个数等于总长度，说明该数据集中，所有元素类别相同，则停止进行划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # len(data_set[0]) == 1 说明数据集只剩类标签，如[['yes'], ['no'], ['yes']]，没办法再分割数据集了
    if len(data_set[0]) == 1:
        return majority_vote(class_list)
    best_feat = choose_best_feat_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del labels[best_feat]  # 删除特征名中的该元素,已经使用过了,之后不能再使用
    feat_values = [entry[best_feat] for entry in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_features = labels
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_features)
    return my_tree


# classify函数说明：
# 输入示例: input_tree : {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
#          feat_labels : ['no surfacing','flippers']
#          test_vec : [1,1]
# 输出示例: yes
# 算法思想:我们在创建决策树的时候,知道最终的类标签一定是在叶子节点上，用字典结果来说，就是最底层的value值，而就算是最简单的决策树，如
# {'happy':{0:'no',1:'yes'},也要获取嵌套在里面的字典，
# 【所以classify实际上也是一个递归函数，一直读到不再是dict结构的key为止！！！】
# 算法步骤:
# 1.获取最外层的key，然后提取所对应的字典结构
# 2.找到上一步的key（特征名）所对应的索引值，便于我们在test_vec中找到对应的特征值
# 3.遍历第一步获取到的字典结构的keys,如果key对应的value仍是字典结构，则递归调用classify函数；
#   否则直接返回key所对应的value值作为test_vec的类标签
def classify(input_tree, feat_labels, test_vec):
    first_key = list(input_tree.keys())[0]  # 字典的key不支持索引，所以转成list形式
    second_dict = input_tree[first_key]
    feat_index = feat_labels.index(first_key)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


# 存储决策树到硬盘
def store_tree(input_tree, file_name):
    # 这里的mode,要写作'wb',Python2中写'w'就行
    fw = open(file_name, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()


# 从硬盘中加载决策树
def load_tree(file_name):
    fr = open(file_name, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    data, feat_labels = create_data_set()
    # 小知识：python 中，"a = b"表示的是对象 a 引用对象 b，对象 a 本身没有单独分配内存空间(重要：不是复制！)
    # 这里对feat_labels进行拷贝操作，因为在create_tree的算法中，对feat_labels进行删除操作
    backup = copy.copy(feat_labels)
    my_tree = create_tree(data, feat_labels)
    print(my_tree)
    store_tree(my_tree, 'my_decision_tree.txt')
    result = classify(my_tree, backup, [1, 1])
    if result == 'yes':
        print("该物种是鱼类")
    else:
        print('该物种不是鱼类')
