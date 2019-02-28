# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 2/28/19 10:49 AM
 @desc:
"""
from numpy import *

"""
 1.引子:
    贝叶斯学派很古老，但是从诞生到一百年前一直都不是主流，主流学派是频率学派，贝叶斯学派凭借着现代特定领域的出色应用而崛起；
    贝叶斯学派主张，我们在实际问题中需要得到的后验概率（如某邮件属于垃圾邮件的概率），可以通过先验概率（存在哪些文本说明是垃圾邮件的历史经验）
    和数据（当前邮件中存在着哪些文本）一起综合得到。频率学派攻击的是先验概率，一般来说，先验概率就是我们对于数据所在的领域的历史经验，但是这个
    经验常常难以量化，于是贝叶斯学派大胆假设先验分布的模型，如正态分布、beta分布等，这个假设没有特定的依据，因此被频率学派认为很荒谬，但是在
    实际应用中，如垃圾邮件分类、文本分类，有着出色的表现。

 2.朴素贝叶斯的算法原理
    (1) 条件独立公式，如果X和Y相互独立，那么X和Y同时发生的概率为：
                    P(X,Y) = P(X) * P(Y)
    (2) 条件概率公式，在X发生的前提下，Y发生的概率为：
                    P(Y|X) = P(X,Y) / P(X)
                    P(X|Y) = P(X,Y) / P(Y)
        推导出后验概率的计算：
                    P(Y|X) = P(X|Y)P(Y) / P(X)
    (3) 全概率公式：
                    P(X) = P(X|Y=Y1)+ P(X|Y=Y2)+P(X|Y=Y3)+...+P(X|Y=Yn),假设Y共有Y1,Y2,...,Yn种可能取值
 3.贝叶斯的决策理论
    假设样本共有n种类别,判断样本X属于哪一类时，分别计算出P(Y=Y1|X),P(Y=Y2|X),...,P(Y=Yn|X)的值，概率值最大的Y，即为样本X的分类.
    
 4.使用朴素贝叶斯进行文档分类
    (1)加载数据集，收集已知分类的文章，保存下来
    (2)创建词汇表，统计在所有文章中出现过的单词，整理成词汇表
    (3)词向量表示，通过词袋法，将每一篇文章转成向量表示
    (4)训练算法
        (a) 用向量W来表示每篇文章转换之后的词向量,Ci表示这篇文章所属的类别,那么:
                    p(Ci|W) = p(W|Ci)*p(Ci)/p(W)
            而W可以展开成一个个独立的特征，w0,w1,w2,...,wn，假设每一个词相互独立,那么可以得到:
                    p(W|Ci) = p(w0|Ci)p(w1|Ci)p(w2|Ci)...p(wn|Ci)
       *(b) 这里有个细节需要提一下，我们最终要比较的是p(Ci|W)的值，不管W属于哪个类别，p(W)的值始终不改变，因此我们这里可以直接舍弃它，得到
                    p(Ci|W) = p(w0|Ci)p(w1|Ci)p(w2|Ci)...p(wn|Ci) * p(Ci)   
        (c) 按照上面的思路，我们需要计算出p(w0|Ci),p(w1|Ci),...,p(wn|Ci)和p(Ci)的值
            实现代码见下方的train_naive_bayes函数，里面有详细的注释
    (5)算法改进
        问题一：
            利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，我们在(4)中用累乘的方式进行简化计算，可能存在
            某些单词出现的概率为0，以至于最终结果为0，为此，我们要解决这个缺陷。
        解决办法：
            我们将每一个类别中，每个单词出现的次数初始化为1，相对应地，每个类别的总单词数，初始化为len(vocab_list)，这种解决0概率问题的做法
            称为加1平滑，也称拉普拉斯平滑。
        问题二：
            由于太多太小的数相乘，可能会导致程序出现下溢出的问题，或者得不到正确的答案
        解决办法：
            将乘积取自然对数,由于ln(a*b) = ln(a)+ln(b)的性质，原本是p(w0|c1)*p(w1|c1)*p(w2|c1)...p(wn|c1)*p(c1)取自然对数得到
            ln(p(w0|c1)*p(w1|c1)*p(w2|c1)...p(wn|c1)) = ln(p(w0|c1))+ln(p(w0|c1))+...+ln(p(wn|c1))+ln(c1)
            采用自然对数处理不会有任何损失，f(x)和ln(f(x))的函数图像为function_curve.png，观察发现，二者具有一致的单调性
    (6)算法预测
        经过前面的铺垫，我们终于学习到了，在某类别中出现某个单词的概率，比如说p(w0|c1)，p(w1|c1)，那么如何预测某文档属于哪一类呢？
        (a)将文档转成词向量表示，例如[0,0,1,...,0,1,0]
        (b)词向量分别与每一个类别所对应的词概率向量(如p_1_vec)相乘,累加求和,再加上该类别文档的先验概率如p(C=c1),最终结果即为文档属于
           类别c1的概率
        【个人感受】前面学习到的p_1_vec向量，可以理解为每个单词在该类别下所占的权重!!!词向量与之相乘，再累加起来，值越大，就越属于该主题
        (c)比较上一步中文档属于每一个类别的概率，将文档划分给概率值最大的那个类别
        实现代码见classifyNB，挺简洁的
    
"""


# 数据集由两部分组成，一个是post列表，表示每一条告示的内容；另一个是类标签向量，对应每一条post是不是用了侮辱性词汇
def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return posting_list, class_vec


# 函数功能：构建词汇表，提取所有句子中，不重复的单词
def create_vocab_list(data_set):
    vocab_set = set()
    for post in data_set:
        # 集合运算，‘|’符号表示两集合取并集，‘&’符号表示两集合取交集，‘^’符号表示仅在一个集合中出现的元素的集合
        vocab_set = vocab_set | set(post)
    return list(vocab_set)


# 函数功能：词向量表示，根据前面创建的词汇表，传进来的句子中，包含此单词，标记为1，不包含此单词，标记为0
def words2vec(vocab_list, post):
    post_vec = [0] * len(vocab_list)
    for word in post:
        if word in vocab_list:
            post_vec[vocab_list.index(word)] = 1  # 这里取值只有1，用的是set_of_words词集模型，之后会用词袋法，统计单词出现的次数
        else:
            print('the word {} is not in my vocabulary list'.format(word))
    return post_vec


# 函数功能：在words2vec的基础上，把所有文档转换成向量形式，返回一个向量组成的矩阵
def docs2matrix(vocab_list, data_set):
    post_matrix = []
    for data in data_set:
        post_matrix.append(words2vec(vocab_list, data))
    return post_matrix


# 本案例是一个二分类问题，所以本算法适用于二分类任务，多分类任务可在此基础上改进
# 输入是 词向量矩阵，类别向量
# 输出是 在类别0中出现某个单词的概率组成的向量，即[p(w0|C=c0),p(w01|C=c0),p(w2|C=c0)...,p(wn|C=c0)]，
#       在类别1中出现某个单词的概率组成的向量，即[p(w0|C=c1),p(w01|C=c1),p(w2|C=c1)...,p(wn|C=c1)]
#       训练集中，某文档属于类别1的概率,即p(C=c1),因为是二分类任务，所以p(C=c0) = 1-p(C=c1)
def train_naive_bayes(train_matrix, train_labels):
    num_of_docs = len(train_matrix)
    num_of_words = len(train_matrix[0])
    p1 = sum(train_labels) / num_of_docs  # 表示类别1的文档占总文档的比例
    # sum_0_vec = zeros(num_of_words)  # num_0_vec，用于统计类别为0的文档向量中，每一个单词出现的总次数
    # sum_1_vec = zeros(num_of_words)
    # 经改进后，采用加1平滑法
    sum_0_vec = ones(num_of_words)
    sum_1_vec = ones(num_of_words)
    # sum_0_words = 0  # 统计类别为0的文档中，单词的总个数
    # sum_1_words = 0
    # 采用加1平滑法后，每个类别单词总量的初始值也进行改正
    sum_0_words = num_of_words
    sum_1_words = num_of_words
    for i in range(num_of_docs):
        if train_labels[i] == 1:
            sum_1_vec += train_matrix[i]  # sum_1_vec表示所有类别为1的文档，词向量的累加和,是一个向量
            sum_1_words += sum(train_matrix[i])  # sum_1_words表示所有类别为1的文档中，单词的总数
        else:
            sum_0_vec += train_matrix[i]
            sum_0_words += sum(train_matrix[i])
    p1_vec = sum_1_vec / sum_1_words  # p1_vec，理解为p(w0|c1),p(w1|c1),p(w2|c1),p(w3|c1)...组成的向量
    p0_vec = sum_0_vec / sum_0_words  # p0_vec，理解为p(w0|c0),p(w1|c0),p(w2|c0),p(w3|c0)...组成的向量
    # 为了防止下溢出，返回值进行自然对数转换
    return log(p0_vec), log(p1_vec), p1


# 朴素贝叶斯分类函数
def classifyNB(test_vec, p0_vec, p1_vec, p1):
    p1 = sum(test_vec * p1_vec) + log(p1)
    p0 = sum(test_vec * p0_vec) + log(1 - p1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    posts, labels = load_data_set()
    my_vocab_list = create_vocab_list(posts)
    train_mat = docs2matrix(my_vocab_list, posts)
    p_0_vec, p_1_vec, p_class_1 = train_naive_bayes(train_mat, labels)
    test_post = ['love', 'my', 'dalmation']
    test_post2vec = words2vec(my_vocab_list, test_post)
    result = classifyNB(test_post2vec, p_0_vec, p_1_vec, p_class_1)
    print('test_post\'s label is', result)
