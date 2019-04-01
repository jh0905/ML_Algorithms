# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 4/1/19 2:42 PM
 @desc: 根据前面的学习，手写k-means代码（比想象中麻烦）
"""

from numpy import *
import matplotlib.pyplot as plt


def load_data_set(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line = line.strip().split('\t')  # 文件里读出来的是str类型，我们计算k-means需要用数值，所以下面转成float类型
        flt_line = map(float, line)  # map函数，第一个参数为函数(可以用lambda x:func(x))，第二个参数为迭代器，这里传的是list
        # flt_line = [float(x) for x in line]   # 方法2
        data_mat.append(list(flt_line))  # python3做了改动，map()函数返回的是一个函数，要取得里面的元素，必须用list进行转换
    return array(data_mat)  # 将普通的list转成np.array，便于之后做各种numpy里的运算


def compute_euclidean_dist(vec_a, vec_b):
    square_sum = sum((vec_a - vec_b) ** 2)
    return sqrt(square_sum)


# 定义一个生成k个随机簇质心的函数
# 输入：特征矩阵m*n，簇个数k
# 输出：k行n列矩阵，表示k个向量
# 初始化一个k行n列的零矩阵
# 计算特征矩阵中每一列的最小值和最大值，随机生成k个该列(min,max)范围的数
# 赋值结束，返回k行n列矩阵
def rand_c_cluster(data_mat, k):
    n = data_mat.shape[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        min_value = min(data_mat[:, j])
        max_value = max(data_mat[:, j])
        centroids[:, j] = random.uniform(min_value, max_value, (k, 1))
    return centroids


"""
 算法思想：（伪代码）
    (1) 获取样本的个数n_samples
    (2) 初始化一个随机产生的质心矩阵centroids
    (3) 初始化布尔型变量centroids_changed=True
    (4) 初始化样本划分向量samples_assignment为零向量（index为样本序号，value为样本所属的簇k）
    (5) while 质心矩阵发生了改变为True:
        (a)对于每一个样本i，i=0,1,2,...,N:
                初始化样本距离簇中心的距离min_dist为无穷inf
                初始化样本所属的簇min_index为-1
                对于每一个簇中心j，j=1,2,3,...,K:
                    计算样本点与它的距离dist
                    如果dist<min_dist:
                        把dist设为min_dist
                        把此时的下标j设为min_index
                判断samples_assignment[i]所保存的值是不是和min_index相等：
                    不是的话，说明要把该样本所属的簇修改为min_index，并且把centroids_changed置为True
        (b)根据新划分的样本簇，重新计算质心矩阵
    (6)退出while循环，返回质心矩阵centroids和样本划分向量samples_assignment
                    
"""


def k_means(data_mat, k):
    n_samples = data_mat.shape[0]
    centroids = rand_c_cluster(data_mat, k)
    cluster_assignment = zeros(n_samples)  # 第一列存样本属于哪一个簇，第二列为样本与簇中心的距离
    centroids_changed = True
    while centroids_changed:
        centroids_changed = False  # 一次迭代中，初始化centroids_changed没有变化，后面如果发生了变化，则置为True
        for j in range(n_samples):
            min_dist = inf
            min_index = -1
            for index in range(k):
                dist = compute_euclidean_dist(data_mat[j], array(centroids[index])[0])
                if dist < min_dist:
                    min_dist = dist
                    min_index = index
            if cluster_assignment[j] != min_index:
                centroids_changed = True
                cluster_assignment[j] = min_index
        for index in range(k):
            cluster_samples = data_mat[cluster_assignment == index]
            centroids[index] = mean(cluster_samples, axis=0)
    return centroids, cluster_assignment


# 这里直接绘制的是k=4时的分类结果图，如果想绘制更多类别，在里面进行调整
def plot_cluster_result(data_mat, centroids, cluster_assignment):
    plt.figure()
    cluster = []
    for i in range(4):
        cluster.append(data_mat[cluster_assignment == i])
    plt.scatter(x=cluster[0][:, 0], y=cluster[0][:, 1], c='red', s=100, marker='v')
    plt.scatter(x=cluster[1][:, 0], y=cluster[1][:, 1], c='orange', s=100, marker='s')
    plt.scatter(x=cluster[2][:, 0], y=cluster[2][:, 1], c='green', s=100, marker='o')
    plt.scatter(x=cluster[3][:, 0], y=cluster[3][:, 1], c='blue', s=100, marker='^')
    plt.scatter(x=centroids[:, 0], y=centroids[:, 1], c='black', s=400, marker='+')
    plt.show()


if __name__ == "__main__":
    data_matrix = load_data_set('testSet.txt')
    k_centroids, assignments = k_means(data_matrix, 4)
    plot_cluster_result(data_matrix, array(k_centroids), assignments)
