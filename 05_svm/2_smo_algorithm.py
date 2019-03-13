# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 3/11/19 9:00 AM
 @desc:
"""
import numpy as np
import matplotlib.pyplot as plt

"""
 SMO算法简介:
    SMO表示序列最小最优化(sequential minimal optimization),Platt的SMO算法是将大优化问题分解成多个小优化问题来求解的，这些小问题往往
 很容易求解，并且对它们进行顺序求解的结果与将它们作为整体求解的结果是完全一致的，但是SMO算法的求解时间短很多。
    
    
"""


def load_data_set(file_name):
    data_matrix = []
    label_matrix = []
    fr = open(file_name)
    for line in fr.readlines():
        line = line.strip().split('\t')
        data_matrix.append([float(line[0]), float(line[1])])
        label_matrix.append(float(line[2]))
    return data_matrix, label_matrix


# i是第一个alpha的下标，m是所有alpha的数目，只要函数值不等于输入值i，函数就会进行随机选择
def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


# 用于调整大于h或者小于l的alpha值
def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


# 简化版的SMO函数的伪代码:
# 1.创建一个alpha向量并且将其初始化为全0向量
# 2.当迭代次数小于最大迭代次数时(外循环):
#      对数据集中的每个数据向量(内循环):
#          如果该数据向量可以被优化:
#             随机选择另外一个数据向量
#             同时优化这两个向量
#             如果这两个向量都不能被优化，退出内循环
#      如果所有向量都没被优化，增加迭代数目，继续下一次循环
# 输入参数：数据集、类别标签、常数c、容错率、最大的循环迭代次数
def smo_simple(data_mat, class_labels, c_value, tolerance, max_iter):
    data_matrix = np.mat(data_mat)
    label_matrix = np.mat(class_labels).transpose()
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m, 1)))
    n_iter = 0  # 存储的是在没有任何alpha改变的情况下遍历数据集的次数
    while n_iter < max_iter:
        alpha_pairs_changed = 0  # 用于记录alpha是否已经进行优化
        for i in range(m):
            f_xi = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + b  # 表示预测的类别
            e_i = f_xi - float(label_matrix[i])  # 计算预测值与真实值的误差
            # 如果误差很大，则对该数据实例所对应的alpha值进行优化，if下面为具体的优化过程
            if ((label_matrix[i] * e_i < -tolerance) and (alphas[i] < c_value)) or (
                    (label_matrix[i] * e_i > tolerance) and (alphas[i] > 0)):
                j = select_j_rand(i, m)  # #随机选择另一个与alpha[i]成对优化的alpha[j]
                # 步骤1：计算误差Ej
                f_xj = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b
                e_j = f_xj - float(label_matrix[j])  # 与前面的alpha[i]一样，计算这个alpha[j]的误差
                # 保存更新前的alpha值，使用深拷贝
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # 步骤2：计算上下界l_和h_,它们用于将alpha[j]调整到0和C之间
                if label_matrix[i] != label_matrix[j]:
                    l_ = max(0, alphas[j] - alphas[i])
                    h_ = min(c_value, c_value + alphas[j] - alphas[i])
                else:
                    l_ = max(0, alphas[j] + alphas[i] - c_value)
                    h_ = min(c_value, alphas[j] + alphas[i])
                if l_ == h_:
                    print('L==H')
                    continue  # 开始下一轮的for循环
                # 步骤3：计算eta
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T
                eta -= data_matrix[j, :] * data_matrix[j, :].T  # 式子写不下，太长了，于是分成两部分，与上式连在一起
                if eta >= 0:
                    print('eta >= 0')
                    continue
                # 步骤4：更新alpha[j]
                alphas[j] -= label_matrix[j] * (e_i - e_j) / eta
                # 步骤5：修剪alpha[j]
                alphas[j] = clip_alpha(alphas[j], h_, l_)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('alpha[j] barely not changed ')
                    continue
                # 步骤6：更新alpha[i]
                alphas[i] += label_matrix[j] * label_matrix[i] * (alpha_j_old - alphas[j])
                # 步骤7：更新b1和b2
                b1 = b - e_i - label_matrix[i] * (alphas[i] - alpha_i_old) * \
                     data_matrix[i, :] * data_matrix[i, :].T - \
                     label_matrix[j] * (alphas[j] - alpha_j_old) * \
                     data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - e_j - label_matrix[i] * (alphas[i] - alpha_i_old) * \
                     data_matrix[i, :] * data_matrix[j, :].T - \
                     label_matrix[j] * (alphas[j] - alpha_j_old) * \
                     data_matrix[j, :] * data_matrix[j, :].T
                # 步骤8：根据b1和b2更新b
                if 0 < alphas[i] < c_value:
                    b = b1
                elif 0 < alphas[j] < c_value:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alpha_pairs_changed += 1
                print('n_iter:%d i:%d pairs changed %d' % (n_iter, i, alpha_pairs_changed))
        # 更新迭代次数
        if alpha_pairs_changed == 0:
            n_iter += 1
        else:
            n_iter = 0
        print('n_iteration number : %d' % n_iter)
    return b, alphas


def plot_best_fit(data_mat, labels_mat, w, b):
    # 绘制样本点
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(data_mat)):
        if labels_mat[i] > 0:
            data_plus.append(data_mat[i])
        else:
            data_minus.append(data_mat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], c='red', s=50, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], c='green', s=50, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(data_mat)[0]
    x2 = min(data_mat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas_):
        if alpha > 0:
            x, y = data_mat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='blue')
    plt.show()


def get_w(data_mat, labels_mat, alphas):
    alphas, data_mat, labels_mat = np.array(alphas), np.array(data_mat), np.array(labels_mat)
    w = np.dot((np.tile(labels_mat.reshape(1, -1).T, (1, 2)) * data_mat).T, alphas)
    return w.tolist()


if __name__ == '__main__':
    data, labels = load_data_set('testSet.txt')
    b_, alphas_ = smo_simple(data, labels, 0.6, 0.001, 40)
    # print("b is {}".format(b))
    # print("alphas:{}".format(alphas[alphas > 0]))
    w_ = get_w(data, labels, alphas_)
    plot_best_fit(data, labels, w_, b_)
