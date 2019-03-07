# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 3/2/19 2:05 PM
 @desc: 了解梯度下降的算法原理，掌握传统计算方法和向量化计算方法，最终通过代码实现
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime

"""
 引子:
    逻辑回归是一个分类算法，它可以处理二分类以及多元分类任务，不是一个回归算法，但是名字中为什么会有“回归”这个误导性的词呢？
 实际上，逻辑回归虽然是分类模型，但是原理中残留着回归模型的影子。
    我们知道，线性回归 的模型是要求出，训练集中 目标向量Y和样本特征矩阵X之间的线性关系系数θ,满足Y=Xθ。此时，我们的Y是连续的，
 所以是回归模型。如果我们想要Y是离散的话，那该怎么办呢？我们可以对Y再做一次函数变换，变为g(Y),规定g(Y)的值在某个实数区间内的
 时候属于类别A，在另一个实数区间内的时候属于类别B，以此类推，我们就得到了一个分类模型。
    逻辑回归的出发点就是从这来的，接下来展开正文，介绍二元逻辑回归。
 
 1.二元逻辑回归的模型
    前面提到对线性回归的结果做一个函数g的转换，可以变化为逻辑回归，这个函数g我们在逻辑回归中一般取sigmoid函数，形式如下:
            g(z) = 1 / (1+np.exp(-z))
    它有一个非常好的性质，即当z趋于正无穷的时候，g(z)趋于1；当z趋于负无穷的时候，g(z)趋于0；
    此外，它还有一个非常好的导数性质:
            [g(z)]' = g(z)(1-g(z)) ，在后面会用到这个式子  
    我们再令z=Xθ,即得到二元逻辑回归模型的一般形式:
            h(X) = 1 / (1+np.exp(-Xθ))  ,
    X为样本输入，h(X)为模型输出，可理解为样本X属于类别1的概率大小，θ即为我们所求解的参数向量；而在训练集中，我们的目标向量是0和1组成的向量，
 所以我们规定，h(X)>0.5时，输出为1，h(X)<0.5时，输出为0.
    
 2.二元逻辑回归的损失函数
    对于目标值是连续值的线性回归模型，我们可以使用模型误差的平方和(即最小二乘法)来定义损失函数，而我们这里提到的逻辑回归是一个二分类模型，
 目标值为离散值，用最大似然估计来推导损失函数。（这里让我想起了上一节的朴素贝叶斯算法，对于离散属性，求解P(X|C)的做法，也是通过MLE计算得到）
 
   【插曲】
    肯定会想，为什么逻辑回归的损失，就要用似然估计来计算呢？这里截取一部分知乎上的回答:
        首先，机器学习的损失函数是人为设计的，用于评判模型好坏（对未知的预测能力）的一个标准、尺子，就像去评判任何一件事物一样，从不同角度
    看往往存在不同的评判标准，不同的标准往往各有优劣，并不冲突。唯一需要注意的就是最好选一个容易测量的标准，不然就难以评判了。
        其次，既然不同标准并不冲突，那使用最小二乘法作为逻辑回归的损失函数当然是可以，那这里为什么不用最小二乘而用最大似然呢？
    请看一下最小二乘作为损失函数的函数曲线(figure_non_convex.jpg)以及最大似然作为损失函数的函数曲线(figure_convex.jpg),
    显然发现最大似然作为损失函数，图像为凸函数，容易得到参数的最优解，而最小二乘作为损失函数时，图像非凸，容易陷入到局部最优。
 
    (1)我们前面提到，h(X)可理解为样本X属于类别1的概率，因此：
                P(y=1|X) = h(X) 
                P(y=0|X) = 1 - h(X)
        把这两个式子写成一个式子，即
                P(y|X,θ) = h(X)^y * (1 - h(X))^(1-y)  ,  其中y的取值为0或1
        得到了y的概率分布函数表达式，我们可以用似然函数最大化来求解我们需要的模型参数θ !!!

   【重点】    
    (2)解释一下，为什么用最大似然估计来求解参数θ时要用到累乘？
       我们假设训练样本之间是独立同分布的，我们现在想做的，就是根据已有的样本特征矩阵X，和类别向量Y，来找到一组合适的参数集θ，
    使得每一个样本与θ组合再经过函数变换之后，得到的值与其真实值尽可能一致，也就是说，这个参数向量θ，要满足下式：
                        max(P(Y|X,θ))  ,这里Y是类别向量，X是样本特征矩阵
    由贝叶斯概率公式，可知，假设事件A，B，C是独立同分布的，则P(A,B,C) = P(A)P(B)P(C),
    同理，我们假设每个样本之间是独立同分布的，因此max(P(Y|X,θ))等价于：
             max(P(y1|x1,θ) * P(y2|x2,θ) * P(y3|x3,θ) * ... * P(ym|xm,θ)) ,其中xi为第i个样本的特征向量，yi为其类别值
    
  **(3)最大似然估计取对数之后再取反，定义为我们的损失函数
       原因如下：
       这里只挑一个乘数如P(y1|x1,θ)为例进行说明即可
                -ln(P(y1|x1,θ)) = -y1*(ln h(x1)) - (1-y1)*ln(1-h(x1))     【注：取对数不是取导数，我在这犯糊涂了】
       已知h(x)的值域为(0,1),那么-log(h(x))的值域呢，就为(0,+∞)，定义算法的损失函数，我们的期望当然是预测越准,惩罚越少;预测越差,惩罚
       越多。我们再看看上式，假如预测h(x1)=0.99,y1=1时，最终惩罚为0.01；预测h(x1)=0.01,y1=1时最终惩罚为4.6，加重了惩罚,所以符合我们
       对损失函数的要求。     
    
    (4)经过上面的描述，逻辑回归的损失函数表达式为:
        J(θ) = -sum( yi*(ln h(xi)) + (1-yi)*ln(1-h(xi)) ,其中i表示第i个样本
        
 3.二元逻辑回归的损失函数的最优化问题
    对于二元逻辑回归的损失函数极小化，有比较多的方法，最常见的有梯度下降法，坐标轴下降法，牛顿法等。
    本次学习中我用的方法是梯度下降法，我在本目录下，写了一个gradient_descent_demo.py代码，以一个函数实例，
 详细介绍了用梯度下降法求极值的思路与代码实现过程。
    (1)了解完梯度下降之后，我们需要关心的就是每次迭代时θ的更新公式，关于这个，我手推了θ的更新过程，实现代码为gradient_descent()；
    (2)书上的实现方式是通过向量的方式进行计算，降低了计算的复杂度，实现代码为gradient_descent_matrix()
    (3)上面的方式，在每次更新回归系数时，都需要遍历整个数据集，计算复杂度过大，改进方法是随机梯度下降，
       一次仅用一个样本点来更新回归系数，可以在新样本到来时对分类器进行增量式更新，是一种在线学习方法，而一次性处理所有数据，被称作是“批处理”
       实现代码为stochastic_gradient_descent()

"""


def load_data_set():
    data_matrix = []
    label_vector = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line = line.strip().split('\t')
        data_matrix.append([1.0, float(line[0]), float(line[1])])  # 为了方便计算，把x_0的值默认为1.0
        label_vector.append(int(line[-1]))  # 这里和上面必须进行类型转换，因为从file里读出的元素为str类型
    return data_matrix, label_vector


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


"""
 这是用到的第一种方法，按照我推导的公式(manuscript_1.jpg)来编写的代码，算法如下:
    (1)初始化参数向量θ为一个元素值全为1,形状为(样本特征数, 1)的列向量，不一定非得是1，全为0也行；
    (2)开始迭代，迭代停止条件为达到迭代次数
        (a) 遍历参数向量θ的每一个元素，执行下面操作：
                θ[j] = θ[j] + α * sum[(Y[i] - sigmoid(θ.T * X[i]))*X[i][j]]
                (注：θ[j]表示θ的第j个元素，Y[i]表示第i个样本的类别，X[i]表示第i个样本的特征向量，X[i][j]]为第i个样本的第j个元素)
        (b) 迭代次数减1
    (3) 返回参数向量θ
"""


def gradient_descent(data_matrix, label_vector, alpha, num_of_rounds):
    theta_vector = np.ones((np.shape(data_matrix)[1], 1))  # 初始化θ 为一个长度与特征矩阵的列数相同的元素值全为1的列向量
    while num_of_rounds:
        for j in range(len(theta_vector)):
            descent = 0.0
            for i in range(len(data_matrix)):
                descent += (label_vector[i] - sigmoid(np.dot(theta_vector.transpose(), data_matrix[i]))) * \
                           data_matrix[i][j]
            theta_vector[j] = theta_vector[j] + alpha * descent
        num_of_rounds -= 1
    return theta_vector


"""
这是第二种梯度下降的方法，按照推导式(manuscript_2.jpg)直接对整个θ向量更新，算法如下：
   (1)样本特征转为(样本数，特征数)的矩阵X
   (2)样本类别转为(样本数，1)的矩阵Y
   (3)初始化一个元素值全为1，形状为(特征数，1)的向量θ
   (4)开始迭代：
       (a)计算梯度向量，公式为:
           gradient_vec = X.T * (sigmoid(Xθ)-Y)
       (b)更新向量θ，公式为:
           θ = θ - α*gradient_vec
       (c)迭代次数减1
   (5) 返回θ
"""


def gradient_descent_matrix(data_matrix, label_vector, alpha, num_of_rounds):
    data_matrix = np.mat(data_matrix)  # 转换为(100,3)的矩阵，每一行代表一个样本的特征向量
    label_vector = np.mat(label_vector).transpose()  # 转换为(100,1)的列向量
    theta_vector = np.ones((data_matrix.shape[1], 1))  # 初始化θ 为一个长度与特征矩阵的列数相同的元素值全为1的列向量
    while num_of_rounds:
        gradient_vector = data_matrix.T * (sigmoid(data_matrix * theta_vector) - label_vector)
        theta_vector = theta_vector - alpha * gradient_vector
        num_of_rounds -= 1
    return theta_vector


"""
随机梯度下降算法,按照我推导的公式(manuscript_3.jpg)来编写的代码，算法如下伪代码：
   (1)参数向量θ初始化为元素值全为1,长度为特征数的列向量
   (2)开始迭代:
       对数据集中的每一个样本
           计算梯度gradient = (sigmoid(X[i]*θ)-yi) * (X[i].T) , 数值 * 列向量的形式
           使用alpha*gradient来更新值
       迭代次数减1
   (3)迭代完成，返回参数向量
"""


def stochastic_gradient_descent(feature_mat, label_vec, alpha, num_of_rounds):
    num_of_samples = len(feature_mat)
    num_of_features = len(feature_mat[0])
    feature_mat = np.mat(feature_mat)
    label_vec = np.mat(label_vec).transpose()  # 转置成(100,1)的列向量
    theta_vector = np.ones((num_of_features, 1))
    while num_of_rounds:
        for i in range(num_of_samples):
            # error是某个样本的特征值与参数θ点乘之后，经过sigmoid函数转换后的结果与真实值yi的误差,形如[[e]]
            error = sigmoid(feature_mat[i] * theta_vector) - label_vec[i]
            gradient = error[0, 0] * feature_mat[i].T
            theta_vector = theta_vector - alpha * gradient
        num_of_rounds -= 1
    return theta_vector


# 绘制分类决策思路:
# 1.不同类别的点，用不同的颜色标注，有两种实现方式：
#   (1)直接将样本点按照类别分成两类，本次代码就是按照这种思路；
#   (2)建立color数组，根据不同类别，color填充不同颜色的元素；
# 2.绘制分类决策线
#   w_0*x_0 + w_1*x_1 + w_2*x_2 = 0是本次的分类决策线，变形一下就是 x_2 = -(w_0*1+w_1*x_1)/w_2
#   本次代码就是用x=x_1,y=x_2进行绘图，具体如下:
def plot_best_fit(weight_vec):
    mat, vec = load_data_set()  # 加载数据集
    data_arr = np.array(mat)  # 把list类型的mat,转换成ndArray类型
    n = np.shape(mat)[0]  # 数据的个数
    x_1 = []  # 类别为1的样本的横坐标
    y_1 = []  # 类别为1的样本的纵坐标
    x_0 = []  # 类别为0的样本的横坐标
    y_0 = []  # 类别为0的样本的纵坐标
    for i in range(n):  # 根据数据集标签进行分类
        if int(vec[i]) == 1:
            x_1.append(data_arr[i, 1])
            y_1.append(data_arr[i, 2])  # 1为正样本
        else:
            x_0.append(data_arr[i, 1])
            y_0.append(data_arr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(x_1, y_1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本
    ax.scatter(x_0, y_0, s=20, c='green', alpha=.5)  # 绘制负样本
    # 绘制分界线
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weight_vec[0] - weight_vec[1] * x) / weight_vec[2]
    ax.plot(x, y)
    plt.title('BestFit')  # 绘制title
    plt.xlabel('X1')
    plt.ylabel('X2')  # 绘制label
    plt.show()


if __name__ == '__main__':
    features, labels = load_data_set()
    # 这里我想对比一下，不同算法程序的运行时间
    start_time = datetime.datetime.now()
    weights = gradient_descent(features, labels, alpha=0.001, num_of_rounds=2000)
    # weights = gradient_descent_matrix(features, labels, alpha=0.001, num_of_rounds=2000)
    # weights = stochastic_gradient_descent(features, labels, alpha=0.001, num_of_rounds=2000)
    end_time = datetime.datetime.now()
    running_time = (end_time - start_time).microseconds
    print("程序运行时长为：{}s".format(running_time))
    print(weights)
    weights = np.array(weights)  # weights 转成array向量的形式
    plot_best_fit(weights)

"""
 对比一下三种实现方式的结果, 基于同一数据集，同样的学习步长，同样的迭代次数
 |--------------------------------------------------------------------------------------------------|
 |         算法                             运行时长                         参数值                    |
 |  gradient_descent                         3.6s             [7.01092219, 0.70376226, -0.98513334] |   
 |                                                                                                  |
 |  gradient_descent_matrix                 0.086s            [6.99233391, 0.70237023, -0.98258342] |                              
 |                                                                                                  |
 |  stochastic_gradient_descent              5.3s             [7.00009837, 0.70071137, -0.98870703] | 
 |--------------------------------------------------------------------------------------------------|
 
 对应的三张分类结果图分别保存为model_fit_line_1.jpg, model_fit_line_2.jpg, model_fit_line_3.jpg
 总结：
    观察发现，在本案例中，三种方式得出的结果差异不大，但是最快的方式，显然是直接通过矩阵的方式，来计算最终的回归系数，
 最慢的是随机梯度下降法，这说明：
    在样本量较少且样本维度较低时，建议使用批处理梯度下降(矩阵形式)；
    在数据量庞大或者样本维度很高时，建议采用随机梯度下降；（在使用随机梯度下降时，可以对数据集进行shuffle处理，使样本具有随机性）
"""
