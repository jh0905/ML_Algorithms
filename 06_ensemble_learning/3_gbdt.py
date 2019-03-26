# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 3/25/19 2:05 PM
 @desc: GBDT是一种基于残差的前向学习算法，残差用负梯度来表示，另外一个重点是每轮基于残差值找到每个叶子节点域的最佳拟合值，组成该轮的决策树
"""

"""
 写在前面的话：如果一开始对于boosting tree没有一定的理解的话，强烈建议阅读《统计学习方法》P149 例8.2，这个实例非常直白地讲解了提升树的原理!
 
 引子：
    GBDT也是集成学习Boosting家族的成员，也叫MART(Multiple Additive Regression Tree 多个加法回归树)，与传统的AdaBoost有很大的不同
    
        AdaBoost，我们是利用前一轮迭代弱学习器的误差率来更新训练集的权重，这样一轮轮的迭代下去；
        GBDT也是迭代，使用了前向分布算法，但是弱学习器限定了只能使用CART回归树模型，迭代思路和AdaBoost也有所不同；
    
    划重点：GBDT中的树都是回归树，不是分类树，尽管GBDT调整后也可用于分类但是用到的还是回归树
    
    GBDT的思想可以用一个通俗的例子解释，假如有个人30岁，我们先用20岁去拟合，发现损失有10岁，这时我们用12岁去拟合剩下的损失，发现差距为-2岁，
 接下来我们再用-1岁拟合剩下的差距，差距就只有-1岁了。迭代停止条件为所有学习器都已经用完了或者训练误差达到了设定的阈值.
 
 
 1.GBDT的基学习器是CART回归树，这个我之前在讲DecisionTree的时候提到过，这里稍微回顾一下：
 
    (1) CART回归树的度量目标是，对于任意划分特征A,对应的任意划分点s,将数据集划分为D1,D2,求出D1和D2的均方差，找到使得D1与D2均方差之和最小时
        对应的特征及特征值划分点.
            min { min sum(yi-c1)^2 + min sum(yi-c2)^2) }   ，其中第一个yi ∈ D1，后面的yi ∈ D2，yi是特征A对应的特征值
            
    (2) 一直迭代上一步的操作，直到达到递归停止条件
        (a)当前节点的数据集为D,如果样本个数低于max_of_samples或者样本中不包含任何特征值，则返回决策子树，递归停止
        (b)所有叶子节点上的平方损失误差和小于指定阈值，则返回决策子树，递归停止
    
    (3) CART回归树最终的输出不是类别，它采用的是用最终叶子的均值(或中位数)作为当前节点的预测值.
 
 
 2.再来了解一下提升树算法
    (1)提升树模型可以理解为决策树的加法模型：
            f_m(x) = T_1(x,Θ1) + T_2(x,Θ2) + T_3(x,Θ3) + ... + T_m(x,Θm) ，T_i(x,Θi)对应着一个决策树模型，拥有独立的决策树参数
    
    (2)在前向学习的第m步，给定当前模型f_m-1(x)，需求解最优的Θm*
            Θm* = arg min sum(L(yi,f_m-1(xi)+T(xi,Θi)))   ， Θm*即为第m棵决策树的参数
            
    (3)当用平方误差作为损失函数时
       L(y,f(x)) = (y-f(x))^2
            
       L(yi,f_m-1(x)+T(x,Θ)) = (y-f_m-1(x)-T(x,Θ))^2 = (r - T(x,Θ))^2   其中，r = y - f_m-1(x) ，表示当前模型拟合的残差!
       
    (4)回归问题的提升树算法
    
        输入：训练数据集T={(x1,y1),(x2,y2),(x3,y3),...,(xN,yN),} ，其中xi为特征向量，yi∈{-1,+1}
    
        输出：最终分类器f_m(x)
        
        (a) 初始化f_0(x) = 0
        
        (b) 对m=1,2,3,...,M
            (b1)计算第m轮第i个样本的残差r_m_i     
                r_m_i = yi - f_m-1(xi)，i=1,2,3,..,N
            
            (b2)基于得到的残差集，学习到一个回归树，得到T(x,Θm)   （学习方法：同CART回归树一样，选择使得划分后均方差之和最小的切割点）
            
            (b3)更新f_m(x) = f_m-1(x) + T(x,Θm)
        
        (c)得到回归问题的提升树
            f_m(x) = sum T(x,Θi))    其中 i=1,2,3,...,M
           
 3.梯度提升树(回归)算法
    前面谈到的提升树利用加法模型与前向分布算法实现学习的优化过程.当损失函数是平方损失或指数损失函数时，每一步优化比较简单，但是若是泛化到更一般
 的损失函数，每一步的优化没那么简单，针对此问题，Freidman提出了Gradient Boosting算法，这是最速下降法的近似方法.
 
    (1)利用损失函数的负梯度来作为本轮损失的近似值(残差)，那么第t轮第i个样本的残差表示为：
        r_t_i = - ∂L(yi,f(xi))/∂(f(xi))  ，其中 f(x) = f_t-1(x) 
    
    (2)利用(xi,r_t_i) 其中i=1,2,3,...,N，我们可以得到第t棵回归树，其对应的叶节点区域为R_t_j，j=1,2,3,...,J，J为第t棵回归树的叶子节点数
    
    (3)针对每一个叶子节点的样本点，我们求出使损失函数最小，也就是拟合叶子节点最好的输出值 c_t_j ，表示第t个叶子节点的输出值，公式为：
                c_t_j = arg min sum L(yi,f_t-1(xi) + c)     其中 xi ∈ R_t_j ， R_t_j为第j个叶子节点区域
                
    (4)得到每个叶子节点的最佳拟合点之后，可以得到第t棵回归树的模型为：
                h_t(x) = sum c_t_j * I(x ∈ R_t_j)
                
    (5)从而本轮最终得到的强学习器的表达式为:
                f_t(x) = f_t-1(x) + h_t(x)
                
 4.梯度提升树(分类)算法
    GBDT的分类算法从思想上与回归算法区别不大，但是由于样本输出不是连续的值，而是离散的类别，导致无法直接从输出的类别值来拟合误差.
    
    为了解决这个问题，主要有两种解决办法，一个是指数损失函数，此时GBDT退化为AdaBoost，另一个是对数似然损失函数，用预测概率值与真实概率值的差
    来拟合损失.
    
    (1)二元GBDT分类算法
        对于二元GBDT算法，如果用对数似然损失函数，则损失函数为：
                    L(y,f(x)) = log(1+exp(-y*f(x)))  ，其中y∈±1
        
        此时的第t棵树中的第i个样本的负梯度(残差)值为:
        
                    r_t_i = - ∂L(yi,f(xi))/∂(f(xi)) 其中f(x)=f_t-1(x)
         
                          = yi / (1+exp(yi*f_t-1(xi)))

        假设这n个样本的残差值分布在J个叶子节点上，叶子节点区域分别为R_t_1, R_t_2, R_t_3, ..., R_t_J
        
        对于生成的决策树，我们第j个叶子节点区域的最佳负梯度拟合值为:
                    
                    c_t_j = arg min sum(log(1+exp(-yi*f_t-1(xi)) + c))     其中xi ∈ R_t_j
                    
        由于上式比较难优化，我们用近似值代替:
                    
                    c_t_j = sum r_t_i / sum |r_t_i|*(1-|r_t_i|)     其中r_t_i是第t棵树中的第i个样本的负梯度(残差)值
          
        除了负梯度计算和叶子节点的最佳负梯度拟合的线性搜索(线性搜索是指遍历找到最佳分割点)，二元GBDT分类和GBDT回归算法过程相同.          
 
 5.GBDT如何构建特征【重要】
   
    GBDT本身是不能产生特征的，但是我们可以利用GBDT去产生特征的组合。利用GBDT去产生特征的组合，再采用逻辑回归LR进行处理，增强逻辑回归对非线性
 分布的拟合能力.
 
    举例说明一下：
    
    我们使用GBDT生成了两棵树，两颗树一共有五个叶子节点。我们将样本X输入到两颗树当中去，样本X落在了第一棵树的第二个叶子节点，第二颗树的第一个
 叶子节点，于是我们便可以依次构建一个五维的特征向量，每一个维度代表了一个叶子节点，样本落在这个叶子节点上面的话那么值为1，没有落在该叶子节点的
 话，那么值为0.于是对于该样本，我们可以得到一个向量[0,1,0,1,0] 作为该样本的组合特征，和原来的特征一起输入到逻辑回归当中进行训练.
 
    实验证明这样会得到比较显著的效果提升。

    GBDT选择特征的细节其实是想问你CART Tree生成的过程.CART TREE 生成的过程其实就是一个选择特征的过程.遍历每个特征和每个特征的所有切分点，
 找到最优的特征和最优的切分点。多个CART TREE 生成过程中，选择最优特征切分较多的特征就是重要的特征.
 
                
 6.GBDT常用损失函数
    
    (1)对于分类算法，其损失函数一般有对数损失函数和指数损失函数两种
        (a)如果是指数损失函数，则损失函数表达式为：
            L(y,f(x)) = exp(-y*f(x))
        
        (b)如果是对数损失函数，对于二元分类，表达式为：
            L(y,f(x)) = log(1+exp(-y*f(x)))  ，其中y∈±1
            
    (2)对于回归算法，常用的损失函数有如下四种
        (a)均方差，最常见的回归损失函数：
            L(y,f(x)) = (y-f(x))^2
        
        (b)绝对损失：
            L(y,f(x)) = |y-f(x)|
            
        (c)Huber损失，它是均方差和绝对损失函数的折衷，远离中心的异常点采用绝对损失，而中心附近的采用均方差损失，界限由分位数点δ度量
        
            L(y,f(x)) = 1/2 * (y-f(x))^2   ，其中|y-f(x)| <= δ
            L(y,f(x)) = δ*(|y-f(x)|-δ/2)   ，其中|y-f(x)| > δ
                    
        (d)分位数损失，对应于分位数回归的损失函数，表达式为：
            L(y,f(x)) = sum θ*|y-f(x)| (其中y>=f(x)) + sum (1-θ)*|y-f(x)| (其中y<f(x))
            
        后面两种损失函数，主要用于健壮回归，减少异常点对损失函数的影响.
 
 7.GBDT的正则化
    
    (1)第一种是和AdaBoost类似的正则化项，learning rate，定义为v，对于弱学习器的迭代：
       
        f_k(x) = f_k-1(x) + v*h_k(x)     v的取值范围为 0 < v <= 1
            
    (2)第二种是通过子采样比例(subsample)，取值(0,1]
       
        注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是  不放回抽样  !!! 
        如果取值为1，则全部样本都使用，等于没有使用子采样.如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，
        即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低.推荐在[0.5, 0.8]之间.
    
        使用子采样的GBDT有时也称作随机梯度提升树(Stochastic Gradient Boosting Tree), SGBT
        
    (3)第三种是对于里面的弱学习器即CART回归树进行正则化剪枝，这里不再赘述.
    
 8.GBDT小结
    由于GBDT的卓越性能，只要是研究机器学习都应该掌握这个算法，包括背后的原理和应用调参方法。GBDT的性能至少可以排进机器学习经典算法的前三.
 目前实现GBDT的算法比较好的库是xgboost.
    
    优点：
        (1)可以灵活处理各种类型的数据，包括连续值和离散值
        (2)在相对少的调参时间情况下，预测的准确率也可以比较高
        (3)使用一些健壮的损失函数，对异常值的鲁棒性非常强。比如 Huber损失函数和Quantile损失函数
        
    缺点：
        由于弱学习器之间存在依赖关系，难以并行训练数据。不过可以通过自采样的SGBT来达到部分并行
"""
