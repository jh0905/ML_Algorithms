# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 3/27/19 9:05 AM
 @desc: xgboost是GBDT的升级版，本文主要是根据xgboost作者的slides以及网上相关博客，对xgboost的原理进行一定的梳理.
"""

"""
 引言：
    xgboost，全称是 extreme gradient boosting （极端梯度提升），它是大规模并行boosted tree的工具，是目前最好最快的boosted tree的开源
 工具包，xgboost所应用的算法，就是GBDT的改进，为了更容易理解xgboost的原理，我决定按照以下的顺序，对xgboost的知识进行梳理.
        
 1.XGBoost和GBDT的区别
    前面我们学习了GBDT的算法原理，那么趁热打铁，先大致了解一下二者的区别，方便我们在后面公式的推导中，有一个初步的认识 ：）
    
    (1)GBDT以CART回归树作为基学习器，XGBoost除此之外还支持线性学习器，此时XGBoost相当于带有L1和L2正则化的linear/logistic regression；
    
    (2)GBDT在求解损失函数最优化的时候只用到一阶导数，XGBoost则对损失函数进行了二阶泰勒展开（不难，后面有讲解），得到一阶和二阶导数；
    
    (3)XGBoost在损失函数中加入了正则项，用于控制模型的复杂度。从权衡方差偏差来看，它降低了模型的方差，使学习出来的模型更加简单，防止过拟合，
       这也是XGBoost优于传统GBDT的一个特性；

    (4)shrinkage（缩减），相当于学习速率（XGBoost中的eta）.XGBoost在进行完一次迭代时，会将叶子节点的权值乘上该系数，主要是为了削弱每棵树
       的影响，让后面有更大的学习空间.（GBDT也有学习速率）；

    (5)列抽样.XGBoost借鉴了随机森林的做法，支持列抽样，不仅防止过拟合，还能减少计算；

    (6)对缺失值的处理.对于特征值有缺失的样本，XGBoost还可以自动学习出它的分裂方向；

    (7)XGBoost工具支持并行.
       
    【疑问】Boosting不是一种串行的结构吗?怎么并行的？
       
       注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值），
    XGBoost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序(因为要确定最佳分割点)，XGBoost在训练之前，
    预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量.这个block结构也使得并行成为了可能，在进行节点
    的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行.
 
 2.回顾一下监督学习中的一些基本概念
    
    (1)模型（根据输入的样本xi，如何生出预测值y_hat_i）            注：xi为第i个样本的特征向量，y_hat常用符号，表示预测值
        
       比如说：线性模型：y_hat = w.T * x （包含linear/logistic regression）
            对于linear regression  ，y_hat就是最终的预测值
            对于logistic regression，sigmoid(y_hat)就是样本x属于正例的概率值
            
    (2)参数（我们需要从训练集中学习到的东西）
       在线性模型中，需要学习的参数向量 θ = (θ1,θ2,θ3,...,θN)
       
    (3)目标函数 （objective function）
        目标函数贯穿了所有机器学习的算法模型，它的通用表达形式为：
                Obj(θ) = L(θ) + Ω(θ)    ，其中L(θ)表示training loss，用来度量模型在训练集上的拟合情况
                                             Ω(θ)表示regularization，用来度量模型的复杂度
       
        (a) Loss on training data ： L = sum l(yi,y_hat_i)  ，i=1,2,3,..,N
            
            平方损失： l(yi,y_hat_i) = (yi - y_hat_i)^2
            
            逻辑损失： l(yi,y_hat_i) = -yi*(ln h(xi)) - (1-yi)*ln(1-h(xi))  
                前面讲逻辑回归的时候提过，对数似然再取反，得到损失函数，其中h(xi) = sigmoid(w.T*xi) = sigmoid(y_hat_i) 
                把逻辑损失式子中的符号，提到ln()中去，可得l(yi,y_hat_i) = yi*(1+exp(-y_hat_i)) + (1-yi)*(1+exp(y_hat_i)) 
    
        (b) Regularization ： how complicated the model is ?
            
            L1 norm :  Ω(θ) = λ * ||θ||1        其中||θ||1表示l1范数，指向量中每个元素绝对值的和
            L2 norm :  Ω(θ) = λ * ||θ||2 ^ 2    其中||θ||2表示l2范数，指向量中每个元素的平方和再开方   
                                                补充：l0范数是指向量中所有非零元素的个数

            L1正则化会使许多权重参数的最优值变成0，使得模型变得稀疏稀疏     
            L2正则化通过权重衰减，保证了模型的简单，提高了泛化能力     
    
        (c) 整合上面的内容：
            · Ridge Regression： 
                Obj(θ) = sum(yi-w.T*xi)^2  +  λ*||w||^2             其中，i=1,2,3,...,N
                （linear model，square loss，L2 regularization）
                
            · Lasso Regression：
                Obj(θ) = sum(yi-w.T*xi)^2  +  λ*||w||1              其中，i=1,2,3,...,N
                （linear model，square loss，L1 regularization）
            
            · Logistic Regression：
                Obj(θ) = sum(yi*ln(1+exp(-w.T*xi)+(1-yi)*ln(1+exp(w.T*xi)))  +  λ*||w||^2      其中，i=1,2,3,...,N
                （linear model，logistic loss，L2 regularization）
                
    (4)偏差和方差的均衡(trade-off)
        
        为什么在目标函数中，会有L(θ)和Ω(θ)这两项呢?原因如下：

        我们希望L(θ)尽可能小，好处就是训练模型的精度高，在训练集上的偏差bias很小，但是也会使得模型更加复杂，模型的方差variance大，容易
    过拟合；另一方面我们也希望Ω(θ)能够尽可能小，也就是使得权重参数减小，模型趋向于简单化，好处是模型泛化能力强，但是模型的偏差bias偏大，
    因此我们的目标函数是L(θ)+Ω(θ)，就是希望找到一组参数θ，能够兼顾模型的bias和variance.
        
 3.xgboost目标函数的推导过程
    
    (1)模型
        假设我们有K棵CART回归树，则对于训练样本xi，模型预测的y_hat_i为：
        y_hat_i = T_1(xi) + T_2(xi) + T_3(xi) +...+ T_K(xi)   ，其中T_k(x)表示特征向量x在第k棵树中的输出值
                
                = sum T_k(xi) 其中k=1,2,3,...,K
                
    (2)参数
        集成树模型中的参数，包括两方面，一是每一棵建立的回归树的结构，二是每一棵回归树叶子节点的分数.
        所以我们要学习的不再是前面的权重向量，而是要学习每一棵子树的模型（function）
        
    (3)目标函数
        Obj = Training LOSS + Complexity of the Tree
            
            = sum l(yi,y_hat_i) + sum Ω(T_k)    （其中i=1,2,3,...,N  ，  k=1,2,3,...,K）
        
        Ω(T_k)看起来有些抽象，有以下几种定义的办法：
            (a)回归树中节点的数量
            (b)回归树的深度
            (c)回归树中，叶子节点权重的L2范数
    
    (4)我们如何学习呢?
        想想xgboost是针对GBDT算法的改进，所以我们不妨用求解GBDT的思想，试着求解xgboost.
        
        (a)Additive Training (Boosting)
        
            初始化y_hat_i_0 = 0，每轮训练一个模型（CART回归树），于是有： 【定义y_hat_i_j为样本xi在第j轮迭代后，模型的预测值】
        
            y_hat_i_0 = 0
            y_hat_i_1 = T_1(xi) = y_hat_i_0 + T_1(xi)
            y_hat_i_2 = T_1(xi) + T_2(xi) = y_hat_i_1 + T_2(xi)
            y_hat_i_2 = T_1(xi) + T_2(xi) + T_3(xi) = y_hat_i_2 + T_3(xi)
             
                                           '''''
            y_hat_i_t = sum T_k(xi) （其中k=1,2,3,...,t） = y_hat_i_t-1 + T_t(xi)
        
        (b)求解T_t(xi)
        
            我们知道在第t轮的迭代中，y_hat_i_t =  y_hat_i_t-1 + T_t(xi)
            那么我们在第t轮的目标函数为：
            Obj_t = sum l(yi,y_hat_i_t) + sum Ω(T_k)    （其中i=1,2,3,...,N ， k=1,2,3,...,t）
                    
                  = sum l(yi, y_hat_i_t-1 + T_t(xi) ) +  Ω(T_k) + constant （其中i=1,2,3,...,N）
                  
                                                                【前面k-1棵树的结构已经固定，于是用常量constant来代替】
            
            如果损失函数定义为平方损失的话，Obj_t即为：
            
            Obj_t = sum(yi - y_hat_i_t-1 - T_t(xi))^2  + Ω(T_k) + constant  （其中i=1,2,3,...,N）
            
            其中，yi是已知的常量，y_hat_i_t-1也是在前t-1轮中已知的预测值，于是我们不妨把这两个看成一个整体，即(yi - y_hat_i_t-1)，这个
        也就是我们在boosting tree中提到的residual(残差)
        
            那么 (yi - y_hat_i_t-1 - T_t(xi))^2 = 2*(yi - y_hat_i_t-1)*T_t(xi) + T_t(xi)^2 + constant
            
            于是：
            Obj_t = sum(2*(yi - y_hat_i_t-1)*T_t(xi) + T_t(xi)^2)  + Ω(T_k) + constant （其中i=1,2,3,...,N，常量+常量=常量）
            
        (c)用泰勒二阶展开式来近似估计损失l(yi,y_hat_i_t)【强推 https://www.zhihu.com/question/25627482/answer/313088784 理解泰勒】
            
            上面是将损失函数定义为平方损失时的情形，我们如何让模型更加通用呢?
            
            在GBDT的学习中，它提到的思想是用损失函数的负梯度值，来做残差(yi-y_hat_i)的近似估计，xgboost换了一个思想，它是把损失函数做二阶
        泰勒展开，作为损失函数的近似估计.
            
            回顾一下， 函数f(x)二阶泰勒展开式为：
                
                f(x+Δx) = f(x) + f'(x)*(Δx) + 1/2 * f''(x) * (Δx)^2
            
            那么损失函数l(yi,x)二阶泰勒展开式为：     
                
                l(yi, x+Δx) = l + l' * (Δx) + 1/2 * l'' * (Δx)^2
                        
            用gi来表示l(yi,y_hat_i_t-1)对y_hat_i_t-1求一阶导，用hi来表示l(yi,y_hat_i_t-1)对y_hat_i_t-1求二阶导
            
            然后再用Δx = y_hat_i - y_hat_i_t-1 = T_t(x)，于是可得：
            
                l(yi,y_hat_i_t-1 + T_t(xi)) = l(yi,y_hat_i_t-1) + gi*T_t(xi) + 1/2 * hi*T_t(xi)^2
                
            于是，经过二阶泰勒展开之后，我们在第t轮的目标函数为：
                
                Obj_t = sum(l(yi,y_hat_i_t-1) + gi*T_t(xi) + 1/2 * hi*T_t(xi)^2) + Ω(T_k) + constant（其中i=1,2,3,...,N）
                
            于是我们又惊喜的发现，l(yi,y_hat_i_t-1)，不也是一个已知量吗?赶紧把它合并到后面的constant中去，得到最终的形式为：
                
                Obj_t = sum(gi*T_t(xi) + 1/2 * hi*T_t(xi)^2) + Ω(T_k) + constant（其中i=1,2,3,...,N）
                
            【整个世界从未如此的整洁有木有，哈哈哈~】
        
        (d)   
                      
"""
