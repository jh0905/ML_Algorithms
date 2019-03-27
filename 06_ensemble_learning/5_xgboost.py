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
        
 3.CART回归树与集成学习
            
"""
