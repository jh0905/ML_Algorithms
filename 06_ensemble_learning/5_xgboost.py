# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 3/27/19 9:05 AM
 @desc: xgboost是GBDT的升级版，本文主要是根据xgboost作者的slides以及网上相关博客，对xgboost的原理进行全面的梳理.
"""

"""
 引言：
    xgboost，全称是 extreme gradient boosting （极端梯度提升），它是大规模并行boosted tree的工具，是目前最好最快的boosted tree的开源
 工具包，xgboost所应用的算法，就是GBDT的改进，为了更容易理解xgboost的原理，我决定按照以下的顺序，对xgboost的知识进行梳理.
        
 1.XGBoost和GBDT的区别
    前面我们学习了GBDT的算法原理，那么趁热打铁，先大致了解一下二者的区别，方便我们在后面公式的推导中，有一个初步的认识 ：）
    
    (1)GBDT以CART回归树作为基学习器，XGBoost除此之外还支持线性学习器，此时XGBoost相当于带有L1和L2正则化的linear/logistic regression；
    
    (2)GBDT在求解损失函数最优化的时候只用到一阶导数，XGBoost则对损失函数进行了二阶泰勒展开（不难，后面有讲解），得到一阶和二阶导数；
    
    (3)XGBoost在损失函数中加入了正则项，引入了Ω(T_k)的概念，用于控制模型的复杂度，这也导致了xgboost在对决策树模型的表示上和GDBT不一样。
       从权衡方差偏差来看，它降低了模型的方差，使学习出来的模型更加简单，防止过拟合，这也是XGBoost优于传统GBDT的一个特性；

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
        
 3.xgboost目标函数的推导过程         【本文最精华的地方，核心中的核心】
    
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
            y_hat_i_3 = T_1(xi) + T_2(xi) + T_3(xi) = y_hat_i_2 + T_3(xi)
             
                                           '''''
            y_hat_i_t = sum T_k(xi) （其中k=1,2,3,...,t） = y_hat_i_t-1 + T_t(xi)
        
        (b)求解T_t(xi)
        
            我们知道在第t轮的迭代中，y_hat_i_t =  y_hat_i_t-1 + T_t(xi)
            那么我们在第t轮的目标函数为：
            Obj_t = sum l(yi,y_hat_i_t) + sum Ω(T_k)    （其中i=1,2,3,...,N ， k=1,2,3,...,t）
                    
                  = sum l(yi, y_hat_i_t-1 + T_t(xi) ) +  Ω(T_k) + constant （其中i=1,2,3,...,N）
                  
                                                【前面k-1棵树的结构固定，于是用常量constant来代替sum Ω(T_k) k=1,2,3,...,t-1】
            
            如果损失函数定义为平方损失的话，Obj_t即为：
            
            Obj_t = sum(yi - y_hat_i_t-1 - T_t(xi))^2  + Ω(T_k) + constant  （其中i=1,2,3,...,N）
            
            其中，yi是已知的常量，y_hat_i_t-1也是在前t-1轮中已知的预测值，于是我们不妨把这两个看成一个整体，即(yi - y_hat_i_t-1)，这个
        也就是我们在boosting tree中提到的residual(残差)
        
            那么 (yi - y_hat_i_t-1 - T_t(xi))^2 = 2*(yi - y_hat_i_t-1)*T_t(xi) + T_t(xi)^2 + constant
            
            于是：
            Obj_t = sum(2*(yi - y_hat_i_t-1)*T_t(xi) + T_t(xi)^2)  + Ω(T_k) + constant （其中i=1,2,3,...,N，常量+常量=常量）
            
        (c)用泰勒二阶展开式来近似估计损失l(yi,y_hat_i_t)【强推https://www.zhihu.com/question/25627482/answer/313088784理解泰勒】
            
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
                
                Obj_t = sum( gi*T_t(xi) + 1/2 * hi*T_t(xi)^2) + Ω(T_k) + constant（其中i=1,2,3,...,N）
                
            【整个世界从未如此的整洁有木有，哈哈哈~】
        
        (d) 接下来的问题，那就是如何量化回归树的复杂度Ω(T_k)了?   
        
            (d1)定义回归树的结构
            
                在前面的学习中，我们知道一棵CART回归树的结构可以这样表示：
                
                T_t(x) = sum c_i * I(x∈Ri)      其中c_i为Ri区域的输出值 ， i=1,2,3,..,m
                
                我们知道根节点到叶子节点的路径是唯一且固定的，因此我们可以把根节点到叶节点的路径上所有节点的分数c累加起来，作为样本x在该棵树
            中的最终输出.
            
                假设某棵回归树中有T个叶子节点，我们用向量w=(w_1,w_2,w_3,...,w_T)表示每个叶子节点的分数，再定义一个叶子节点映射函数q(x)，
            用来把样本x映射到某个叶子节点上，于是CART回归树的结构变为：
            
                T_t(x) = w_q(x) ，其中q(x)的取值为[1,2,3,...,T]
                
            (d2)定义树的复杂度
            
                Ω(T_t(x)) = γ*T + 1/2 * λ * sum w_j ^ 2   ，其中γ表示gamma,T为叶子数，λ为l2正则系数，j=1,2,3,...,T
        
        (e)再回过头来看看目标函数
        
            Obj_t = sum( gi*T_t(xi) + 1/2 * hi*T_t(xi)^2) + Ω(T_k) （其中i=1,2,3,...,N，常量不用考虑了，可去掉）
            
                  = sum( gi*w_q(xi)) + 1/2 * hi*w_q(xi)^2) + γ*T + 1/2 * λ * sum w_j ^ 2 （其中i=1,2,3,...,N，j=1,2,...,T）
                  
            定义在第j个叶子节点中的样本点集合为I_j , I_j = {i | q(xi) = j}，那么所有样本点在回归树中的分数总和，可以用每一个叶子节点的
        分数w_i乘以该叶子节点中的样本总数，再求和得到. 公式表示为：
            
                sum w_q(xi) = sum size(I_j)*w_j      其中j=1,2,3,...,T，size(I_j)表示集合I_j中的元素个数
                
            于是进一步转化Obj_t，得：
            
            Obj_t = sum ( gi*size(I_j)*w_j + 1/2 * (hi*size(I_j) + λ)*w_j^2 )  +  γ*T  其中j=1,2,3,...,T
            
            由初中数学知识我们都知道，当H>0时 ，对于函数f(x) = 1/2 * H * x^2 + G * x
            
            函数最小值在 x = -G/H 处取得，并且最小值为f(-G/H) = -1/2 * G^2 / H
            
            那么我们在上面Obj_t的式子中，定义 gi*size(T_j) = G_j  ， 定义 hi*size(T_j) = H_j ， 可得Obj_t为：
            
            Obj_t = sum ( G_j*w_j + 1/2 * (H_j + λ)*w_j^2) + γ*T  ，其中j=1,2,3,...,T           
            
            假设本轮回归树的结构已经固定，即q(x)已经构造完成，那么此时Obj_t取的极小值的点为：
            
            w_j = -G_j/(H_j+λ)   
             
            此时对应最小的Obj_t
             
            Obj_t = -1/2 * sum G_j^2/(H_j+λ) + γ*T 其中j=1,2,3,...,T 
                
        (f) 构建每一轮的回归树
        
            经过不懈的努力，我们终于知道了，构建新一轮的回归树的目标函数为：
                  
                   Obj_t = -1/2 * sum G_j^2/(H_j+λ) + γ*T 其中j=1,2,3,...,T       
            
            理论上的做法，是根据训练样本集(xi,yi - y_hat_i_t-1)（i=1,2,3,...,N），穷举出所有可能的树结构，然后找到Obj_t最小的那一个
        作为本轮生成的回归树，但是实际上，由于计算量过于庞大，穷举法不可行，我们采用贪心策略，具体如下：
        
            (f1)从树的深度0开始（此时为空树）
            
            (f2)对当前树中的每一个叶子节点，试图找到一个最佳分割点，于是：
            
                左孩子节点的Obj_L = -1/2 * G_L^2/(H_L+λ) + γ*1
                
                右孩子节点的Obj_R = -1/2 * G_R^2/(H_R+λ) + γ*1
                
                分割前的节点Obj = -1/2 * (G_L+G_R)^2 / (H_L+H_R +λ) + γ*1
            
                Gain = Obj - (Obj_L + Obj_R)
                
                     = 1/2 * ( G_L^2 / (H_L +λ) + G_R^2 / (H_R +λ) - (G_L+G_R)^2 / (H_L+H_R +λ) )  - γ
            
            (f3)最佳分割点查找算法
                
                · 对于训练集中的每一个特征，按照特征值进行排序；
                
                · 然后用线性扫描法将特征值划分成左右两部分；
                
                · 根据f(2)计算Gain，其中Gain最大的对应的特征值，作为该特征的最佳划分点
                
                · 对其它所有特征也进行上述操作，最终所有特征中最大的Gain对应的特征及特征值作为本轮的最佳分割点
                
                这里可以节省时间
                
                    一是并行的计算分裂节点的增益
                    
                    二是特征值的排序结果可以保存在cache中
                    
                    三是对于特征值较多的特征提出了Weighted Quantile Sketch（带权重的分位点算法）
                
                最终的时间复杂度为 O(d * K * n * log n) ， 其中n为样本数，d为特征数，K为树的深度，n*log n 为特征值排序的时间
                
            (f4)如何处理类别型数据呢
            
                通过one-hot编码，将类别型特征转成{0,1}向量的形式，也不用担心数据集会变得稀疏的问题，xgboost有专门的稀疏矩阵处理技术
                                                                                                    
            (f5)剪枝与正则化
                
                回想一下前面提到的Gain:
                
                Gain = 1/2 * ( G_L^2 / (H_L +λ) + G_R^2 / (H_R +λ) - (G_L+G_R)^2 / (H_L+H_R +λ) )  - γ
                
                · Pre-stopping （早停止）
                
                    当“叶子节点”继续往下分裂，产生的Gain就会越来越少，当Gain<=0的时候，我们可认为已经达到了最佳深度，就不往下继续分裂了，
                但是，也有可能仍然继续往下分裂时，后面会产生一些好的分裂节点；
                
                · Post-Pruning （后剪枝）
                    
                    也就是我们所熟知的后剪枝技巧啦，先把树生成的指定的最大深度，然后递归地从带有负增益(negative gain)的叶子节点开始剪枝
    
    (5) xgboost算法完整回顾：
    
        (a) 在每一轮迭代中新增加一棵树（基于残差的前向学习）
        
        (b) 每一次迭代前，计算 gi 和 hi 的值 （gi是第i个训练样本代入到loss函数中，一阶导数的值，hi为二阶导数的值）
        
        (c) 基于数理统计来贪心的生成一棵树T_t(x) （Obj = sum ( G_j*w_j + 1/2 * (H_j + λ)*w_j^2) + γ*T  ，其中j=1,2,3,...,T）
        
        (d) 把新生成的树，添加到模型中 f_t(x) = f_t-1(x) + T_t(x)
            
                （通常，我们用 f_t(x) = f_t-1(x) + ε * T_t(x) ，用到了缩减的思想，通常取0.1，也是俗称的学习步长，防止过拟合）
                

"""
