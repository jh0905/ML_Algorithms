# ML_Algorithms
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本项目是我整理机器学习经典算法时的学习笔记，主要是以代码的形式，来加深对于机器学习算法的了解，在每一份代码中，我都以注释的形式，来讲解算法背后的数学原理，有公式的我会附上手稿推导的照片，在大部分算法的具体实现前，我都给出了详细的伪代码，以便自己能够独立编程实现算法。
另外，对于每一个算法，我还写了一个xx_conclusion.py文件，里面主要是关于这个算法的学习总结，主要是优缺点或者其他一些需要注意的点之类的。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我在做学习总结时，主要学习资料是《机器学习实战》&《统计学习方法》&《机器学习》&《网络博客》，其中，
我主要是以《机器学习实战》为主，然后在《统计学习方法》中找到对应的章节，了解背后原理，同时也在西瓜书《机器学习》中，找找有没有什么遗漏的知识点，或者
利用案例来检查自己的了解情况，如果前面的讲解仍然不能让我搞清楚某个算法的话，再会去网上找相关的博客，主要是刘建平老师的博客，写的深入浅出，推荐一波。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以上就是这个项目的介绍，目前仍在更新中，学而不思则罔，加油～

# 记录本项目的工作日志

### 2019/3/26 星期二 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 今天主要学习的内容是bagging框架下的随机森林算法，随机森林是bagging的一个进化版，主要体现在采样时对样本的特征也进行了随机采样，俗称列采样。由于弱学习器之间没有依赖关系，所以它最大的优点则是可以很方便的并行训练!然后需要说一下的是，它用到了自助采样，于是多了一个OOB(out of bag)的概念，即在bagging的每轮随机采样中，大概有36.8%的样本没有被采样过，方便我们用于模型检验.
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 随机森林还需要提一下的就是它的几个变种，ERF(极端随机森林)，决策树的构建时，训练集为原始集，并随机选择一个特征作为分割点；TRTE，一种无监督学习方法，用于把特征从低维到高维转换；最后还有一种是IForest,是一种异常点检测的方法.
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;再聊一聊RF还有一个功能，就是评估特征的重要性，有两种度量办法，一是基于OOB的错误率，二是基于基尼指数，具体计算方法在4_rf.py里有提及到.
 
### 2019/3/25 星期一

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 今天做了两件事，一是重新梳理了一下AdaBoost的笔记并做了一些补充，二是对于GBDT做了一个基本的介绍.
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AdaBoost算法，它的损失函数是指数函数L(y,f_m(x))=exp(-y\*f(x))，我们计算每一个基模型的系数以及模型的参数是通过前向学习算法得到的，何为前向学习算法呢？ 即 f_m(x) = f_m-1(x) + α\*G(x)，于是我们把求多个基模型的系数及参数转为每次只求1个模型的方式，即第m个基模型G_m(x)和对应的权重系数α_m为 ： α_m,G_m(x) = arg min (sum L(y,f_m-1(x)+ α\*G(x))，由此我们可以得到分类器权重系数的计算式和样本权重的更新式。
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GBDT是一种提升树类算法中效果最好的一个模型，先来说说提升树算法吧，它的思想我觉得一句话描述就是“基于残差的学习算法”，每一轮的训练样本都会改变，改变方法就是 本轮训练的样本值xi与本轮训练完之后的预测值之间的差值.然后一直迭代下去，直至均方差之后达到指定的阈值或者达到迭代轮数.那么梯度提升树算法的核心是什么呢？它的作用其实就是用 损失函数的负梯度 来 近似 残差，也就是说现在残差不再用减法来求了.另外呢，GBDT也是一种前向学习算法，首先根据上一轮的学习，我们的样本分布在了不同的叶子节点上，于是对每一个样本求完残差(负梯度)之后，针对每一个叶子节点领域，找到一个最能拟合该叶子节点所有样本的值，然后把每一个叶子节点的最佳拟合值组合起来，就得到了该轮的最佳拟合回归树.上面谈的是GBDT用于回归问题，那么GBDT用于分类问题呢，这里说一下二元分类，它把对数似然函数作为损失函数，仍然用负梯度近似残差，然后样本分散在各个叶子节点之后，每个叶子节点的最佳拟合值也为了计算方便，换了其他方式来近似代替，一句话说就是GBDT用于分类主要是损失函数换了和叶子节点最佳拟合也换了，其余过程都一样.
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最后稍微提一句，不管GBDT用于分类还是回归，它的基模型都是CART回归树，没有用分类树.
 
### 2019/3/24 星期日 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 今天简单的了解了一下AdaBoost的算法流程，涉及到样本权重的初始化，每一轮训练完之后，误差率的计算、弱分类器的权重计算、训练样本的权重更新，迭代下去，直至达到停止条件(误差率达到接受范围或者学习器的数量达到制定的数量)

### 2019/3/22 星期五

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;差不多一周没更新了...并不是因为我这几天偷懒了，而是最近一着忙着毕业论文开题报告，在网上搜刮各种paper看，当然效率也不是很高，只能说还是要坚持看下去吧。至于今天为啥更新了呢？是因为看论文实在无趣味，所以就奖励自己来学习机器学习算法啦，哈哈~

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;废话少说，今天主要看了机器学习的一个大杀器，ensemble learning，中文名叫集成学习，大概就是“三个臭皮匠赛过诸葛亮”的思想咯。集成学习有三种框架，一是bagging,千万不要以为是 bag(包)?! 完全无关，它是 bootstrap aggregating 的缩写，全称也不用管那么多，它的核心就是选取一大堆的强学习器，然后通过有放回采样的方式给每一个base model分配一个训练子集，最终多数投票或者取均值完成分类或回归任务，代表算法是RandomForest；二是boosting，可以理解为"learning from mistakes"，针对上一个模型中的错误，在下一个模型中进行改进，是一种迭代式的算法，它的base model是弱学习器.而boosting框架中，每个base model的权重是不一样的，最终通过加权的方式生成预测结果，代表算法是AdaBoost和GBDT；三是stacking，一般来说是一个两层结构，第一层是多个不同的强学习器，每一个学习器对样本生成一个预测结果，然后组成一个新的特征矩阵，输入给第二层，第二层的模型为简单模型，比如LogisticRegression，为的是防止过拟合，更多内容在1_introduction.py里。

### 2019/3/17 星期日

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;今天根据网上的一些文章，把前天简易版SMO算法中存疑的几个点理解了一下。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一是之前没弄懂算法中，对于alpha[i]是否需要优化的判断条件，为什么是那样子的。实际上还是计算当前的yi*g(xi)与1的关系，根据KKT对偶互补条件，如果该值大于1的话，那该样本对应的权重alpha[i]需要减小到0，如果该值小于1的话，那该样本对应的权重alpha[i]需要增大到C，这一点还是参考前面的学习内容；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;二是为什么L==H时，就不对当前alpha[i]进行优化了，回顾之前的学习，L和H是根据上一轮alpha[i]移动的区间计算出来的，如果上下界都相等了，那就说明本轮alpha[i]也就没有移动的范围了，无须优化；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;三是eta到底干嘛用的，看笔记发现，之前用αi表示αj，代入到优化目标函数中，再对αj求导时计算出来的，为了方便，令 η =  Kii+Kjj-2Kij，即eta就是这个式子的代替，后面会被作为分母计算αj_new_unc，也解释了为什么它为0，循环跳出去不对其进行优化；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;四是外循环到底是干什么用的，因为一轮内循环中，所有的变量αi都不进行优化时，迭代次数才加1，然后画了一下分类图，此时也是能分类啊，那外循环有什么用呢？实际上啊，看看输出就知道啦，的确存在某轮内循环中，所有的变量αi都没有进行优化，但是由于αj选择的随机性，仍然不能保证所有的变量αi真的都已经优化完毕，分析输出可知，之后的迭代中，仍然还是发生了αi的更新，而且不在少数，并且只要发现了<αi,αj>对发生了更新，n_iter重新置为了0，因此，外循环的作用其实就是看α向量是不是真的收敛了，如果连续max_iter轮都没有发生任何αi的更新的话，那就是证明，我们已经求得了最优解!

### 2019/3/15 星期五

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;今天主要是来理解之前抄书上面的simple_smo的代码实现，simple_smo的主要思想是：程序分为外循环和内循环，先从内循环说起吧，内循环的目标就是启发式的对α向量优化，首先判断αi是否需要优化，是的话，再随机挑一个αj，然后对它们开始优化，如果这两个变量在优化结束前还没有跳出去的话，就说明这轮变量完成了优化，将alpha_pairs_changed的值加1，当内循环遍历完数据集中所有的样本点之后，如果alpha_pairs_changed大于0，就再继续一轮内循环的遍历，直到所有的alpha_pairs都不需要优化，我们才将迭代次数加1!也就是说此时算出来的alpha向量已经可以用来分类，实际上我也绘制了迭代次数为1时样本的分类图，的确能够正确分类；那我只能认为外循环是将alpha向量进一步调优的过程，因为是启发式的算法，所以尽可能的把所以的α变量调到精度范围内吧，之后再好好看看这个算法吧，把这些疑问解了。

### 2019/3/14 星期四 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 终于可以骄傲的说，SMO我弄懂啦！！！哈哈哈，容我先小得瑟一会儿，其实肯定还有一些地方没弄懂，但是今天一整天的学习，感觉把SMO的精髓给掌握住了，α的分布其实是有特点的，每一个αi对应着一个样本点，样本点与分类超平面或者间隔边界的位置关系，也决定了αi的值，这里就不再赘述了。最后整体回顾一下SMO算法的流程吧!
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先是按照一定原则，选择αi和αj；然后用αi表示αj，代入到目标函数中，求偏导数，令其值为0，得到对应的αj_new_unc，但是这个新值，必须满足约束条件，所以我们要对它进行裁剪，然后得到αj_new，再由αi和αj之间的关系，得到αi_new，我们就把αi和αj更新完啦，如果它们的值满足0<αi<C的话，我们就认为它对应的样本点，就处于间隔边界上，对于我们求解b起决定性作用，其他不在间隔边界上的点，对我们求解b没有任何帮助，不考虑，于是我们得到b_new之后，就可以更新Ei的值了，然后开始下一轮的迭代工作。SMO算法大体上就是这样了，具体学习笔记，看2_svm_algorithm.py .

### 2019/3/13 星期三 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 在连续迷迷糊糊好几天快要放弃SVM的学习的时候，感觉今天灵光一闪，对于SVM的整个求解过程明朗了许多。在这里记录一下，首先是把SVM间隔最大化转成原始问题的形式，即min 1/2 * ||w||2   s.t. 1 - yi(wT*xi+b) ≤ 0的形式，然后把它经过拉格朗日乘子转换，变形为L(w,b,α)的形式，min max L(w,b,α)的最优化问题，如果满足KKT条件的话，可以用 max min L(w,b,α)来求解，KKT条件的来源有对w,b求偏导数，值为0时所得出的式子，以及朗格朗日乘子的默认要求α≥0，和拉格朗日乘子对应的拉格朗日项≤ 0，
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 掌握了这些，就不觉得KKT条件难了，然后后面所说的一起，都要满足KKT条件，才能求解，继续说到max min L(w,b,α)，对w,b求偏导赋值为0后，代入L函数，最终得到一个仅关于α的函数，我们就可以用SMO算法来求解,明天具体说说SMO，不再那么迷糊了。
另外今天还学习了软间隔最大化，引入松弛变量的处理，没想到最终得到的优化目标函数与之前的形式一致，只是KKT条件多了一条 0≤α≤C,然后又从纯数学式的角度，弄清楚如何判断一个点与间隔边界和分类超平面的位置，总的来说，今天收获很多!

### 2019/3/12 星期二 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 在昨天的基础上，继续学习SMO算法的原理，SMO算法同时固定两个参数，固定其他N-2个参数，假如选择的变量是α1，α2，固定其他参数α3，α4，... , αN ，可以简化目标函数为只关于α1，α2的二元函数，利用约束条件，可以用α2表示α1，最终得到只关于α2的一元函数，再对此一元函数求极值点，然后对原始解进行修剪，计算出新的α1，再取临界情况，整体来说还是挺复杂的，只搞懂一些皮毛。
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;另外一件事呢，就是简单的看了一下核函数，一句话概括它的作用就是把SVM的优化目标函数中的(xi∙xj)换成了K(xi,xj),在低维进行计算，而将实质上的分类结果表现在高维上，避免了在高维空间中的复杂计算。

### 2019/3/11 星期一 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 先摘录一段话，“拉格朗日对偶的重要作用是将w的计算提前并消除w，使得优化函数变为拉格朗日乘子的单一参数优化问题，而SMO算法就是解上述问题的快捷算法。在解决最优化的过程中，发现了w可以由特征向量内积来表示，进而发现了核函数，仅需要调整核函数就可以将特征进行低维到高维的变换，在低维上进行计算，实质结果表现在高维上。由于并不是所有的样本都可分，为了保证SVM的通用性，进行了软间隔的处理，导致的结果就是将优化问题变得更加复杂，然而惊奇的是松弛变量没有出现在最后的目标函数中。最后的优化求解问题，也被拉格朗日对偶和SMO算法化解，使SVM趋向于完美。”
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这段话可能是对SVM算法最直白的阐述吧，因此记录了下来。今天做的事情也不多，主要是照着《机器学习实战》上的代码，实现了SMO算法的简单实现，不过底层原理也还没搞清楚，继续学习吧。

### 2019/3/8 星期五 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 今天终于能够开始对SVM这个算法系列动笔了，昨天心浮气躁地在网上翻阅了一天的博客，反而是越看越恼火，好像有收获，又好像什么都没学到。今天就是把昨天所看的知识点，进行了梳理，弄懂了函数间隔与几何间隔的由来以及二者的联系，从间隔再推到最大化间隔的由来，然后弄明白了约束条件yi(wT*xi+b) ≥ 1中，为什么是1而不是0,然后今天的重点就是拉格朗日对偶问题了，重点学习了原始问题如何推导到极小极大问题、极小极大问题再怎么推到对偶问题，后面就是得出结论，在满足KKT条件下，原始问题和对偶问题的最优值相等，这个地方我还是糊涂的。后面就是解对偶问题了，然后用到SMO算法来求解，暂时还没做了解，到这一步了，就接上了《机器学习实战》中的内容了，这本书就是直接用SMO算法进行求解SVM，之后先从代码的角度来尝试理解吧，今天就先这样了。

### 2019/3/7 星期四 

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;今天一天干了些什么呢？一句话概括就是在网上看了各种各样的讲解SVM(支持向量机)的博客，所以主要是以看为主，没有做一些实质性的事情，简单概括一下我的收获。普通的SVM呢，是一个线性分类器，用于二分类任务，将线性可分的样本分隔开，并且最终得到的是最优分类面，即分类面到两个不同类型样本的距离最大，如果想划分线性不可分的数据呢？那就需要核函数的技巧，将样本映射到高维空间，使其可被一个超平面分隔开。
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;未来要做的事情是，介绍SVM的分类函数，介绍函数间隔与几何间隔的联系，由此再介绍y(i)(wT*x(i)+b) ≥ 1是怎么来的，然后是min 1/2||w|| ,s.t. y(i)(wT*x(i)+b) -1≥0的由来，以及如何通过拉格朗日对偶问题来求解，然后介绍KKT条件，以及最终如何求解出w和b,然后通过SMO代码，来实战练习SVM算法。这一节的确是很难啃的骨头，有点耐心吧。

### 2019/3/6 星期三 

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;关于逻辑回归这个算法，我原以为是一个很简单的模型，了解其原理通过代码实现，我预计最多花费两天的时间，而实际上，却花了一倍的时间，时间主要花在哪呢？一部分用在梯度下降法的背后原理，一部分用在理解梯度下降的向量推导上。之前对于梯度下降法，也是似懂非懂，而这个正好是面试的重点，不针对与LR模型，其他模型也需要详细掌握梯度下降背后的原理，所以现在了解了梯度下降的由来了，也知道向量化公式是如何推出来的了。
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  总的来说，这次对于LR的学习，收获是非常的多，从求解参数θ的角度，来看到机器学习的整个过程，无非两步，一是最好是找到具有凸函数性质的损失函数，二是通过梯度下降等方法求解θ。因此，我更深入地接触到了最大似然估计和对数似然函数，知其然还知其所以然，了解了两种不同定义损失函数的方法，以及如何用批量梯度下降（全样本）、随机梯度下降（单样本）法求解回归系数的过程，受益良多。

### 2019/3/5 星期二

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在昨天的基础上，今天学习了通过矩阵的形式，计算梯度下降中θ的更新式，然后再以代码的形式来实现这个算法。不太顺利，从向量的角度来看这个问题，废了不少脑力，想到一半就放弃了，最终通过CSDN上的一篇博客，看了别人的推导，才弄明白了梯度下降向量化是如何演变出来的。我的失败主要在于，一会考虑θ的某一个元素的向量表示，一会儿又从θ向量的角度来考虑，最后乱七八糟，正确做法是先考虑其中的一个元素，比如说θ0，核心在于这个式子的转换，我们取i=0,1,2,...,m，把式子转成一个列向量的形式，发现这个式子可以转为g(X*θ)的形式，实际上我已经推到了这一步，就是没想到后面的元素xj(i)怎么与XT联系起来，然后就糊涂了，接下来的做法就是把后面的x0(i)展开成[x0(1),x0(2),x0(3),...,,x0(m)]的行向量形式，θ1就是对应[x1(1),x1(2),x1(3),...,,x1(m)]，以此类推，最终得到了XT, 这就是今天的收获。

### 2019/3/4 星期一 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 难得的晴天，让我先开心一下～今天只做了一件事，就是弄懂梯度下降，分成三块就是通过简单的一元二次函数，来理解梯度下降法是怎么工作的，它的”一步一步逼近极小值”中的“一步一步”怎么用数学式来表示，我目前还没弄明白为什么梯度的方向就是函数变化最快的方向，但是弄明白了梯度下降或者梯度上升为什么一个要减变化率，一个要加变化率，在这个demo中，我用 Δy<ε 作为梯度下降迭代过程停止的条件；第二块呢，也算是很重要的一件事，如何手推梯度下降每次迭代时参数θ的更新式，手稿图片也保存在本项目；第三块呢，就是将数学原理应用实践，跟刚才demo不同，我这里迭代停止的条件是达到了指定的迭代次数，另外呢，我的代码复杂度比较高，我是严格按照更新式的逻辑写的代码，但是书中好像是直接按照矩阵的方式，进行运算，复杂度比我低，而且最终结果不大一致，通过图像显示，我的算法最后得到的分界线，效果也不差。明天继续学习矩阵计算梯度下降。

### 2019/3/2 星期六 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 感悟：机器学习，无非就是用训练样本的特征矩阵+目标向量，找到一组参数w，来拟合二者的关系，前面学的朴素贝叶斯，现在看的逻辑回归，或者线性回归，神经网络之类的，都是想学参数w，区别在于参数w的用法不一样，线性回归，可能直接是作为权重，逻辑回归呢，就是在线性回归的基础上，加上一个sigmoid转换，神经网络呢，就更加复杂，用于神经元连接的权重。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  那我们如何计算w呢？换句话说，关于w的损失函数如何定义呢？理想的损失函数，应该是一个凸函数，这样我们就容易求解到全局最优解，而不是陷入到局部最优，因此我们定义损失函数时要考虑是否为凸函数。回归模型是一般采用最小二乘法，(预测值-真实值)的平方和，用梯度下降求到最小值，然后得到了w；对于分类模型，可以通过最大似然估计，定义一个类别y的概率公式P(y|x,w)，假设样本是独立同分布的，所有样本组成联合概率公式，即把P(y|x,w)累乘起来，在逻辑回归中，将累乘式取对数再取反，即为损失函数（原因在逻辑回归的笔记中），然后用梯度下降来求解。
 
### 2019/3/1 星期五 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 昨天的学习，是了解了朴素贝叶斯算法，怎么在一个简单的文本分类中应用的，但是我心里一直有疑问，朴素贝叶斯不是还可以应用在其他常见的数据集中吗？比如说，给了西瓜的各个属性，如何用朴素贝叶斯来判断是不是好瓜呢？于是今天主要做的任务就是了解朴素贝叶斯更通用的算法原理。首先根据贝叶斯公式，我们知道后验概率=先验概率×似然，先验概率的求法不多说，根据大数定律，用频率近似概率，某类别的总样本数/总样本数；而似然就有两种不同的求法了，频率学派推荐的方法是MLE，最大似然估计，假设样本之间是独立同分布的，贝叶斯学派的方法就是NB，朴素贝叶斯，假设样本各个属性之间是相互独立的，针对连续型特征和离散型特征有不同的计算方法。

### 2019/2/28 星期四 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 上午准备用Python来实现CART的算法的，《机器学习实战》上面的数据集划分，与前面所学的划分方式不太一致，加上自己偷懒，我就没有动手实现了......今天主要完成的任务是朴素贝叶斯算法的学习，了解了朴素贝叶斯算法的原理，主要就是用先验概率和数据来推算后验概率，它的假设是各个特征之间，是相互独立的，即一个特征的发生不会影响第个特征的发生。但是有点奇妙的是，在文本分类中，特征是词汇表中的每一个单词，而我们所知道，单词之间不是独立的，比如说’bacon’出现在‘delicious’附近的概率，大于出现在’unhealthy’附近的概率，这个就是’naive’的含义;另一个假设是特征相同重要，而实际上，文本分类任务中，少数关键词就可以表达文章的核心信息，尽管上述假设存在一些小瑕疵，但朴素贝叶斯的实际效果还是很好的。今天主要是动手实现了基于朴素贝叶斯算法的文本二分类模型，预测效果还不错，弄明白了朴素贝叶斯在文档分类中的是如何应用的。
 
  ### 2019/2/27 星期三 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 今天完成了一个重要内容，就是当下常用的决策树算法CART的理论知识学习，相比于前面所学习的ID3、C4.5算法，CART做了不小的改变。首先是数据集的信息量的度量发生了变化，CART采用Gini指数，对于样本只有两个类别时，Gini指数的计算式为2p(1-p)，相比于香农熵的对数运算，这里简化了不少的计算量；然后是CART既可以做分类也可以做回归，而前面提到的两种算法，只能完成分类任务；还有一个很关键的区别是，CART采用的二元分类，所有的非叶子节点，都有并且只有2个孩子，便于我们计算机的运算。
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 今天学习的重点是，CART为了防止过拟合，是如何进行剪枝操作的，这里主要设计到后剪枝策略，有点费脑，详细的学习笔记在2_classify_and_regression_tree.py文件中，明天计划是实现CART的Python代码，加强自己对算法的理解程度。
 
### 2019/2/26 星期二 

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在昨天的基础上，今天实现了用Python代码进行决策树的预测，预测也使用了递归的技巧，因为我们前面生成的决策树是一个字典结构，我们进行预测，那么预测结果肯定分布在叶子节点上，那么这个问题实际上就转化成了一个字典的遍历问题，如何获取到key所对应的value不再是一个dict类型而是一个准确的值，是这个函数的核心问题，我们用递归就可以解决，如果是字典，继续向下搜索。

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;决策树的预测实现完之后，做了一些简单的工作，把决策树即数据字典保存在硬盘上，使用了Python的序列化模块pickle。在最后，对于决策树算法，进行了一些总结，主要包括决策树算法的步骤、决策树ID3算法存在的弊端，以及基于这些弊端，如何在C4.5中一一进行改进，大致的思路记了下来，想通过案例来学习，可参看《机器学习》周志华：P83 4.4连续值与缺失值章节。
  
### 2019/2/25 星期一

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 今天主要是通过代码实现了决策树的算法过程，分三个阶段进行学习，首先是决策树的数学原理，考察的是香农熵的计算，通过香农熵我们可以知道如何度量数据集的信息量，然后通过信息增益来找到数据集的最佳分割特征，找到之后细分为几个数据子集，然后再次进行数据集分割，是一个递归的操作；第二个阶段就是用Python实现决策树的创建，最终想得到的是一个字典结构，键名即为判断节点，最底层的键值为类别名；第三个阶段就是测试算法：使用决策树执行分类。目前只完成了前两个阶段，而且构建决策树只用了ID3算法，还有C4.5,CART等算法尚未了解，明天继续。

### 2019/2/22 星期五 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 今天做的事情：在网站对比了几篇博客，刘建平老师写的KNN算法的总结很全面，主要是了解KD Tree的算法思想，学习了 如何构建kd树，如何搜索目标点的最近邻节点，如何预测目标点的类别 这三个部分，同时对比《统计学习方法》中实现，来加深理解，二者的构建kd树的方法不相同，刘的思路是对kd树的优化，由方差来决定节点的分割，后者是逐个特征作为分割节点，关于kd树的代码实现，我没有做这方面的工作，只想着了解其背后的原理；最后总结了一下如何缺点KNN算法的K值，以及KNN算法的优缺点。

### 2019/2/21 星期四 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;结合网上的博客和机器学习实战的代码，将昨天实现的knn代码应用在约会网站的预测上面。突然来了一个想法，这个代码主要是根据三个特征输入，预测结果为类别型（喜欢，一般，不喜欢）,这是knn用在分类任务上的应用场景，那如果用knn作回归呢，打个比方，也是这个项目，样本输入的标签不再是类别型，而是评分（0~9，分越高代表越喜欢），那么回归预测不就是直接可以将距离最近的几个点的值，求均值，或者加权值，那不就是knn在回归上的应用吗？明天再看看回归是不是这样子应用的。
  
### 2019/2/20 星期三
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;下定计划，准备开始复现经典的机器学习算法，打算一边看书《统计机器学习》,一边结合《机器学习实战》上的代码，来熟悉掌握算法的原理。今天只完成了knn算法的简单实现方式，主要是暴力法搜索，明天继续完成knn算法的实践。
