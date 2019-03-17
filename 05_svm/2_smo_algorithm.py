# encoding: utf-8
"""
 @project:ML_Algorithms
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 3/11/19 9:00 AM
 @desc: 本节内容主要是参考李航《统计学习方法》P124 7.4序列最小最优化算法，由于格式问题，本文公式看起来不太方便，可以对比书上的公式来看
"""
import numpy as np
import matplotlib.pyplot as plt

"""
    写在前面的话：
    
    我们知道，SMO算法是要求解一系列的αi值，在做这件事情之前，不妨先想一想，αi的值有什么特点呢？
    
    我们首先回忆一下αi的来源，它是拉格朗日乘子，后面跟着的(1-yi*(w.T*xi+b)),也就是说，每一个αi是和一个(xi,yi)唯一关联，这看似是一个简单
的知识点，但是我在后面忘记了这一点，导致很多地方理解起来非常困难。
    
    于是我们换个角度来思考，αi的值由对应的(xi,yi)来确定，与其他样本点是没关系的，还记得我在1_svm_introduction.py文件中讲到，如何根据αi
来确定样本点的位置，αi=0时，我们认为样本点 yi*(w.T*xi+b) >= 1 , 即样本点处于间隔边界上，其实更多的点分布在远离间隔边界的地方，那么这种点
对我们的分类决策函数没多大意义，因为我们想要的是真正的支持向量所在的点.

    这就给了我们几点启示：
        第一呢，分类决策函数，由那些支持向量所决定，而我们根据前面的学习也知道，这些支持向量对应的αi，需要满足 0< αi <= C
        第二呢，我们的数据集是十分庞大的，支持向量肯定也就少数那么几个，大多数点都处于远离间隔边界的地方，对应的αi自然也就是0
        
    综合一下，我们可以知道，对于一个存在噪音的训练数据集来说，最终SMO算法求解出来的αi的分布肯定是这样子的：
        (1)绝大多数αi的值为0 ， 
        (2)几个噪音的点(处于间隔边界和分类超平面之间、处于分类超平面上、处于超平民错误的一侧)，对应的αi的值为C
        (3)剩下的支持向量对应的αi，则处于(0,C)开区间范围内
"""

"""     
 1.SMO算法概述
    SMO算法是一种启发式的算法，其核心思路是：
    
    已知KKT条件是该最优化问题的充要条件，如果所有变量αi的解都满足此最优化问题的KKT条件，那么我们就认为这个最优化问题的解就是α向量了，如果
 不满足的话，那我们就要选取两个变量，如何选取也是有讲究的，然后固定其他变量，先对这两个变量进行优化.
 
    整个SMO算法理解完之后的感悟，所谓的启发式，就是一个整体求最优的问题，换成一小块一小块求最优的问题，因为整体求最优，涉及太多参数，计算困难，
 不妨化解为一个个的小问题，然后认为所有的小问题达到最优之后，即达到整体最优,也就是说，我们把每一对(αi,αj)都调整到满足KKT，最终所有的αi都满足
 KKT条件了，也就满足最优化问题的充要条件，发现这一环扣一环，真的巧妙!
        
 2.SMO算法的基本思想 ** 
 
    对于含有N个变量αi的目标函数，SMO采取的做法是每次只优化两个变量，将其他的变量都视为常数，由于sum αi*yi = 0这个KKT限制条件，我们将α3,α4,
 α5,...,αN固定，那么α1和α2的关系自然也就固定了，即我们可以用α1来表示α2，那最后优化目标函数，不就变成了一个单变量的凸优化问题，直接求导不就得
 了，计算出一个变量，另一个变量也随即算出，α1和α2的更新不就完成了吗?！
    
    为了表示方便表示，这里定义 Kij = np.dot(ϕ(xi),ϕ(xj))  # 用核函数来代替高维特征的内积
    
    由于α3,α4,α5,...,αN都成了常量，我们不妨将它们从目标函数中去除，由此以来，我们的优化目标函数变成了：
    
    min 1/2*K11*α1^2 + 1/2*K22*α2^2 + y1y2K12α1α2 - (α1+α2) + y1α1*sum(yi*αi*Ki1) + y2α2*sum(yi*αi*Ki2) ,其中i=3,4,...,N
    
    s.t. α1y1 + α2y2 = - sum(αi*yi) = ς   其中i=3,4,...,N  ， α1y1 + α2y2的值是固定的(因为后面的各项都是固定的)
         0 <= αi <= C    其中i = 1,2
         
    
 3.SMO算法目标函数的优化【重要】
    (1)根据上面的约束条件，我们可知：
                                α1y1 + α2y2 = ς  , 其中ς为一常量
                                0 <= αi <= C     ,  i = 1,2 ###  强调一下，SMO算法中只要求选出来的两个αi，满足0 <= αi <= C !!!
       而y1和y2只能取值1或者-1，那么α1和α2的函数关系只有四种情况：
       
                                (a) α1 + α2 = ς
                                (b) α1 + α2 = -ς
                                (c) α1 - α2 = ς
                                (d) α1 - α2 = -ς
       
       四种情况都受到 0 <= α1 <= C , 0 <= α2 <= C所约束
        
       也就是说，(α1,α2)组成的点只能在某条斜率为±1的线上移动，而这条线会被α1 = 0,α1 = C,α2 = 0,α2 = C这四条直线所截断(为了后面好描述，
    把这四条线段构成的约束条件暂且称为正方形边界)，也就说，最终α1和α2组成的点，实际上是在一条线段上移动，当然，因为有四种可能性的情况,α1与α2
    的函数图像则为四条不同的线段，详见α1α2_figure.png文件.
    
    (2)我们已知α1与α2的关系是固定的，那么两变量的优化问题实际上仅仅是一个变量的优化问题，不妨考虑对α2进行优化求解.
        
       假设上一轮迭代得到的解是α1_old,α2_old,假设沿着α2方向未经修剪的解为α2_new_unc,假设本轮迭代完成之后的解为α1_new,α2_new.
    
       由于α2_new必须也在那条线段上(要满足KKT条件)，设L和H为α2移动范围的上下界，α2_new也得在范围内，则有：
                                
                                L <= α2_new <= H
                                
       (a)当y1!=y2时，α1与α2所在直线的斜率为1，此时直线有两种情况(c)或(d)，由于α2必须严格满足大于等于最小值，所以这里的L要取两条直线与正方
    形边界交点处对应的α2中的较大的那个值，同理H则取两条直线与正方形边界交点处对应的α2中的较小的那个值，即：
                                L = max(0 , α2_old - α1_old )
                                H = min(C , C + α2_old - α1_old )
                                
       (b)当y1==y2时，α1与α2所在直线的斜率为-1，此时直线有两种情况(a)或(b)，由于α2必须严格满足大于等于最小值，所以这里的L要取两条直线与正
    方形边界交点处对应的α2中的较大的那个值，同理H则取两条直线与正方形边界交点处对应的α2中的较小的那个值，即： 
                                L = max(0 , α1_old + α2_old - C)
                                H = min(C , α1_old + α2_old)        
                                
    (3)假如我们求导得到的α2_new_unc，那么如何对它进行修剪得到最终的α2_new呢？ 具体如下：
    
                               H             ,  α2_new_unc > H
                   α2_new  =   α2_new_unc    ,  L <= α2_new_unc <= H
                               L             ,  α2_new_unc < L
    
    
    (4)现在的问题就在于如何求解α2_new_unc的值了，实际上我们只需要将目标函数对α2求偏导数即可
        
       为了简化描述，我们令g(x)为：
                                g(x) = w.T * ϕ(x) + b 
       
       由前面所学对w求偏导得， 
                                w = sum (αi*yi*xi)
                    
       综合可得，
                                g(x) = sum(αi*yi*K(x,xi) + b         ，i=1,2,...,N
       
       
       我们令预测值与真实值的误差函数为：    ### 我们为了后面方便起见，这里一次性算出所有样本点的误差Ei
                                E(i) = g(xi) - yi
                                     = w.T * ϕ(xi) + b  - yi 
                                     = sum(αj*yj*K(xi,xj) + b - yi   ，其中i,j=1,2,...,N
    
       由 α1y1 + α2y2 = ς , 并且yi^2 = 1，可以得到用α1表示α2的式子为：
        
                                α1 = y1(ς - α2y2)
            
       代入到上面的目标优化函数(第2节中min后面的函数)，得到一个只关于α2的式子，为W(α2)
       
       W(α2)是一个凸函数，即导数为0的时，对应的α2取得极小值 (知道这么一回事就行了，我也不知道为啥就是凸函数...)，于是W(α2)对α2求导，并且
       令其值为0，值为0时对应的α2值，即为α2_new_unc： 
           
                                α2_new_unc = α2_old + y2(E1-E2)/(K11+K22-2K12)   
                                (此处根据《统计学习方法》得到，计算过程较为复杂，知道求解过程即可)
            
       求出α2_new_unc的值之后，再根据前面的修剪关系式，得出α2_new的值
       
                                           H             ,  α2_new_unc > H
                                α2_new  =  α2_new_unc    ,  L <= α2_new_unc <= H
                                           L             ,  α2_new_unc < L
                                           
    (5)得出α2_new的值之后，我们再来计算α1_new的值
       已知α1_new*y1 + α2_new*y2 = α1_old*y1 + α2_old*y2，将α2_new*y2移到右边，再同时乘以y1，即得α1_new为：
       
                                α1_new = α1_old + y1y2(α2_old-α2_new)
       
       于是本轮中，我们得到最优化问题的两个变量的解为(α1_new,α2_new)
                                
 
 4.SMO算法两个变量的选择【重要的知识点】
    (1)关于第一个变量的选择
        SMO算法称选择第一个变量为外层循环，外层循环是在训练样本中选取违反KKT条件最严重的样本点，注意哦，要选取违反KKT条件最严重的样本点，
    并把该样本点对应的变量αi作为第一个变量!!!
        
        具体来说，检查样本点是否满足KKT条件，即：
                        当αi = 0 时， 须满足yi * g(xi) >= 1   ， 此时样本点在间隔边界上或者远离间隔边界的地方
                        当0<αi<C 时， 须满足yi * g(xi)  = 1   ， 此时样本点在间隔边界上
                        当αi = C 时， 须满足yi * g(xi) <= 1   ， 此时样本点在间隔边界和分类超平面之间，或者在超平面的另一侧(噪音点)
                        
        注：
          yi*g(xi)等于yi(w.T*xi+b),结合我在1_svm_introduction.py中，关于样本点位置的判断那一部分的讲解，即可得出上述点的位置的结论.
        
        也就是说，违反KKT条件的点，即对应不同的αi的取值时，yi * g(xi)的乘积没有满足后面的不等式，另外呢，根据网上的讲解，判断违反KKT条件的
    严重程度的依据是：
            (a)优先违反第二个条件的样本点，如果这样的点很多，那么任选一个即可
            (b)如果没有违反条件2的样本点的话，那么再从违反条件1或条件3的样本点中任选一个
        
    (2)关于第二个变量的选择
        SMO算法称选择第二个变量为内层循环，假设我们在外层已经找到了α1， 第二个α2的标准是让|E1 - E2|有足够大的变化!!!
        由于α1定了的时候，E1的值也就确定了，我们一开始不是早就整理好每个样本点的E值吗？所以这里就很方便了，想|E1 - E2|最大，只需要
            (a)在E1为正时，选择最小的Ei作为E2
            (b)在E1为负时，选择最大的Ei作为E2
        这里应该很好理解，正值减最小的再取绝对值，负值见最大的再取绝对值，都是为了找到最大的|E1 - E2|.
        
 5.计算阈值b和差值Ei
    经过第4节中讲解的变量的筛选，再加上我在前面提到的α1_new和α2_new的计算，在得到α1_new和α2_new的值之后，我们现在讨论如何更新b和Ei的值
    
    (1) 计算阈值b
        
        当 0 < α1_new < C时，说明α1_new仍符合KKT条件，由 sum αi*yi*Ki1 + b1 = y1 (中i = 1,2,...,N) , 可知b1_new为:
        
                    b1_new = y1 - sum αi*yi*Ki1 - α1_new*y1*K11 - α2_new*y2*K21  , 这里的 i = 3,4,...,N
        
                    (因为α3,α4,...,αN还是固定没变的，而α1和α2的值改变了，所以这两项要单独拿出来)
                    
        由E1的定义式，可知E1为：
        
                    E1 = g(x1) - y1 = sum αi*yi*Ki1 + a1_old*y1*k11 + a2_old*y2*k21 + b_old - y1 ,其中i=3,4,...,N
                    
        其中，上面的两个式子都出现了(y1 - sum αi*yi*Ki1)，于是用E1来表示b1_new，有:
        
                    b1_new = -E1 - y1K11(α1_new-α1_old) - y2K21(α2_new-α2_old) + b_old
    
        同理，当 0 < α2_new < C时，说明α2_new也符合KKT条件我们有:
    
                    b2_new = -E2 - y1K11(α1_new-α1_old) - y2K21(α2_new-α2_old) + b_old
        
        在上个文件中提过，理论上来说，无论是通过α1_new对应的支持向量(x1,y1)计算出的b1_new，还是通过α2_new对应的支持向量(x2,y2)计算出的
        b2_new，二者是相等的，即 b1_new = b2_new，但是这里还是进行求均值的操作.
        
        即最终的b_new为：
    
                    b_new = (b1_new + b2_new) / 2
    
    (2) 得到了b_new之后，我们再更新对应的Ei值       注意，Ei值的更新需要所有支持向量对应的αj !!!
                    
                    Ei = sum αj*yj*K(xi,xj) + b_new - yi  ,其中j是αj的下标，不是全集，xj是所有支持向量的集合，yj对应支持向量的下标
        
  【疑问】为什么这里更新的Ei不是用全集，而是用支持向量的集合呢?
        
        回顾一下，w与αi,xi,yi的关系式为:
            
                    w = sum(αi*yi*xi)   i = 1,2,3,...,N
                    
        我们初始化的α是一个全为0的向量，即α1=α2=α3=...=αN=0，w的值即为0.
        
        我们进行SMO算法时，每轮挑选出两个变量αi，固定其他的α值，也就是说那些从来没有被挑选出来过的α，值始终为0，而根据前面所学，支持向量对应
    的αi一定是满足 0<αi<=C的.
    
        有了这个认识之后，为什么不用全集呢，因为不是支持向量的样本点，对应的αi值为0啊，加起来也没有意义，对w产生不了影响，只有支持向量对应的点
    (xi,yi)与对应的αi相乘，产生的值才对w有影响啊。
    
        从这里也能理解，为什么李航书中，认为支持向量不仅仅是处于间隔边界上的点，还包括那些处于间隔边界和分类超平面之间，分类超平面之上，分类超
    平面错误的一侧处的点了，因为后面所说的那些点，对应的αi为C，对w的计算可以起到影响作用，换句话说，就是能对w起到影响作用的点，都属于支持向量!
        
    
        
 6.SMO算法总结（将前面所有的内容梳理一下，总结SMO算法的整个过程##）
    
    输入：N个样本(x1,y1),(x2,y2),...,(xN,yN), 精度ε
    输出：α向量的近似解
    
    (1)从第0轮开始迭代，n_iter初始为0，α向量也初始为零向量
    
    (2)按照第4小节中的介绍，选取第一个变量αi和第二个变量αj
    
    (3)用αi表示αj,求出αj_new_unc
    
    (4)根据第3小节中(3)中讲的修剪的方法，将αj_new_unc修剪为αj_new
    
    (5)根据αj_new求出αi_new
    
    (6)根据αi_new和αj_new，求出bi_new , bj_new
    
    (7)根据bi_new和bj_new，求出b_new
    
    (8)根据b_new更新Ei , 其中i=1,2,3,..,N
    
    (9)如果某一轮αi迭代更新的变化(|αi_new−αi_old|)小于精度ε，是的话，就认为找到αi的最优解了，并检查是否满足如下的终止条件：
                                
                                sum αi*yi = 0    
                                0 <= αi <= C     
                                yi * g(xi) >= 1  ， 当对应的αi满足αi=0时
                                yi * g(xi)  = 1  ， 当对应的αi满足0<αi<C时
                                yi * g(xi) <= 1  ， 当对应的αi满足αi=C时
                                                    
       这里是要对所有的样本点和其对应的αi进行审查哦 ～
    
    (10)如果在(9)中满足小于精度ε，并且还满足终止条件，那么迭代停止，返回最终的α (这是所有需要更新的αi都更新完之后的α向量)
        如果在(9)中没有满足小于精度ε，又或者说满足了ε但是没满足终止条件，那就要再次回到第(2)步咯.
        
 7. SMO simple版本的实现
    
    下面给出《机器学习实战》中，simple_smo的实现算法，与上面的核心思想大致相同，细节地方略有偏差，但是影响也不大啦~
"""


def load_data_set(file_name):
    X = []
    y = []
    fr = open(file_name)
    for line in fr.readlines():
        line = line.strip().split('\t')
        X.append([float(line[0]), float(line[1])])
        y.append(float(line[2]))
    return X, y


# i是第一个alpha的下标，m是所有alpha的数目，只要函数值不等于输入值i，函数就会进行随机选择
# 简单版本的SMO算法，是随机选取不同的两个alpha，不同于我们上面提到的|Ei - Ej|最大
def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


# 修剪α_new_unc的值 , 这个对应第3小节中(3)部分的解释
def clip_alpha(alpha_j, h, l):
    if alpha_j > h:
        alpha_j = h
    if l > alpha_j:
        alpha_j = l
    return alpha_j


# 简化版的SMO函数的伪代码:                    【十分言简意赅的伪代码，可以看完整个代码之后，再来理解一遍这里的伪代码】

# 1.创建一个alpha向量并且将其初始化为全0向量
# 2.当迭代次数小于最大迭代次数时(外循环):
#      对数据集中的每个数据向量(内循环):
#          如果该数据向量可以被优化:
#             随机选择另外一个数据向量
#             同时优化这两个向量
#             如果这两个向量都不需要被优化，遍历下一个数据向量
#      如果所有向量都不需要被优化，增加迭代数目，继续下一次循环
# 输入参数：数据集、类别标签、常数c、容错率、最大的循环迭代次数

def smo_simple(data_mat, class_labels, C, tolerance, max_iter):
    X = np.mat(data_mat)  # X为所有样本点的集合{x1,x2,x3,...,xN}
    y = np.mat(class_labels).transpose()  # y为所有样本点对应的标签的集合
    b = 0  # 初始化b为0
    m, n = np.shape(X)
    alphas = np.mat(np.zeros((m, 1)))
    n_iter = 0  # n_iter只有在一轮内循环中，所有的αi变量都没有更新时，才会自增1
    # 当所有的αi变量连续达到max_iter轮无任何更新的话，认为已经达到了收敛条件!
    while n_iter < max_iter:
        alpha_pairs_changed = 0  # 用于记录alpha是否已经进行优化
        print('n_iter:', n_iter)
        for i in range(m):
            g_xi = float(np.multiply(alphas, y).T * (X * X[i, :].T)) + b  # 计算现有的模型对于样本xi的预测值
            e_i = g_xi - float(y[i])  # Ei = g(xi) - yi
            # y[i]*Ei < -tolerance 则需要alphas[i]增大，但是不能>=C，参考第6节(9)中终止条件的要求
            # y[i]*Ei >  tolerance 则需要alphas[i]减小，但是不能<=0
            if ((y[i] * e_i < -tolerance) and (alphas[i] < C)) or (
                    (y[i] * e_i > tolerance) and (alphas[i] > 0)):
                j = select_j_rand(i, m)  # 随机选择另一个与alpha[i]成对优化的alpha[j]，这里是简化版的，所有随机选择
                # 计算误差Ej
                g_xj = float(np.multiply(alphas, y).T * (X * X[j, :].T)) + b
                e_j = g_xj - float(y[j])
                # 保存更新前的alpha值，使用深拷贝
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # 计算alpha的线段边界L和H，保证alpha更新之后，在(0,C)范围内
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue  # 第一个跳出条件，L==H，说明alpha[i]没有移动的范围了，跳出，遍历下一个alpha进行更新
                # 计算eta，eta就是上面第3节(4)部分提到的(K11+K22-2K12)，用于后面计算αj_new_unc
                eta = X[i, :] * X[i, :].T + X[j, :] * X[j, :].T - 2.0 * X[i, :] * X[j, :].T
                if eta == 0:
                    print('eta == 0')
                    continue  # 第二个跳出条件，eta之后要作为分母，分母不能为0
                # 更新alpha[j]，这里的alpha[j]是刚刚求导，导数为0时计算出来的，未经过修剪
                alphas[j] += y[j] * (e_i - e_j) / eta
                # 修剪alpha[j]
                alphas[j] = clip_alpha(alphas[j], H, L)
                # 这里的0.00001就是我们上面提到的精度ε
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('alpha[j] 已经达到精度要求啦! ')
                    continue  # 第三个跳出条件，认为alpha_j已达到可接受范围内的最优
                # 根据alpha[j]更新alpha[i]
                alphas[i] += y[j] * y[i] * (alpha_j_old - alphas[j])
                # 步骤7：更新b1和b2
                b1 = b - e_i - y[i] * (alphas[i] - alpha_i_old) * \
                     X[i, :] * X[i, :].T - \
                     y[j] * (alphas[j] - alpha_j_old) * \
                     X[i, :] * X[j, :].T
                b2 = b - e_j - y[i] * (alphas[i] - alpha_i_old) * \
                     X[i, :] * X[j, :].T - \
                     y[j] * (alphas[j] - alpha_j_old) * \
                     X[j, :] * X[j, :].T
                #  b的更新选择，之前也提过了
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数，如果运行到这里，说明前面几个continue都没有跳出去，alpha[i]和alpha[j]完成了一次更新
                alpha_pairs_changed += 1
                print('n_iter:%d i:%d pairs changed %d' % (n_iter, i, alpha_pairs_changed))
        # 只有在内循环未对任何一对alpha做修改时，n_iter+1；否则我们让n_iter回到0，继续内循环；
        # 只有当内循环未修改任一alpha对，且连续maxIter次迭代，才会结束(以保证所有alpha得到了充分的修改)
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
    print(w_, b_)
    plot_best_fit(data, labels, w_, b_)
