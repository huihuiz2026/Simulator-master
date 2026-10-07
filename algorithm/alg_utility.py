# ===================== 算法侧公共工具模块 =====================
# 供 IDQN / cDQN / cA2C 三个训练算法（均 `from .alg_utility import *`）复用，包含：
#   1) 策略概率分布类 Pd / DiagGaussianPd（改自 OpenAI baselines）；
#   2) 数值工具：奖励标准化 normalize_reward、连续二次背包求解 continuous_quadratic_knapsack
#      及其包装 projection_fast（把调度总量压到实际空闲司机数）；
#   3) 神经网络组件：全连接层 fc / dense、正交初始化器 ortho_init；
#   4) DQN 调度辅助函数：有效网格编号提取、对向冲突统计、状态/奖励抽取、Q 表求和等。
# 注：部分组件（对角高斯分布、分组采样等）在当前仓库内暂无调用方，属 baselines 移植的通用件。
from .tf_compat import tf  # 导入 TensorFlow：经本包 tf_compat 包装的 TF1 兼容 API（placeholder/session 图计算风格）
import numpy as np  # 导入 NumPy 数值计算库并简写为 np：数组与矩阵运算
# （已注释的导入）cvxpy 凸优化库：原本供下方整段注释掉的 projection 投影函数使用
# import cvxpy as cvx
from simulator.utilities import *  # 导入仿真器工具模块的全部公开名字：本文件用到 ids_2dto1d（二维坐标转一维编号）等

# （原文：以下部分代码改写自 OpenAI 的 baselines 开源项目）
""" Some of Following codes are modified from https://github.com/openai/baselines
"""
# 求和辅助函数：把轴号统一包装成列表后调用 tf.reduce_sum
def tfsum(x, axis=None, keepdims=False):
    # 参数：x 待求和的张量；axis 求和的轴（None 表示全部元素相加）；keepdims 是否保留被约简的维度
    axis = None if axis is None else [axis]  # None 保持 None（全局求和），否则把单个轴号包装成单元素列表
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)  # 按指定轴（或全局）求和并返回结果

# 概率分布抽象基类（改自 OpenAI baselines）：为"策略网络输出的动作分布"规定统一接口，
# 子类需实现取众数、负对数概率、KL 散度、熵、采样等方法
class Pd(object):
    # （原文 docstring：一种特定的概率分布）
    """
    A particular probability distribution
    """

    # 取分布的众数（概率最大的取值；对连续分布即密度峰值处）
    def mode(self):
        raise NotImplementedError  # 抛出"未实现"异常：抽象方法，由子类实现

    # 计算负对数概率 -log P(x)（对数似然取负，常直接作为损失项来最小化）
    def neglogp(self, x):
        # （原文：通常先定义负对数概率会更容易）
        # Usually it's easier to define the negative logprob
        raise NotImplementedError  # 抛出"未实现"异常：抽象方法，由子类实现

    # 计算与另一分布 other 的 KL 散度（衡量两个概率分布差异程度的指标）
    def kl(self, other):
        raise NotImplementedError  # 抛出"未实现"异常：抽象方法，由子类实现

    # 计算分布的熵（不确定性的度量：分布越均匀熵越大）
    def entropy(self):
        raise NotImplementedError  # 抛出"未实现"异常：抽象方法，由子类实现

    # 从该分布中随机抽取一个样本值
    def sample(self):
        raise NotImplementedError  # 抛出"未实现"异常：由子类实现

    # 计算对数概率 log P(x)：直接由子类实现的 neglogp 取负得到
    def logp(self, x):
        return - self.neglogp(x)  # logp(x) = -neglogp(x)


# 对角高斯分布（各维度相互独立的正态分布）：连续动作策略的输出分布，
# 参数为均值向量 mu 与对数标准差向量 logstd（存对数可保证标准差恒为正）
class DiagGaussianPd(Pd):
    # 构造函数：保存分布参数并换算出标准差
    def __init__(self, mu, logstd):
        self.mean = mu  # 保存均值向量
        self.logstd = logstd  # 保存对数标准差向量
        self.std = tf.exp(logstd)  # 标准差 = exp(logstd)，指数化后必为正数

    # 取分布众数：高斯分布的众数就是均值
    def mode(self):
        return self.mean  # 返回均值向量

    # 计算多维独立高斯下 x 的负对数概率（三项之和，公式见下）
    def neglogp(self, x):
        # （原文：axis=-1 沿最后一维求和，第一维是 batch 大小）
        # axis = -1, sum over last dimension, first dimension is batch size
        # 公式：0.5*Σ((x-mean)/std)² + 0.5*ln(2π)*d + Σlogstd，其中 d 为维度数（x 最后一维的长度）
        return 0.5 * tfsum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tfsum(self.logstd, axis=-1)  # 三项之和：标准化偏差平方项 / 正态归一化常数项（维度数 d 转 float32）/ 各维 logstd 之和

    # 随机采样：均值 + 标准差×标准正态噪声（重参数化写法，梯度可穿过采样传回网络参数）
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))  # mean + std*ε，ε 为与 mean 同形状的逐元素标准正态随机数



# 奖励标准化：把整条（折扣）奖励序列调整为均值 0、标准差 1，消除量纲与整体偏移（策略梯度训练的稳定技巧）
def normalize_reward(discounted_epr):
    # 参数 discounted_epr：一条轨迹按折扣因子 γ 累计得到的奖励序列（一维数组）
    reward_mean = np.mean(discounted_epr)  # 求奖励序列的均值
    reward_std  = np.std(discounted_epr)  # 求奖励序列的标准差
    discounted_epr = (discounted_epr - reward_mean)/reward_std  # 减均值、除以标准差，完成标准化
    return discounted_epr  # 返回标准化后的奖励序列

# （以下为整段注释掉的 projection 投影函数：用 cvxpy 凸优化把向量 Y 投影到
#  "各分量非负、总和恰为空闲司机数"的约束集上，后被下方二次背包的快速解法替代）
# def projection(Y, n, num_idle_driver):
#     assert np.sum(Y) > num_idle_driver
#     X = cvx.Variable(n)
#     objective = cvx.Minimize(cvx.sum_squares(X - Y))
#     constraints = [0 <= X,
#                    num_idle_driver == cvx.sum_entries(X)]
#     prob = cvx.Problem(objective, constraints)
#
#     # The optimal objective is returned by prob.solve().
#     result = prob.solve()
#     return X.value


# 求解连续二次背包问题：min 1/2*||x||²  s.t. bᵀx = r, 0≤x≤u（要求 b>0）；
# 最优解有解析形式 x_i = clip(-t·b_i, 0, u_i)，故对拉格朗日乘子 t 在断点集合上二分搜索（Kiwiel 2007）
def continuous_quadratic_knapsack(b, u, r):
    # 参数：b 等式约束的系数向量；u 各分量的上界向量；r 等式约束的右端项（目标总量）
    """
    OBJECTIVE
     min 1/2*||x||_2^2
     s.t. b'*x = r, 0<= x <= u,  b > 0
    
    Related paper
     [1] KC. Kiwiel. On linear-time algorithms for the continuous 
         quadratic knapsack problem, Journal of Optimization Theory 
         and Applications, 2007
    Coding Reference: 
    https://github.com/jiayuzhou/MALSAR/blob/master/MALSAR/functions/CMTL/bsa_ihb.m
    """
    # 迭代解法：令 g(t) = bᵀx(t)，g 随 t 增大单调不增；反复取候选断点的中位数试算 g(t)，
    # 与 r 比较后淘汰一半候选，直到精确命中 r 或候选耗尽（后者用两端点线性插值收尾）
    n = len(b)  # 问题维度（向量分量个数）
    # 精确命中标记：1 表示某轮中位数处 g(t) 恰等于 r，已提前跳出循环
    break_flag = 0 
    # 两组"断点"：t_l=0 是分量 x_i 降到 0 的临界 t；t_u=-u/b 是分量 x_i 顶到上界 u 的临界 t
    t_l = np.zeros(n)  # 下断点数组：全 0（t≥0 时对应分量 x_i 已归零）
    # 上断点数组：-u/b，系数 b=1 时即 -u（原行末英文注释 "( 0 - u)/1" 表达的即此）
    t_u = -u    # ( 0 - u)/1
    t_L = -float('Inf')  # 乘子 t 的搜索区间下界：初始为负无穷，随二分逐步收紧
    t_U = float('Inf')  # 乘子 t 的搜索区间上界：初始为正无穷
    g_tL = 0  # 下界 t_L 处的约束函数值 g（供最后的线性插值使用）
    g_tU = 0  # 上界 t_U 处的约束函数值 g（供最后的线性插值使用）
    T = np.concatenate((t_l, t_u), axis=0)  # 把两组断点拼成一维数组，作为二分搜索的候选集
    n_iter = 0  # 迭代轮数计数器
    while len(T) !=0:  # 主循环：候选断点集非空就继续二分（每轮约淘汰一半候选）
        n_iter += 1  # 迭代轮数 +1
        g_t = 0  # 当前试探点处的 g(t) = Σ b_i·x_i(t)，从 0 开始累加
        t_hat = np.median(T)  # 取候选集的中位数作为本轮试探点 t_hat

        # 按各分量 x_i(t_hat) 所处的"三段"位置归类：顶上界 u / 线性中间段 / 已归零
        U_inds = np.where(t_hat < t_u)  # "顶上界"分量下标：t_hat < t_u[i]，此时 x_i = u_i
        M      = np.where((t_u <= t_hat) & (t_hat <= t_l))  # "中间段"分量下标：此时 x_i = -t_hat·b_i（介于 0 与 u 之间）

        # 若存在"顶上界"的分量
        if len(U_inds[0]) != 0: 
            g_t = g_t  + np.dot(b[U_inds], u[U_inds])  # 顶上界分量各贡献 b_i*u_i，点积后累加进 g(t)

        if len(M[0]) != 0:  # 若存在"中间段"的分量
            # 中间段分量贡献 b_i*(-t_hat·b_i)，即 -t_hat·b_i²（对应原行末注释公式中 a=0 的情形）
            g_t = g_t - np.dot(b[M], t_hat*b[M])  # a = 0   np.sum(b(M).*(a(M) - t_hat*b(M)))
        if g_t > r:  # g(t_hat) 偏大：g 随 t 递减，故真正的 t* 在更大处，向右半区收缩
            t_L = t_hat  # 收紧区间下界为当前试探点
            T = T[np.where(T > t_hat)]  # 候选集只保留大于 t_hat 的断点
            # 记录新下界处的 g 值（供最后线性插值用）
            g_tL = g_t 
        elif g_t < r:  # g(t_hat) 偏小：真正的 t* 在更小处，向左半区收缩
            t_U = t_hat  # 收紧区间上界为当前试探点
            T = T[np.where(T < t_hat)]  # 候选集只保留小于 t_hat 的断点
            g_tU = g_t  # 记录新上界处的 g 值
        else:  # g(t_hat) 恰好等于 r：试探点本身就是精确解
            t_star = t_hat  # 直接取 t_hat 作为最优乘子 t*
            break_flag = 1  # 置"精确命中"标记
            break  # 跳出主循环
            
    if break_flag == 0:  # 候选耗尽仍未精确命中：利用区间两端点的信息线性插值逼近 t*
         t_star = t_L - (g_tL -r)*(t_U - t_L)/(g_tU - g_tL)  # 过 (t_L,g_tL)、(t_U,g_tU) 两点的直线与水平线 g=r 的交点
    x_star = np.minimum(np.maximum(0, -t_star*b), u)  # 代回解析解：x_i = min(max(0, -t*·b_i), u_i)，先截非负、再截上界 u
    return x_star  # 返回最优解向量 x*


# 快速投影（cvxpy 版 projection 的替代）：把"各网格期望调度司机数向量 u"以最小的改动削减，
# 使总和恰好等于实际空闲司机数 num_idle_driver（预期 Σu > 空闲司机数，即供不应求时往下压）
def projection_fast(u, n, num_idle_driver):
    # 参数：u 期望调度数向量；n 维度（网格数）；num_idle_driver 实际可用的空闲司机总数
    b = np.ones((n))  # 系数向量全取 1：等式约束简化为"各分量之和 = r"
    r = np.sum(u) - num_idle_driver  # 右端项 r = 期望调度总数 - 实际空闲司机数，即需要削减的总量
    x_star = continuous_quadratic_knapsack(b, u, r)  # 解二次背包：求范数最小的削减向量 x*（满足 0≤x*≤u 且 Σx* = r）
    return u - x_star  # u 减去削减量：总和恰为 num_idle_driver，各分量非负且不超过原值


# 分组多类别采样：对 d 组 logits，各自按 exp(logits) 的概率逐行抽取 1 个类别下标
def categorical_sample_split(logits, d=6):
    # 参数：logits 未经 softmax 的得分（采样概率正比于 exp(logits)）；d 分组数（默认 6，对应 6 个邻居方向）
    """
    :param logits: sampling according to the probability exp(logits)
    :param d: first dimension of logits. 6 in our case.
    :return:
    """

    value = [tf.multinomial(logits[i] - tf.reduce_max(logits[i], [1], keepdims=True), 1)  # tf.multinomial 按行依概率抽 1 个类别；减去每行最大值仅为数值稳定（不改变概率）
             for i in np.arange(d)  # 遍历 d 组 logits
             ]
    return value  # 返回长度 d 的列表，每个元素是对应组的采样结果张量

# 构建全连接层（fully-connected layer）：h = act(x·W + b)，供 cDQN / cA2C 的价值与策略网络使用
def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    # 参数：x 输入张量 [batch, 输入特征维]；scope 变量作用域名（TF1 按名字隔离/复用变量的机制）；
    # nh 该层神经元个数（输出维数）；act 激活函数（默认 ReLU）；init_scale 正交初始化缩放系数（仅被下方注释代码引用，未实际生效）
    with tf.variable_scope(scope):  # 进入变量作用域：本层变量的名字自动带上 scope 前缀
        nin = x.shape.as_list()[1]  # 从输入的静态形状取特征维数（第 0 维是 batch 大小）
        # （已注释）正交初始化版的权重定义；实际改用下面一行的 Glorot 均匀初始化
        # w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        w = tf.get_variable("w", [nin, nh], initializer=tf.glorot_uniform_initializer(seed=0))  # 权重矩阵 W，形状 [nin, nh]；Glorot 均匀初始化并固定种子 0（可复现）
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))  # 偏置向量 b（长度 nh），全 0 初始化
        z = tf.matmul(x, w)+b  # 线性变换 z = x·W + b
        h = act(z)  # 过激活函数（默认 ReLU：负值归零、正值保留），为网络引入非线性
        return h  # 返回该层输出张量 h


# 全连接层的另一套实现：与 fc 功能等价（IDQN 的 Q 网络使用），不依赖 Keras 2
def dense(x, units, activation, name):
    """Equivalent of the old tf.layers.dense without a Keras 2 dependency."""
    # （原文 docstring：不依赖 Keras 2 的旧版 tf.layers.dense 的等价实现）
    # 参数：x 输入张量（特征维取最后一维）；units 输出神经元个数；activation 激活函数；name 变量作用域前缀
    with tf.variable_scope(name):  # 进入以 name 命名的变量作用域
        kernel = tf.get_variable(  # 创建（或按名复用）权重矩阵 kernel
            "kernel", [x.shape.as_list()[-1], units],  # 形状 [输入特征维, units]，特征维取输入张量的最后一维
            initializer=tf.glorot_uniform_initializer())  # Glorot（Xavier）均匀初始化
        bias = tf.get_variable("bias", [units], initializer=tf.zeros_initializer())  # 偏置向量（长度 units），全 0 初始化
        return activation(tf.matmul(x, kernel) + bias)  # 返回 activation(x·kernel + bias)


# 正交初始化器工厂（Lasagne 风格，改自 baselines）：返回符合 TF 初始化器接口的函数，
# 用随机矩阵的 SVD 分解产生近似正交的权重，使初始各维方差一致、训练更稳定
def ortho_init(scale=1.0):
    # 参数 scale：生成的正交矩阵的缩放系数
    def _ortho_init(shape, dtype, partition_info=None):  # 内层初始化函数：按 TF 初始化器统一接口，依 shape 生成权重初始值
        # （原文：TF 版的 lasagne 正交初始化）
        #lasagne ortho init for tf
        shape = tuple(shape)  # 形状列表转为元组，便于后续判断与重塑
        if len(shape) == 2:  # 二维（普通全连接层的权重矩阵）
            flat_shape = shape  # 展平后的形状即原形状
        # 四维（卷积核）：把前三维（空间与输入通道）的乘积当行数、末维当列数
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])  # 展平成二维 (空间位置×输入通道, 输出通道数)
        else:  # 其他维度数不支持
            raise NotImplementedError  # 抛出"未实现"异常
        np.random.seed(1)  # 固定 NumPy 随机种子 1：保证初始化结果可复现
        a = np.random.normal(0.0, 1.0, flat_shape)  # 生成标准正态分布的随机矩阵 a
        u, _, v = np.linalg.svd(a, full_matrices=False)  # 奇异值分解（SVD）：所得 u、v 的列向量两两正交
        # 从 u、v 中挑选形状与目标一致者作为正交矩阵（SVD 只保证其中一个形状吻合）
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)  # 重塑回目标形状 shape
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)  # 截取到目标行列数、乘缩放系数 scale，并转为 float32 返回
    return _ortho_init  # 工厂函数返回内层初始化函数（闭包捕获 scale）


##############################################################
### utility function for tune DQN
##############################################################

# （原文小节标题：用于调 DQN 的工具函数）
# 从地图矩阵提取"有效网格"的行优先一维编号列表：矩阵中数值 >= 0 的格子视为有效格
def get_target_ids_local(mapped_matrix_int_local):
    # 参数 mapped_matrix_int_local：地图矩阵（正数为有效网格编号，负数如 -100 为无效格）
    row_inds, col_inds = np.where(mapped_matrix_int_local >= 0)  # 找出全部有效格的行下标数组与列下标数组（两数组逐位对应）

    M_local, N_local = mapped_matrix_int_local.shape  # 地图的行数 M_local、列数 N_local
    # 结果列表：收集有效网格的一维编号（原行末注释说明编号从 0 开始）
    target_ids_local  = []  # start from 0.
    for x, y in zip(row_inds, col_inds):  # 逐个遍历有效格的 (行 x, 列 y) 坐标
        node_id = ids_2dto1d(x, y, M_local, N_local)  # 工具函数：把二维坐标按行优先转成一维编号 x*N + y
        target_ids_local.append(node_id)  # 把编号追加进结果列表
    return target_ids_local  # 返回全部有效网格的一维编号列表

# 统计"对向冲突"的调度动作对数：若 A→B 与 B→A 两个方向的调度同时存在，记一次冲突
# （两网格互相对发司机、运力互相抵消；run_*.py 用它监控算法产出动作的质量）
def collision_action(action_tuple):
    # 参数 action_tuple：调度动作列表，每项形如 (起点网格编号, 终点网格编号, 调度司机数)
    count = 0  # 冲突计数器
    action_set = set(())  # 已出现动作的集合，元素为 "起点-终点" 字符串键
    for item in action_tuple:  # 遍历每个调度动作
        if item[1] == -1:  # 终点为 -1：表示原地不动/占位动作
            continue  # 跳过，不计入冲突
        grid_id_key = str(item[0]) + "-" + str(item[1])  # 当前动作的键："起点-终点"
        action_set.add(grid_id_key)  # 记入已出现集合
        conflict_id_key = str(item[1]) + "-" + str(item[0])  # 反向动作的键："终点-起点"
        if conflict_id_key in action_set:  # 若反向动作此前已出现，说明构成对向冲突
            count += 1  # 冲突数 +1
    return count  # 返回冲突（对向调度对）总数

# 建立"网格编号 <-> 节点编号"的双向映射字典
def construct_grid_nodeid_mapping(target_ids_local, grid_ids_local):
    # 参数：target_ids_local 节点编号列表（行优先一维编号）；grid_ids_local 与之逐位对应的网格编号列表
    node_mapping = {}  # 字典 1：网格编号 -> 节点编号
    # 字典 2：节点编号 -> 网格编号（与字典 1 互为反向；原行末的 # 是遗留空注释）
    grid_mapping = {} #
    for nodeid, gridid in zip(target_ids_local, grid_ids_local):  # 按位置配对遍历两组编号
        node_mapping[gridid] = nodeid  # 记录 网格编号 -> 节点编号
        grid_mapping[nodeid] = gridid  # 记录 节点编号 -> 网格编号
    return node_mapping, grid_mapping  # 同时返回两个映射字典

# 按有效网格下标，从（展平的）状态向量中抽取对应分量
def utility_conver_states(curr_s, target_id_states):
    # 参数：curr_s 展平后的状态向量（前段为空闲司机分布、后段为订单分布）；target_id_states 有效网格下标列表
    curr_s_new = [curr_s[idx] for idx in target_id_states]  # 列表推导：只保留有效网格下标处的分量
    return np.array(curr_s_new)  # 转成 NumPy 数组返回

# 按有效网格下标，从逐节点奖励向量中抽取对应分量（与上面的状态版本同构）
def utility_conver_reward(reward_node, target_id_states):
    # 参数：reward_node 逐节点（全网格）的奖励向量；target_id_states 有效网格下标列表
    reward_node_new = [reward_node[idx] for idx in target_id_states]  # 列表推导：只保留有效网格的奖励
    return np.array(reward_node_new)  # 转成 NumPy 数组返回

##############################################################


# 把嵌套 Q 表中的所有数值全部加总（Q 表：记录各"状态-动作"对估计长期回报的表）
def compute_sum_qtable(temp_qtable):
    # 参数 temp_qtable：嵌套列表结构的 Q 表（元素可能还是数组）
    temp_q = 0  # 累加器
    for item in temp_qtable:  # 遍历外层条目
        for jj in item:  # 遍历条目内的每个元素
            temp_q += np.sum(jj)  # 元素若为数组则先求和，再累加进总计

    return temp_q  # 返回所有 Q 值的总和
