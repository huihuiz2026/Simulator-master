

# 导入 NumPy 数值计算库并简写为 np：Python 中数组与矩阵运算的基础库
import numpy as np
# 通过兼容层导入 TensorFlow：tf_compat 内部使用 tensorflow.compat.v1 并关闭 TF2 行为，
# 使本文件 TF1 的 session（会话）风格代码可以在 TensorFlow 2 环境下运行
from .tf_compat import tf
# 导入标准库 random（回放池随机采样/随机覆盖）和 os（创建日志目录）
import random, os
# 导入本包的算法公共工具（如 fc 全连接层构造函数）；* 表示导入全部公开名称
from .alg_utility import *

# ================== cDQN（contextual DQN，上下文深度 Q 网络）==================
# 背景知识：DQN（Deep Q-Network）用深度神经网络近似 Q 函数——评估“在某状态
# 采取某动作”的长期累计回报；常配合经验回放池（见本文件 ReplayMemory）、
# target network 目标网络（见本文件 ModelParametersCopier）与 ε-greedy 探索策略使用。
# cDQN 针对多智能体“信用分配”问题的核心改动：全城只训练一个价值网络，
# 为每个网格输出一个价值；“把空闲司机派往邻居网格 k”这一动作的价值直接用
# 邻居 k 的价值来近似，学习目标也取动作指向网格的奖励与下一时刻价值来构造
# （形如 Bellman 目标 r + γ·V(s')）——即把邻居信息（上下文）引入训练，
# 缓解“自己的调度会影响邻居收益、只看自己奖励学不好”的问题。
# 原注释含义：这个算法本质上更接近 deep expected SARSA
# （SARSA 按实际执行的动作更新价值；expected SARSA 按动作概率加权构造目标）
# this is essentially deep expected SARSA.
# 价值网络估计器：搭建并训练“每个网格输出一个价值”的神经网络，
# 提供动作选择、学习目标计算与参数更新能力（与 IDQN.Estimator 结构对应，
# 但网络输出与目标构造方式不同——见各方法注释）
class Estimator:
    """ build value network
    """
    # 初始化：保存环境信息、搭建网络、准备 TensorBoard 日志，
    # 并预计算每个网格的邻居关系表与合法动作掩码
    def __init__(self,
                 sess,  # TF1 会话对象：所有计算图操作都要通过 sess.run 执行
                 action_dim,  # 动作维度：6 个邻居方向 + 原地不动 = 7
                 state_dim,  # 每个网格状态的维度（3*n_valid_grid + 144）
                 env,  # 城市仿真环境：提供网格数、地图尺寸、邻居关系等
                 scope="estimator",  # 变量命名空间：区分主网络与目标网络
                 summaries_dir=None):  # TensorBoard 日志根目录；None 表示不记录
        self.sess = sess  # 保存会话引用
        self.n_valid_grid = env.n_valid_grids  # 有效网格总数（论文实验城市为 504）
        self.action_dim = action_dim  # 动作数
        self.state_dim = state_dim  # 状态维度
        self.M = env.M  # 地图行数（仅 _build_cnn_model 会用到）
        self.N = env.N  # 地图列数（仅 _build_cnn_model 会用到）
        self.scope = scope  # 本网络的变量命名空间
        self.T = 144  # 一天的时间步数（每步 10 分钟，共 24 小时）
        self.env = env  # 保存环境引用，供动作转换时查询网格编号

        # Writes Tensorboard summaries to disk
        # TensorBoard（TensorFlow 自带的可视化工具）日志写入器，先置空
        self.summary_writer = None
        # 在命名空间 scope 下创建变量：两套网络（主/目标）互不干扰、便于整体复制参数
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()  # 搭建价值网络计算图（占位符、网络层、损失、训练算子）

            # 计算损失对本命名空间内全部可训练变量的梯度（仅用于记录梯度范数摘要，不直接用于更新）
            self.loss_gradients = tf.gradients(self.loss, tf.trainable_variables(scope=scope))  # 梯度列表：loss 对域内各可训练变量
                                           # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        # Summaries for Tensorboard
        # 合并多条摘要（标量日志）为一个算子，训练时一并执行、写入 TensorBoard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),  # 记录训练损失（均方误差）
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),  # 记录全部网格价值的均值
            tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))  # 记录梯度范数：各变量梯度向量的长度之和
        ])

        if summaries_dir:  # 指定了日志目录才创建写入器
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))  # 拼接子目录路径，名字形如 summaries_q_estimator
            if not os.path.exists(summary_dir):  # 目录不存在时
                os.makedirs(summary_dir)  # 逐级创建该目录
            self.summary_writer = tf.summary.FileWriter(summary_dir)  # 创建 TensorBoard 事件文件写入器

        # 预计算邻居序号表：neighbors_list[i] = [网格 i 各有效邻居在 target_grids 中的序号..., 网格 i 自己的序号]，
        # 顺序与动作维度对齐（前面是各方向邻居、末位是自己），action/compute_targets 靠它取邻居价值
        self.neighbors_list = []  # 每个网格的邻居序号表（含自己在末位）
        for idx, node_id in enumerate(env.target_grids):  # 遍历全部有效网格：idx=网格序号，node_id=网格节点编号
            # 第一层邻居（一步可达的 6 个方向中有效的那些）在 env.nodes 中的编号
            neighbor_indices = env.nodes[node_id].layers_neighbors_id[0]  # index in env.nodes
            # 把邻居编号换算成 target_grids 序号——该序号同时也是状态矩阵/价值输出的行号
            neighbor_ids = [env.target_grids.index(env.nodes[item].get_node_index()) for item in neighbor_indices]
            neighbor_ids.append(idx)  # 末位追加自己的序号，对应“原地不动”动作
            # index in env.target_grids == index in state
            self.neighbors_list.append(neighbor_ids)  # 存入邻居序号表

        # compute valid action mask.
        # 预计算合法动作掩码与“动作->目的网格”映射表：共 7 列，
        # 第 0~5 列对应六边形地图的 6 个邻居方向，第 6 列（最后一列）对应“原地不动”
        self.valid_action_mask = np.ones((self.n_valid_grid, self.action_dim))  # 动作掩码：1=合法动作，0=非法（该方向越界或无有效网格）
        # 动作目的网格编号表（env.nodes 体系）：行=出发网格，列=动作
        self.valid_neighbor_node_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.nodes
        # 动作目的网格序号表（env.target_grids 体系）：行=出发网格，列=动作
        self.valid_neighbor_grid_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.target_grids
        for grid_idx, grid_id in enumerate(env.target_grids):  # 遍历每个有效网格
            for neighbor_idx, neighbor in enumerate(self.env.nodes[grid_id].neighbors):  # 遍历该网格 6 个方向的邻居槽位（None=该方向越界或无有效网格）
                if neighbor is None:  # 该方向没有有效邻居
                    self.valid_action_mask[grid_idx, neighbor_idx] = 0  # 该动作标记为非法
                else:  # 该方向有有效邻居
                    # 取该邻居网格的唯一编号（env.nodes 体系）
                    node_index = neighbor.get_node_index()  # node_index in env.nodes
                    self.valid_neighbor_node_id[grid_idx, neighbor_idx] = node_index  # 记录该动作的目的网格编号
                    self.valid_neighbor_grid_id[grid_idx, neighbor_idx] = env.target_grids.index(node_index)  # 记录目的网格在 target_grids 中的序号

            self.valid_neighbor_node_id[grid_idx, -1] = grid_id  # 最后一列（第 7 个动作）=原地不动：目的即本网格编号
            self.valid_neighbor_grid_id[grid_idx, -1] = grid_idx  # 原地不动动作的目的网格序号=本网格序号

    # 搭建价值网络：3 层全连接隐层 + 1 个输出；输入每个网格的状态，输出该网格的一个价值
    def _build_model(self):
        trainable = True  # 可训练标记（仅在下方已注释掉的旧代码中用到，当前未使用）
        # 状态占位符 [批大小, state_dim]：placeholder 是 TF1 的输入占位口，运行时才喂入具体数据
        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # The TD target value
        # TD 目标占位符（TD=时序差分）：作为回归标签 y = r + γ·V(s')
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")

        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")  # 学习率占位符：训练时动态指定步长

        # 3 layers feed forward network.
        # 3 层前馈全连接网络（fc 来自 alg_utility：输入×权重+偏置，再过激活函数）
        l1 = fc(X, "l1", 128, act=tf.nn.relu)  # 第 1 层：state_dim -> 128 个隐单元（ReLU 激活）
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)  # 第 2 层：128 -> 64
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)  # 第 3 层：64 -> 32
        # l1 = tf.layers.dense(X, 1024, tf.nn.sigmoid, trainable=trainable)
        # l2 = tf.layers.dense(l1, 512, tf.nn.sigmoid, trainable=trainable)
        # l3 = tf.layers.dense(l2, 32, tf.nn.sigmoid, trainable=trainable)
        # 输出层：每个网格输出 1 个非负价值（ReLU 保证非负），可理解为“该网格的前景评分”，
        # 分数越高越值得被派去——注意与 IDQN 不同：这里每格只输出 1 个价值，而非每个动作 1 个 Q 值
        self.value_output = fc(l3, "value_output", 1, act=tf.nn.relu)  # 输出层：32 -> 1（每格一个价值）

        # self.losses = tf.square(self.y_pl - self.value_output)
        self.loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))  # 损失=TD 目标 y 与网络输出的均方误差（本质是回归问题）

        # 训练算子：AdamOptimizer（自适应学习率的梯度下降变种）按喂入的学习率最小化损失
        self.train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.loss)  # 训练算子：每执行一次即完成一步参数更新


    # 前向推理：对一批网格状态输出各自的价值（只预测，不训练）
    def predict(self, s):
        value_output = self.sess.run(self.value_output, {self.state: s})  # 运行计算图的 value_output 节点，喂入状态 s（[网格数, state_dim]）

        return value_output  # 返回 [网格数, 1] 的价值矩阵

    # 为全部网格一次性计算调度动作：用邻居价值（上下文）构造 ε-greedy 动作概率，
    # 再按各网格的空闲司机数做多项式抽样，输出 (起点, 终点, 人数) 形式的调度列表
    # 参数：s=各网格状态 [网格数, state_dim]；context=各网格可派出的空闲司机数列表；
    #       epsilon=探索概率（ε-greedy：以 ε 随机选动作去探索，以 1-ε 选最优动作去利用）
    def action(self, s, context, epsilon):
        """ Compute current action for all grids give states

        :param s: 504 x stat_dim,
        :return:
        """
        # 注：docstring 里的 504 是论文实验城市的有效网格数，stat_dim 即 state_dim

        # value of each grid at next time step, dispatched according to this value.
        # 网络前向：得到每个网格的价值（一维数组，按 target_grids 序号排列），
        # 司机依据这些价值决定去向——这是 cDQN 的核心：去哪个格子就看哪个格子的价值
        value_output = self.sess.run(self.value_output, {self.state: s}).flatten()
        action_tuple = []  # 调度动作列表，元素为 (起点网格编号, 终点网格编号, 调度司机数)
        valid_prob = []  # 各网格的动作概率分布（每行和为 1），供 compute_targets 构造学习目标

        # 状态最后 n_valid_grid 维是网格编号 one-hot，argmax 还原每行对应哪个网格
        grid_ids = np.argmax(s[:, -self.n_valid_grid:], axis=1)

        for idx, grid_valid_idx in enumerate(grid_ids):  # 逐网格决策：idx 与 context 对齐，grid_valid_idx 为当前网格序号
            # cDQN 关键一步：取“该网格各有效邻居 + 自己”的价值作为各调度动作的价值
            # （去邻居 k 的动作价值=邻居 k 的价值；原地不动=自己的价值）
            valid_qvalues = value_output[self.neighbors_list[grid_valid_idx]]
            temp_qvalue = np.zeros(self.action_dim)  # 7 个动作的临时价值数组

            if np.sum(valid_qvalues) == 0:  # 邻居与自己的价值全为 0（如训练初期网络输出全 0）
                # 在合法动作上均匀分布，相当于纯随机探索
                # all value equals to 0. this could explores conflicts action.
                temp_qvalue[self.valid_action_mask[grid_valid_idx] > 0] = 1. / np.sum(  # 每个合法动作的概率 = 1/合法动作数
                    self.valid_action_mask[grid_valid_idx])
                action_prob = temp_qvalue  # 该均匀分布即动作概率
                valid_prob.append(action_prob)  # 记录本网格的动作概率
            else:  # 存在非零价值：按价值构造 ε-greedy 概率

                temp_qvalue[self.valid_action_mask[grid_valid_idx] > 0] = valid_qvalues  # 把邻居/自己的价值填入对应的合法动作位（非法动作位保持 0）
                temp_qvalue[temp_qvalue < temp_qvalue[-1]] = 0  # 价值低于“原地不动”价值（最后一列）的动作置 0：不去价值不如留下的邻居

                best_action = np.argmax(temp_qvalue)  # 最优动作：价值最大的动作下标
                num_valid_action = np.count_nonzero(temp_qvalue)  # 候选动作数（价值非 0 的动作个数）
                action_prob = np.zeros_like(temp_qvalue)  # 动作概率向量
                action_prob[temp_qvalue > 0] = epsilon / float(num_valid_action)  # ε-greedy：每个候选动作分得 ε/候选数 的探索概率
                action_prob[best_action] += 1 - epsilon  # 最优动作再叠加 1-ε 的利用概率
                valid_prob.append(action_prob)  # 记录本网格的动作概率

            if int(context[idx]) == 0:  # 本网格没有可派出的空闲司机（预匹配后剩余 0 个）
                continue  # 跳过本网格，不产生调度动作
            # 多项式抽样：把 context[idx] 个空闲司机按动作概率分配到各动作，
            # 返回每个动作分到的司机数（总试验次数=司机数，每次按概率落在一个动作上）
            curr_action_indices = np.random.multinomial(int(context[idx]),  # 抽样总次数=本网格可派出的司机数
                                                        action_prob)  # 各动作被抽中的概率

            start_node_id = self.env.target_grids[grid_valid_idx]  # 本网格编号（env.nodes 体系），即调度起点
            for curr_action_idx, num_driver in enumerate(curr_action_indices):  # 遍历各动作分到的司机数
                if num_driver > 0:  # 该动作分到了司机
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx, curr_action_idx])  # 查表得到该动作的目的网格编号
                    if end_node_id != start_node_id:  # 目的地不是本网格（排除“原地不动”）
                        action_tuple.append((start_node_id, end_node_id, num_driver))  # 记录一条调度指令

        return action_tuple, np.stack(valid_prob)  # 返回调度动作列表 + 各网格动作概率矩阵 [网格数, 7]

    # 计算学习目标（TD 目标）：对每个网格，取其概率最大的动作所指向的目的网格，
    # 用“该目的网格的奖励 + γ × 该目的网格在下一状态的价值”作为本网格价值的回归目标
    # （γ=折扣因子，权衡眼前与长远收益；这是 Bellman 目标 r + γ·V(s') 的形式，
    #   且奖励与价值都取自邻居/目的网格——正是 cDQN 引入上下文的地方）
    # 参数：valid_prob=各网格动作概率矩阵（action 的第二个返回值）；next_state=下一步各网格状态；
    #       node_reward=各网格的单司机平均奖励；gamma=折扣因子
    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        targets = []  # 收集每个网格的学习目标
        node_reward = node_reward.flatten()  # 奖励展平为一维（按网格序号排列）
        # 网络前向：下一状态下每个网格的价值（调用方通常传入 target network 目标网络，
        # 即用较旧、较稳定的参数算目标值，避免“自己追自己”导致训练不稳定）
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()  # qvalue of next state

        for idx in np.arange(self.n_valid_grid):  # 逐网格计算目标
            grid_prob = valid_prob[idx][self.valid_action_mask[idx]>0]  # 本网格在合法动作上的概率
            neighbor_grid_ids = self.neighbors_list[idx]  # 本网格的邻居序号表（末位是自己）
            best_grid = np.argmax(grid_prob)  # 概率最大动作的下标 = 其目的网格在邻居表中的位置
            curr_grid_target = node_reward[neighbor_grid_ids][best_grid] + gamma * qvalue_next[neighbor_grid_ids][best_grid]  # Bellman 目标：目的网格的奖励 + γ × 目的网格在下一状态的价值
            targets.append(curr_grid_target)  # 加入目标列表

        return np.array(targets).reshape([-1, 1])  # 整形为 [网格数, 1]，与网络输出形状对齐

    # 执行一次训练步（不写 TensorBoard 日志）：可用于训练开始前的预热更新
    # 参数：s=状态批；y=目标批；learning_rate=本步学习率
    def initialization(self, s, y, learning_rate):
        sess = self.sess  # 取出会话
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}  # 准备喂数据：状态、目标、学习率
        _, loss = sess.run([self.train_op, self.loss], feed_dict)  # 执行一次梯度更新并取回损失
        return loss  # 返回本批损失

    # 执行一步梯度更新：让网络输出向目标 y 回归，并把摘要写入 TensorBoard
    # 参数：s=状态批 [批大小, state_dim]；y=TD 目标批 [批大小, 1]；
    #       learning_rate=本步学习率；global_step=全局步数（日志横轴）
    # 注：docstring 里的参数 a 是从 DQN 版本遗留的，本方法实际没有动作参数
    def update(self, s, y, learning_rate, global_step):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, state_dim]
          a: Chosen actions of shape [batch_size, action_dim], 0, 1 mask
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        sess = self.sess  # 取出会话
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}  # 喂数据：状态、TD 目标、学习率
        summaries, _, loss = sess.run([self.summaries, self.train_op, self.loss], feed_dict)  # 执行训练算子，同时取回摘要与损失

        if self.summary_writer:  # 若配置了日志写入器
            self.summary_writer.add_summary(summaries, global_step)  # 把本步摘要写入 TensorBoard 日志
            self.summary_writer.flush()  # 立即落盘
        return loss  # 返回本批损失

    # 备选的 CNN（卷积神经网络）建模方案：把全城司机/订单分布当作 MxN“图像”用卷积层提特征；
    # 注意：构造函数只会调用 _build_model，本方法在整个项目中没有被调用（实验性/遗留代码）
    def _build_cnn_model(self):

        # states of grid id and time
        # 输入占位符：网格编号 + 时间步的 one-hot 拼接向量（uint8 整型；本方法内后续未再使用）
        self.state_spacetime = Xst = tf.placeholder(shape=[None, self.n_valid_grid + self.T], dtype=tf.uint8, name="Xst")

        # states of distribution
        # 输入占位符：司机/订单分布图 [批大小, M, N, 4]，类似 4 通道的小图像
        self.state = X = tf.placeholder(shape=[None, self.M, self.N,  4], dtype=tf.uint8, name="X")

        # The TD target value
        # TD 目标占位符（此处为一维形状）
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        batch_size = tf.shape(self.state)[0]  # 动态取批大小（占位符第一维为 None；本方法内后续未再使用）

        # 三个卷积层逐级提取空间特征：Conv2D(滤波器个数, 卷积核尺寸, 步幅)，
        # padding="same" 保持特征图尺寸不变，activation="relu" 提供非线性
        conv1 = tf.keras.layers.Conv2D(32, 8, strides=4, padding="same", activation="relu")(  # 第 1 层：32 个 8x8 卷积核、步幅 4
            tf.cast(X, tf.float32))  # 先把 uint8 输入转成 float32 再送入卷积层
        conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(conv1)  # 第 2 层：64 个 4x4 卷积核、步幅 2
        conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")(conv2)  # 第 3 层：64 个 3x3 卷积核、步幅 1

        # Fully connected layers
        # 展平后接全连接层，最终为每个动作输出一个预测值
        flattened = tf.keras.layers.Flatten()(conv3)  # 把卷积特征图展平成一维向量
        fc1 = tf.keras.layers.Dense(512, activation="relu")(flattened)  # 全连接层：-> 512 个隐单元
        self.predictions = tf.keras.layers.Dense(self.action_dim, activation="relu")(fc1)  # 输出层：-> 每个动作一个预测值


# 状态处理器：把仿真环境的原始全局状态加工成神经网络所需的逐网格状态/奖励，
# 并计算上下文（各网格可派出的空闲司机数）；与 IDQN.stateProcessor 基本一致
class stateProcessor:
    """
        Process a raw global state into the states of grids.
    """

    # 参数：target_id_states=从展平状态向量中挑有效网格数值用的下标表；
    #       target_grids=有效网格编号列表；n_valid_grids=有效网格数
    def __init__(self,
                 target_id_states,  # 下标表：用于从展平状态中挑出各有效网格的司机数/订单数
                 target_grids,  # 有效网格编号列表（env.nodes 体系）
                 n_valid_grids):  # 有效网格数量
        # 状态取值下标表：原始状态展平后按它挑出各有效网格的司机数与订单数
        self.target_id_states = target_id_states  # valid grid index for driver and order distribution.
        # 有效网格编号列表（如 [22, 24, ...]，长度即有效网格数）
        self.target_grids = target_grids   # valid grid id [22, 24, ...]
        self.n_valid_grids = n_valid_grids  # 有效网格数
        self.T = 144  # 一天 144 个时间步（每步 10 分钟）
        self.action_dim = 7  # 动作数：6 个邻居方向 + 原地不动
        self.extend_state = True  # 标记使用“扩展状态”格式（含时间与网格编号 one-hot）

    # 把原始全局状态（[空闲司机分布 MxN, 订单分布 MxN]）转成有效网格上的向量：
    # 展平后按下标表挑出各有效网格的司机数与订单数
    def utility_conver_states(self, curr_state):
        curr_s = np.array(curr_state).flatten()  # 转数组并展平：前 M*N 个是司机数，后 M*N 个是订单数
        curr_s_new = [curr_s[idx] for idx in self.target_id_states]  # 只保留有效网格的数值（先各格司机数，后各格订单数）
        return np.array(curr_s_new)  # 返回长度 2*n_valid_grids 的一维数组

    # 状态归一化：司机数、订单数各自除以全城最大值，压到 [0,1] 区间，利于网络训练
    def utility_normalize_states(self, curr_s):
        max_driver_num = np.max(curr_s[:self.n_valid_grids])  # 全部有效网格中的最大空闲司机数
        max_order_num = np.max(curr_s[self.n_valid_grids:])  # 全部有效网格中的最大订单数

        curr_s_new = np.zeros_like(curr_s)  # 开辟与输入同形状的结果数组
        curr_s_new[:self.n_valid_grids] = curr_s[:self.n_valid_grids] / max_driver_num  # 司机数归一化
        curr_s_new[self.n_valid_grids:] = curr_s[self.n_valid_grids:] / max_order_num  # 订单数归一化
        return curr_s_new  # 返回归一化后的状态

    # 从全城逐节点奖励向量中挑出各有效网格上的奖励
    def utility_conver_reward(self, reward_node):
        reward_node_new = [reward_node[idx] for idx in self.target_grids]  # 只保留有效网格的奖励
        return np.array(reward_node_new)  # 返回长度 n_valid_grids 的奖励数组

    # 奖励包装：从 env.step 返回的 info 中提取各有效网格的奖励，
    # 并除以该网格当时的空闲司机数，得到“平均每个司机”的奖励
    # 参数：info=env.step 的返回信息（info[0]=[含邻居广播的逐格奖励, 邻居广播奖励]）；
    #       curr_s=有效网格状态（前半段是各格司机数，作除数）
    def reward_wrapper(self, info, curr_s):
        info_reward = info[0]  # 取逐格奖励部分：[node_reward(本格+邻居广播), neighbor_reward]
        valid_nodes_reward = self.utility_conver_reward(info_reward[0])  # 取“本格奖励+邻居广播来的奖励”在有效网格上的值
        devide = curr_s[:self.n_valid_grids]  # 各有效网格的空闲司机数（原变量名 devide 即除数；注意切片是视图，改动会同步到 curr_s）
        devide[devide == 0] = 1  # 司机数为 0 的格子除数置 1，避免除零
        valid_nodes_reward = valid_nodes_reward/devide  # 逐格相除：得到单司机平均奖励
        return valid_nodes_reward  # 返回各有效网格的平均奖励

    # 计算上下文：从“预匹配后的剩余司机/订单”信息中提取各有效网格的剩余空闲司机数，
    # 该数值即 action() 中本网格可派出的司机数
    def compute_context(self, info):
        # 计算context
        # info 形如 [剩余司机分布 MxN, 剩余订单分布 MxN]（env.step_pre_order_assigin 的输出：
        # 网格内及邻格间先贪心匹配一轮订单后，还剩多少没活干的空闲司机）
        context = info.flatten()  # 展平成一维：前 M*N 个是剩余司机数，后 M*N 个是剩余订单数
        context = [context[idx] for idx in self.target_grids]  # 只取各有效网格的剩余司机数（下标均落在前半段）
        return context  # 返回各有效网格的剩余空闲司机数列表

    # 把全局状态加工成“逐网格状态矩阵”：每行=[全城司机分布, 全城订单分布, 时刻 one-hot, 本网格编号 one-hot]，
    # 列数=3*n_valid_grids+T；所有行共享同一份全城信息，仅最后的网格 one-hot 不同
    # 参数：curr_s 通常为归一化后的 [司机分布 n, 订单分布 n] 拼接向量；curr_city_time=当前时刻
    def to_grid_states(self, curr_s, curr_city_time):

        T = self.T  # 一天的时间步数（144）

        # curr_s = self.utility_conver_states(curr_state)
        # 时刻 one-hot：T 维向量，“一天内的时刻序号”（对 T 取余）对应位置为 1
        time_one_hot = np.zeros((T))
        time_one_hot[curr_city_time % T] = 1  # curr_city_time % T = 当前时刻在一天内的序号
        onehot_grid_id = np.eye(self.n_valid_grids)  # n 阶单位矩阵：第 i 行即网格 i 的 one-hot 编码

        s_grid = np.zeros((self.n_valid_grids, self.n_valid_grids * 3 + T))  # 逐网格状态矩阵：n_valid_grids 行 x (3n+T) 列，初始全零
        s_grid[:, :self.n_valid_grids * 2] = np.stack([curr_s] * self.n_valid_grids)  # 前 2n 列：每行复制同一份 [司机分布, 订单分布]（全城上下文对所有网格可见）
        s_grid[:, self.n_valid_grids * 2:self.n_valid_grids * 2 + T] = np.stack([time_one_hot] * self.n_valid_grids)  # 中间 T 列：每行复制同一份时刻 one-hot
        s_grid[:, -self.n_valid_grids:] = onehot_grid_id  # 末尾 n 列：网格编号 one-hot，第 i 行仅第 i 列为 1（标识本行是哪个网格的状态）

        return np.array(s_grid)  # 返回 [n_valid_grids, 3n+T] 的状态矩阵

    # 把逐网格奖励向量整形成 [网格数, 1] 列向量，与网络输出/目标占位符形状对齐
    def to_grid_rewards(self, node_reward):
        return np.array(node_reward).reshape([-1, 1])  # 展平的奖励 -> 列向量

    # 把动作下标列表转成 one-hot 动作矩阵：第 i 行在 action_neighbor_idx[i] 列为 1
    def to_action_mat(self, action_neighbor_idx):
        action_mat = np.zeros((len(action_neighbor_idx), self.action_dim))  # [样本数, 7] 的全零矩阵
        action_mat[np.arange(action_mat.shape[0]), action_neighbor_idx] = 1  # 每行在所选动作列置 1
        return action_mat  # 返回 one-hot 动作矩阵


# 经验回放池（replay buffer）：把交互产生的 (状态, 动作, 目标, 下一状态) 存起来，
# 训练时随机抽一批，打破样本间的时间相关性，让更新更稳定（“without time ordering”即不保时序）
class ReplayMemory:
    """ collect the experience and sample a batch for training networks.
        without time ordering
    """
    # 参数：memory_size=池容量（存满后随机覆盖旧样本）；batch_size=每次训练抽取的批大小
    def __init__(self, memory_size, batch_size):
        self.states = []  # 状态存储区（初始为空列表，存入第一批后变 numpy 数组）
        self.next_states = []  # 下一状态存储区
        self.actions = []  # 动作存储区
        self.rewards = []  # 奖励（学习目标）存储区

        self.batch_size = batch_size  # 每次采样的批大小
        self.memory_size = memory_size  # 池容量
        self.current = 0  # （保留字段）当前写入位置，本实现未使用
        self.curr_lens = 0  # 当前已存样本总数

    # 添加一批样本 (s, a, r, next_s)：首批直接存入；未满则沿样本维拼接；
    # 已满则在随机位置整块覆盖一段旧样本（近似均匀地遗忘旧经验）
    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:  # 池为空：第一批样本直接保存
            self.states = s  # 保存状态
            self.actions = a  # 保存动作
            self.rewards = r  # 保存奖励
            self.next_states = next_s  # 保存下一状态
            self.curr_lens = self.states.shape[0]  # 当前样本数 = 第一批的行数

        elif self.curr_lens <= self.memory_size:  # 池未满：新样本沿第 0 维（样本维）拼接
            self.states = np.concatenate((self.states, s),axis=0)  # 拼接状态
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)  # 拼接下一状态
            self.actions = np.concatenate((self.actions, a), axis=0)  # 拼接动作
            self.rewards = np.concatenate((self.rewards, r), axis=0)  # 拼接奖励
            self.curr_lens = self.states.shape[0]  # 更新样本总数
        else:  # 池已满：随机选一段连续位置，整块覆盖为新样本
            new_sample_lens = s.shape[0]  # 本批样本数
            index = random.randint(0, self.curr_lens - new_sample_lens)  # 随机覆盖起点（保证待覆盖段不越界）

            self.states[index:(index + new_sample_lens)] = s  # 覆盖状态
            self.actions[index:(index + new_sample_lens)] = a  # 覆盖动作
            self.rewards[index:(index + new_sample_lens)] = r  # 覆盖奖励
            self.next_states[index:(index + new_sample_lens)] = next_s  # 覆盖下一状态

    # 随机采样一个批次：样本不多于批大小时全量返回，否则无放回随机抽取 batch_size 条
    def sample(self):

        if self.curr_lens <= self.batch_size:  # 样本数不多于批大小：直接返回全部
            return [self.states, self.actions, self.rewards, self.next_states]  # 返回全部 [状态, 动作, 奖励, 下一状态]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)  # 无放回随机抽取 batch_size 个行号
        batch_s = self.states[indices]  # 按行号取状态批
        batch_a = self.actions[indices]  # 按行号取动作批
        batch_r = self.rewards[indices]  # 按行号取奖励批
        batch_next_s = self.next_states[indices]  # 按行号取下一状态批
        return [batch_s, batch_a, batch_r, batch_next_s]  # 返回一个批次 [状态, 动作, 奖励, 下一状态]

    # 清空回放池：所有存储区重置为空列表
    def reset(self):
        self.states = []  # 清空状态
        self.actions = []  # 清空动作
        self.rewards = []  # 清空奖励
        self.next_states = []  # 清空下一状态
        self.curr_lens = 0  # 样本数归零



# 参数复制器：把 estimator1（如主 Q 网络）的全部可训练参数复制给 estimator2（如 target network）；
# 目标网络参数隔一段才同步一次、期间保持不变，让学习目标更稳定，避免“自己追自己”
class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """

    # 参数：estimator1=参数来源网络；estimator2=参数去向网络
    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        # 按变量名前缀（scope）筛出 estimator1 的全部可训练变量
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)  # 按变量名排序，保证与另一侧同结构变量一一配对
        # 同样筛出 estimator2 的全部可训练变量
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)  # 同样按变量名排序

        self.update_ops = []  # 收集“赋值操作”的列表
        for e1_v, e2_v in zip(e1_params, e2_params):  # 逐对配对两个网络的同结构变量
            op = e2_v.assign(e1_v)  # 构造赋值算子：把 e1 的当前值赋给 e2
            self.update_ops.append(op)  # 加入操作列表

    # 执行复制：在会话中一次性运行全部赋值算子，完成参数同步
    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)  # 运行所有 assign 操作，完成参数复制
