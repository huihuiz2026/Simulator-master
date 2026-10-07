
# 本文件实现 IDQN（Independent DQN，独立 DQN）算法：把城市里每个网格当作一个独立的智能体
# （智能体 = 能感知状态、选择动作、接收奖励的学习单元），各自学习"把本网格空闲司机调度去哪"
# 的策略。所有网格智能体共用同一个 Q 网络参数，靠状态末段的网格 one-hot 编码区分身份。
# DQN = 用深度神经网络近似 Q 函数的值迭代算法；Q 函数 = 评估"在某状态下采取某动作"的长期回报期望。
# 文件含四个组件：Estimator（Q 网络估计器）、stateProcessor（状态/奖励加工器）、
# ReplayMemory（经验回放池）、ModelParametersCopier（主网络到 target 网络的参数复制器）。
# 代码为 TF1 session 风格（session = TensorFlow 会话，负责执行计算图），经 tf_compat 适配层运行在 TF2 下。

# 导入 NumPy 数值计算库并简写为 np：用于矩阵/数组运算
import numpy as np
# 导入同包 tf_compat 提供的 TensorFlow：其内部执行 import tensorflow.compat.v1 as tf
# 并关闭 v2 行为，使 session/placeholder 等旧 API 在装了 TF2 的机器上仍可用
from .tf_compat import tf
# 导入标准库 random（回放池随机抽样等）与 os（拼接/创建日志目录）
import random, os
# 导入同包 alg_utility（算法公共工具）的全部内容：本文件用到其中的全连接层 dense() 等组件
from .alg_utility import *


# Q 网络估计器：搭建 DQN 的 Q 网络并提供"预测 Q 值 / 选调度动作 / 训练更新"三类接口
class Estimator:
    """ build Deep Q network
    """
    # 初始化估计器：保存维度信息、搭建网络计算图、预计算邻居表与合法动作掩码
    # 参数说明：sess —— TF 会话（执行计算图的运行器）；
    #   action_dim —— 动作维度（7 = 6 个相邻网格方向 + 1 个原地不动）；
    #   state_dim —— 状态向量维度（有效网格数*3 + 144）；env —— 城市仿真环境对象；
    #   scope —— 变量命名空间前缀（同一张计算图中区分不同网络的变量名）；
    #   summaries_dir —— TensorBoard（TF 官方训练过程可视化工具）日志目录，None 表示不记录
    def __init__(self,
                 sess,
                 action_dim,
                 state_dim,
                 env,
                 scope="estimator",
                 summaries_dir=None):
        self.sess = sess  # 保存会话，后续 predict/action/update 都通过它执行计算图
        self.n_valid_grid = env.n_valid_grids  # 有效网格（可通行、参与调度的网格）数量
        self.action_dim = action_dim  # 动作维度：6 个邻居方向 + 原地不动，共 7
        self.state_dim = state_dim  # 状态向量维度 = 有效网格数*3 + 144（时间 one-hot 长度）
        self.M = env.M  # 城市地图行数
        self.N = env.N  # 城市地图列数
        self.scope = scope  # 本网络的变量命名空间，ModelParametersCopier 靠它定位本网络参数
        self.T = 144  # 一天的时间步数：144 步 x 10 分钟 = 24 小时
        self.env = env  # 保存环境引用，action() 里查询网格拓扑时使用

        # TensorBoard 摘要写入器先置空（配置了日志目录才创建）
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        # 在 scope 命名空间内建网（variable_scope：给内部变量名加前缀，避免不同网络的变量重名）
        with tf.variable_scope(scope):
            # Build the graph
            # 调用内部方法搭建网络计算图（占位符、全连接层、损失、优化器）
            self._build_model()
            # 若指定了日志目录，则为本网络创建独立的摘要输出目录
            if summaries_dir:
                # 拼接子目录名，如 summaries_q_estimator（用 scope 区分不同网络）
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                # 目录不存在则逐级创建
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                # 创建摘要写入器：把 loss 等数据写盘，供 TensorBoard 读取展示
                self.summary_writer = tf.summary.FileWriter(summary_dir)

        # 每个网格的"第一层邻居 + 自身"在 target_grids 中的下标列表
        self.neighbors_list = []
        # 逐网格构建：idx 是网格在 target_grids 中的下标，node_id 是它在 env.nodes 中的编号
        for idx, node_id in enumerate(env.target_grids):
            # 取该网格第一层直接邻居的节点编号列表
            neighbor_indices = env.nodes[node_id].layers_neighbors_id[0]  # index in env.nodes
            # 把每个邻居的节点编号换算成 target_grids 下标（与状态向量中的网格下标一致）
            neighbor_ids = [env.target_grids.index(env.nodes[item].get_node_index()) for item in neighbor_indices]
            # 把自身下标追加到末尾：动作空间最后一维"原地不动"对应自身
            neighbor_ids.append(idx)
            # target_grids 的下标 == 状态向量里的网格下标，直接存入邻居表
            # index in env.target_grids == index in state
            self.neighbors_list.append(neighbor_ids)

        # 预计算三张 [网格下标 x 动作槽位] 查询表：动作掩码、终点节点编号、终点网格下标
        # compute valid action mask.
        self.valid_action_mask = np.ones((self.n_valid_grid, self.action_dim))  # 全 1 初始化；某方向无邻居时对应位置置 0
        # 每个动作槽位的终点节点编号表（env.nodes 下标），先置 0 待填充
        self.valid_neighbor_node_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.nodes
        # 每个动作槽位的终点网格下标表（target_grids 下标），先置 0 待填充
        self.valid_neighbor_grid_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.target_grids
        # 逐网格填表：grid_idx 是 target_grids 下标，grid_id 是 env.nodes 编号
        for grid_idx, grid_id in enumerate(env.target_grids):
            # 遍历该网格的 6 个邻居槽位（neighbor 为 None 表示该方向没有相邻网格）
            for neighbor_idx, neighbor in enumerate(self.env.nodes[grid_id].neighbors):
                if neighbor is None:  # 该方向无相邻网格
                    self.valid_action_mask[grid_idx, neighbor_idx] = 0  # 掩码置 0：禁止往这个方向调度
                else:  # 有邻居：记录终点编号，掩码保持 1
                    # 取邻居节点在 env.nodes 中的编号
                    node_index = neighbor.get_node_index()  # node_index in env.nodes
                    self.valid_neighbor_node_id[grid_idx, neighbor_idx] = node_index  # 填入终点节点编号表
                    self.valid_neighbor_grid_id[grid_idx, neighbor_idx] = env.target_grids.index(node_index)  # 换算成 target_grids 下标填入

            self.valid_neighbor_node_id[grid_idx, -1] = grid_id  # 第 7 个动作（最后一维）= 原地不动：终点就是自身节点
            self.valid_neighbor_grid_id[grid_idx, -1] = grid_idx  # 同理，终点网格下标填自身下标



    # 搭建网络计算图：定义输入占位符、3 层全连接 Q 网络、平方 TD 损失和 Adam 优化训练算子
    def _build_model(self):
        trainable = True  # "变量可训练"标志（本方法后续未再使用，保留自原代码）
        # 状态占位符 X（placeholder = 计算图里的输入"坑位"，运行时才喂数据；第一维 None 表示批大小可变）
        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # TD 目标占位符 y（TD = 时序差分；即 Bellman 目标 r + γ·max Q(s')：当前奖励加上折扣后下一状态的最大价值）
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")  # 每个样本一个目标值

        # 已选动作占位符：one-hot 掩码（仅选中位置为 1 的 0/1 向量），用于从 Q 值向量里取出所选动作的 Q 值
        # action chosen
        self.ACTION = tf.placeholder(tf.float32, [None, self.action_dim], 'action_chosen')

        # 学习率占位符：训练时动态传入优化器（控制每步参数更新的幅度）
        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        # 3 层隐藏全连接网络（dense 是 alg_utility 封装的全连接层；elu 为平滑激活函数，允许负值输出）
        # 3 layers feed forward network.
        l1 = dense(X, 128, tf.nn.elu, "dense")  # 第 1 层：128 个神经元
        l2 = dense(l1, 64, tf.nn.elu, "dense_1")  # 第 2 层：64 个神经元
        l3 = dense(l2, 32, tf.nn.elu, "dense_2")  # 第 3 层：32 个神经元
        self.qvalue = dense(l3, self.action_dim, tf.nn.elu, "dense_3")  # 输出层：第 i 维即 Q(s, 第 i 个动作)


        # one-hot 动作掩码点乘 Q 值向量再按行求和，等价于取出每个样本所选动作的 Q(s,a)
        # get the Q(s,a) for chosen action
        self.action_predictions = tf.reduce_sum(self.qvalue * self.ACTION, axis=1)

        # 逐样本平方 TD 误差：(目标 y - 网络预测 Q(s,a))^2
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)  # 批内平均损失，作为优化目标

        # 训练算子：AdamOptimizer（自适应步长的梯度下降变体）按传入学习率最小化损失；run(train_op) 即更新一步参数
        self.train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.loss)

        # 合并 TensorBoard 摘要：记录损失与 Q 值的监控量
        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([  # 把以下多条摘要合并成一个算子，一次 run 全部取出
            tf.summary.scalar("loss", self.loss),  # 标量：平均损失
            tf.summary.histogram("loss_hist", self.losses),  # 直方图：逐样本损失分布
            tf.summary.histogram("q_values_hist", self.qvalue),  # 直方图：所有动作的 Q 值分布
            tf.summary.scalar("max_q_value", tf.reduce_max(self.qvalue))  # 标量：最大 Q 值（观察是否发散）
        ])

    # 批量预测各状态的最大 Q 值 max_a Q(s,a)：供计算 Bellman 目标 r + γ·max Q(s')（通常跑在 target 网络上）
    def predict(self, s):
        qvalues = self.sess.run(self.qvalue, {self.state: s})  # 喂状态跑网络，得到各动作 Q 值 [batch, action_dim]
        max_qvalue = np.max(qvalues, axis=1)  # 沿动作维取最大值
        return max_qvalue  # 返回每个状态的最大 Q 值

    # 为所有网格选择调度动作：按 ε-greedy（以小概率 ε 随机探索、其余按模型贪心选动作）
    # 构造每个网格的动作概率，再用多项分布把该网格的空闲司机数分配到各动作上
    # 参数：s = 每行一个网格的状态矩阵；context = 每个网格可调度的空闲司机数列表；epsilon = 探索概率 ε
    def action(self, s, context, epsilon):
        """ Compute Q(s, a) for all actions give states
        :return:
        """
        # A = np.ones(self.action_dim, dtype=float) * epsilon / self.action_dim
        qvalues = self.sess.run(self.qvalue, {self.state: s})  # 跑网络：得到每个网格在各动作上的 Q 值
        # 动作终点节点编号列表（env.nodes 下标，含原地不动）
        action_idx = []            # go to which node, the index in nodes
        # 动作终点网格下标列表（target_grids 下标）
        action_idx_valid = []            #the index in env.target_grids
        action_neighbor_idx = []  # 动作槽位下标列表（0~5 为邻居方向，最后一位为原地不动）
        # 节点数 x 节点数 的调度数量矩阵：[起点, 终点] = 调过去的司机数
        action_tuple_mat = np.zeros((len(self.env.nodes), len(self.env.nodes)))
        action_tuple = []  # 输出给 env.step 的调度三元组列表，形如 (起点网格, 终点网格, 司机数)
        action_starting_gridids = []  # 每个网格实际选用的不同动作数（训练时按此复制状态样本）

        # 从状态末段的网格 one-hot 反解出每行样本的起始网格下标
        grid_ids = np.argmax(s[:, -self.n_valid_grid:], axis=1)  # starting grid of each sample
        valid_probs = []  # 收集每个网格的动作概率分布

        # 逐网格处理：idx 是样本行号，grid_valid_idx 是该行网格在 target_grids 中的下标
        for idx, grid_valid_idx in enumerate(grid_ids):
            curr_qvalue = qvalues[idx]  # 当前网格这行样本的各动作 Q 值
            temp_qvalue = self.valid_action_mask[grid_valid_idx] * curr_qvalue  # 掩码相乘：非法动作的 Q 值清零

            # 掩码后 Q 值全为 0（网络没给出有效偏好）：在合法动作上均匀随机，鼓励探索
            if np.sum(temp_qvalue) == 0:  # encourage exploration
                # 合法动作位置均设为 1/合法动作数，构造均匀分布
                temp_qvalue[self.valid_action_mask[grid_valid_idx]>0] = 1. / np.sum(self.valid_action_mask[grid_valid_idx])
                action_prob = temp_qvalue / np.sum(temp_qvalue)  # 归一化成概率分布（和为 1）
                # 多项分布抽样：把该网格 context[idx] 个空闲司机按概率分配到各动作，返回每个动作分到的司机数
                curr_action_indices = np.random.multinomial(int(context[idx]), action_prob)
            else:  # 常规情形：按 Q 值做 ε-greedy
                best_action = np.argmax(temp_qvalue)  # 贪心动作：合法动作中 Q 值最大者
                action_prob = np.zeros(self.action_dim)  # 动作概率先全部置 0
                num_valid_action = np.count_nonzero(temp_qvalue)  # Q 值非 0 的可选动作个数
                action_prob[temp_qvalue > 0] = epsilon / float(num_valid_action)  # 每个可选动作分得 ε/可选数 的探索概率
                action_prob[best_action] += 1 - epsilon  # 最优动作再叠加 1-ε 的贪心概率
                # 同上：按该概率把空闲司机数抽样分配到各动作（返回各动作分到的司机数）
                curr_action_indices = np.random.multinomial(int(context[idx]), action_prob)

            valid_probs.append(action_prob)  # 记录该网格的动作概率
            start_node_id = self.env.target_grids[grid_valid_idx]  # 该网格的节点编号，作为调度起点
            num_distinct_action = 0  # 计数器：该网格本轮实际选了几个不同动作
            # 遍历各动作分到的司机数：curr_action_idx 是动作槽位，num_driver 是司机数
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:  # 只处理分到司机的动作
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx, curr_action_idx])  # 查表得终点节点编号

                    action_idx.append(end_node_id)  # 记录终点节点编号
                    action_idx_valid.append(int(self.valid_neighbor_grid_id[grid_valid_idx, curr_action_idx]))  # 记录终点网格下标
                    action_neighbor_idx.append(curr_action_idx)  # 记录动作槽位（训练时构造 one-hot 动作用）
                    action_tuple_mat[start_node_id, end_node_id] = num_driver  # 起终点对记入调度数量矩阵
                    num_distinct_action += 1  # 不同动作数 +1
            action_starting_gridids.append(num_distinct_action)  # 记录该网格的不同动作数（含原地不动）

        # 从调度矩阵中取出所有非零 (起点, 终点) 对
        action_indices = np.where(action_tuple_mat > 0)
        for xx, yy in zip(action_indices[0], action_indices[1]):  # 逐对遍历 (起点 xx, 终点 yy)
            if xx != yy:  # 跳过起点 = 终点：原地不动不算调度
                action_tuple.append((xx, yy, int(action_tuple_mat[xx, yy])))  # 生成三元组 (起点, 终点, 司机数)

        # 依次返回：全部 Q 值 / 终点节点编号 / 终点网格下标 / 动作槽位 / 调度三元组 / 各网格动作数
        return qvalues, action_idx, action_idx_valid, action_neighbor_idx, action_tuple, action_starting_gridids

    # 用一小批样本执行一步梯度下降训练 Q 网络，返回本批损失
    # 参数：s = 状态批次；a = one-hot 动作掩码；y = TD 目标（r + γ·max Q(s)）；learning_rate = 学习率；global_step = 全局步数（摘要打点用）
    def update(self, s, a, y, learning_rate, global_step):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, state_dim]
          a: Chosen actions of shape [batch_size, action_dim], 0, 1 mask
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        sess = self.sess  # 取成员会话的局部别名
        feed_dict = {self.state: s, self.y_pl: y, self.ACTION: a, self.loss_lr: learning_rate}  # 把状态/目标/动作/学习率填入各占位符
        # 一次 run 同时取出摘要、执行训练算子更新参数（其返回值用 _ 丢弃）、计算当前损失
        summaries, _, loss = sess.run([self.summaries, self.train_op, self.loss], feed_dict)

        if self.summary_writer:  # 配置了日志写入器才记录
            self.summary_writer.add_summary(summaries, global_step)  # 把本批摘要写入 TensorBoard 日志
        return loss  # 返回本批平均损失



# 状态处理器：把环境的原始全局状态（全城 MxN 的司机/订单分布矩阵）加工成
# 每个网格智能体各自的状态矩阵、奖励向量、可调度司机数（上下文）等训练所需格式
class stateProcessor:
    """
        Process a raw global state into the states of grids.
    """

    # 初始化：保存有效网格的下标映射和维度参数
    # 参数：target_id_states = 全局状态展平后有效网格的抽取下标（前段取司机、后段取订单）；
    #       target_grids = 有效网格编号列表；n_valid_grids = 有效网格数量
    def __init__(self,
                 target_id_states,
                 target_grids,
                 n_valid_grids):
        # 抽取下标表（run_IDQN 中构造：司机段用网格编号、订单段用网格编号+M*N），用于从展平状态里挑出有效网格
        self.target_id_states = target_id_states  # valid grid index for driver and order distribution.
        # 有效网格编号列表（按地图编号升序，如 [22, 24, ...]，共 504 个）
        self.target_grids = target_grids   # valid grid id [22, 24, ...]  504
        self.n_valid_grids = n_valid_grids  # 有效网格数量
        self.T = 144  # 一天的时间步数（144 步 x 10 分钟 = 24 小时）
        self.action_dim = 7  # 动作维度：6 个邻居方向 + 原地不动
        self.extend_state = True  # 状态扩展标志（本文件后续未再使用，保留自原代码）

    # 状态裁剪：把全局状态 [司机分布, 订单分布] 两个 MxN 矩阵压成有效网格的一维向量
    def utility_conver_states(self, curr_state):
        curr_s = np.array(curr_state).flatten()  # 展平成长度 2*M*N 的一维数组（前半司机、后半订单）
        curr_s_new = [curr_s[idx] for idx in self.target_id_states]  # 只保留有效网格对应的分量
        return np.array(curr_s_new)  # 返回长度 2*n_valid_grids 的向量：[各有效网格司机数, 各有效网格订单数]

    # 奖励裁剪：把全城 M*N 的逐节点奖励向量压成有效网格的奖励向量
    def utility_conver_reward(self, reward_node):
        reward_node_new = [reward_node[idx] for idx in self.target_grids]  # 只保留有效网格位置的奖励
        return np.array(reward_node_new)  # 返回长度 n_valid_grids 的奖励向量

    # 奖励包装：计算每个网格的"人均"即时奖励 = 该网格奖励 / 该网格空闲司机数
    # 参数：info = env.step 返回的 [逐节点奖励, 上下文]；curr_s = 裁剪后的当前状态（前 n_valid_grids 位是司机数）
    def reward_wrapper(self, info, curr_s):
        """
        :param info: [node_reward(including neighbors), neighbor_reward]
        :param curr_s:
        :return:
        """

        info_reward = info[0]  # 取逐节点奖励部分 [node_reward(已含邻居广播), neighbor_reward]
        valid_nodes_reward = self.utility_conver_reward(info_reward[0])  # 取 node_reward 并裁剪到有效网格
        devide = curr_s[:self.n_valid_grids]  # 各有效网格的空闲司机数（注意这是 curr_s 的视图，原地修改会同步影响 curr_s）
        devide[devide == 0] = 1  # 司机数为 0 的网格分母置 1，避免除零
        valid_nodes_reward = valid_nodes_reward/devide  # 逐网格相除：得到每个空闲司机的平均奖励
        return valid_nodes_reward  # 返回逐网格的人均奖励向量

    # 计算上下文：从剩余司机/订单矩阵中抽出每个有效网格的剩余空闲司机数（即可调度的司机数量）
    def compute_context(self, info):
        # compute context
        context = info.flatten()  # info 是 [剩余司机, 剩余订单] 两个 MxN 矩阵，展平后前 M*N 位是司机
        context = [context[idx] for idx in self.target_grids]  # 只取有效网格的剩余空闲司机数
        return context  # 返回每个有效网格可调度司机数组成的列表

    # 状态归一化：司机段、订单段各自除以段内最大值，压到 0~1 区间，便于网络训练
    def utility_normalize_states(self, curr_s):
        max_driver_num = np.max(curr_s[:self.n_valid_grids])  # 司机段的最大值
        max_order_num = np.max(curr_s[self.n_valid_grids:])  # 订单段的最大值

        curr_s_new = np.zeros_like(curr_s)  # 建同形状数组存放归一化结果
        curr_s_new[:self.n_valid_grids] = curr_s[:self.n_valid_grids] / max_driver_num  # 司机段归一化
        curr_s_new[self.n_valid_grids:] = curr_s[self.n_valid_grids:] / max_order_num  # 订单段归一化
        return curr_s_new  # 返回归一化后的状态向量


    # 全局状态 -> 网格状态矩阵：每个网格一行，所有行共享全城分布与时间特征，
    # 末段用 one-hot 标识"这一行属于哪个网格"（IDQN 靠它区分共用网络的各个网格智能体）
    def to_grid_states(self, curr_s, curr_city_time):
        T = self.T  # 一天的时间步数 144

        # curr_s = self.utility_conver_states(curr_state)
        time_one_hot = np.zeros((T))  # 时间 one-hot 向量：长度 144
        time_one_hot[curr_city_time % T] = 1  # 当前时刻对 T 取模后在对应位置置 1（周期表示一天内的时间）
        onehot_grid_id = np.eye(self.n_valid_grids)  # 网格身份 one-hot 矩阵：第 i 行即第 i 个网格的编码（单位阵）

        s_grid = np.zeros((self.n_valid_grids, self.n_valid_grids * 3 + T))  # 每网格一行：2*网格数(分布) + 144(时间) + 网格数(身份)
        s_grid[:, :self.n_valid_grids * 2] = np.stack([curr_s] * self.n_valid_grids)  # 前 2*网格数 列：每行复制同一份全城 [司机分布, 订单分布]
        s_grid[:, self.n_valid_grids * 2:self.n_valid_grids * 2 + T] = np.stack([time_one_hot] * self.n_valid_grids)  # 中间 144 列：每行复制同一份时间 one-hot
        s_grid[:, -self.n_valid_grids:] = onehot_grid_id  # 末 网格数 列：第 i 行填自己的网格身份 one-hot
        return np.array(s_grid)  # 返回 [网格数, state_dim] 的状态矩阵

    # 奖励映射：按每条已选动作的终点网格，取出对应网格的奖励
    def to_grid_rewards(self, action_idx_valid, node_reward):
        r_grid = []  # 收集每个动作对应的奖励
        for end_grid_id in action_idx_valid:  # 遍历各动作的终点网格下标
            r_grid.append(node_reward[end_grid_id])  # 取终点网格的（人均）即时奖励

        return np.array(r_grid)  # 返回与动作序列逐条对齐的奖励向量


    # 构造训练用下一状态：每行填下一时刻的全城分布与时间 one-hot，
    # 网格身份位则改成"该动作到达的终点网格"（即智能体到达的新身份）
    def to_grid_next_states(self, s_grid, next_state, action_index, curr_city_time):
        """
        :param s_grid:  batch_size x state_dimension
        :param action_index: batch_size, end_valid_grid_id, next grid id.
        :return:
        """
        T = self.T  # 一天的时间步数 144
        next_s = self.utility_normalize_states(self.utility_conver_states(next_state))  # 下一全局状态裁剪到有效网格并归一化

        time_one_hot = np.zeros((T))  # 下一时刻的时间 one-hot 向量
        time_one_hot[curr_city_time % T] = 1  # 在对应时刻位置置 1

        s_grid_next = np.zeros(s_grid.shape)  # 按传入状态矩阵的形状建下一状态矩阵
        s_grid_next[:, :self.n_valid_grids*2] = next_s  # 前 2*网格数 列填下一时刻的全城分布（广播到每行）
        s_grid_next[:, self.n_valid_grids*2:self.n_valid_grids*2+T] = time_one_hot  # 中间 144 列填时间 one-hot（广播到每行）

        action_index = np.array(action_index) + self.n_valid_grids*2 + T  # 终点网格下标 + 偏移量 = 身份段在状态向量中的列位置
        s_grid_next[np.arange(s_grid_next.shape[0]), action_index] = 1  # 第 i 行在其终点网格身份列置 1

        return s_grid_next  # 返回与动作逐行对齐的下一状态矩阵

    # 训练状态复制：第 i 个网格选了几个动作，就把它的状态行复制几份
    # （每个"网格-动作"对构成一条独立的训练样本）
    def to_grid_state_for_training(self, s_grid, action_starting_gridids):
        s_grid_new = []  # 存放复制后的状态行
        for idx, num_extend in enumerate(action_starting_gridids):  # idx 是网格下标，num_extend 是该网格的动作数
            temp_s = s_grid[idx]  # 取出该网格的状态行
            s_grid_new += [temp_s] * num_extend  # 复制 num_extend 份，与动作/奖励/下一状态逐条对齐
        return np.array(s_grid_new)  # 返回复制后的训练状态矩阵

    # 动作编码：把每条样本的动作槽位下标转成 one-hot 动作矩阵（训练时用于从 Q 值向量里挑出所选动作的 Q 值）
    def to_action_mat(self, action_neighbor_idx):
        action_mat = np.zeros((len(action_neighbor_idx), self.action_dim))  # 全 0 矩阵：行数 = 样本数，列数 = 动作维度
        action_mat[np.arange(action_mat.shape[0]), action_neighbor_idx] = 1  # 第 i 行在其动作槽位列置 1，即 one-hot 编码
        return action_mat  # 返回 [样本数, action_dim] 的 one-hot 动作矩阵



# 经验回放池（缓存历史交互样本，训练时随机抽取以打破样本相关性、提高样本利用率）；
# 本实现不保留时间顺序，样本以矩阵形式整批存取
class ReplayMemory:
    """ collect the experience and sample a batch for training networks.
        without time ordering
    """
    # 初始化回放池：四类缓存置空，记录容量与批大小
    # 参数：memory_size = 池的最大容量（样本条数）；batch_size = 每次训练抽样的批大小
    def __init__(self, memory_size, batch_size):
        self.states = []  # 状态缓存
        self.next_states = []  # 下一状态缓存
        self.actions = []  # 动作（one-hot）缓存
        self.rewards = []  # 奖励缓存（实际存的是 TD 目标值）

        self.batch_size = batch_size  # 训练批大小
        self.memory_size = memory_size  # 池容量
        self.current = 0  # 写入指针（本实现未使用，保留自原代码）
        self.curr_lens = 0  # 当前池中样本条数

    # 添加一批样本：池空直接存入；未满纵向拼接；已满则随机覆盖一段旧样本
    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:  # 池为空：首批样本直接赋值
            self.states = s  # 存入状态
            self.actions = a  # 存入动作
            self.rewards = r  # 存入奖励
            self.next_states = next_s  # 存入下一状态
            self.curr_lens = self.states.shape[0]  # 更新样本计数

        elif self.curr_lens <= self.memory_size:  # 池未满：新样本纵向拼接追加
            self.states = np.concatenate((self.states, s),axis=0)  # 拼接状态
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)  # 拼接下一状态
            self.actions = np.concatenate((self.actions, a), axis=0)  # 拼接动作
            self.rewards = np.concatenate((self.rewards, r), axis=0)  # 拼接奖励
            self.curr_lens = self.states.shape[0]  # 更新样本计数
        else:  # 池已满：不再扩容，随机覆盖一段等长旧样本（近似均匀淘汰）
            new_sample_lens = s.shape[0]  # 本批新样本条数
            index = random.randint(0, self.curr_lens - new_sample_lens)  # 随机选覆盖起点（保证不越界）

            self.states[index:(index + new_sample_lens)] = s  # 覆盖对应段的旧状态
            self.actions[index:(index + new_sample_lens)] = a  # 覆盖旧动作
            self.rewards[index:(index + new_sample_lens)] = r  # 覆盖旧奖励
            self.next_states[index:(index + new_sample_lens)] = next_s  # 覆盖旧下一状态

    # 随机抽一批训练样本；池中样本不足一批时整池返回
    def sample(self):

        if self.curr_lens <= self.batch_size:  # 样本数不超过批大小
            return [self.states, self.actions, self.rewards, self.next_states]  # 直接返回整池数据
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)  # 不放回地随机抽 batch_size 个行下标
        batch_s = self.states[indices]  # 按下标取状态批
        batch_a = self.actions[indices]  # 按下标取动作批
        batch_r = self.rewards[indices]  # 按下标取奖励批
        batch_next_s = self.next_states[indices]  # 按下标取下一状态批
        return [batch_s, batch_a, batch_r, batch_next_s]  # 返回 [状态, 动作, 奖励, 下一状态]

    # 清空回放池
    def reset(self):
        self.states = []  # 状态清空
        self.actions = []  # 动作清空
        self.rewards = []  # 奖励清空
        self.next_states = []  # 下一状态清空
        self.curr_lens = 0  # 计数归零


# 模型参数复制器：把主 Q 网络的参数定期复制到 target 网络。
# target network（参数定期从主网络复制、更新滞后的稳定目标网络）：用它算 Bellman 目标可防止训练发散
class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """

    # 建图：按变量名配对两个网络的参数，生成一组 assign 赋值操作（此时尚未执行）
    # 参数：estimator1 = 参数来源（主网络）；estimator2 = 参数去向（target 网络）
    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]  # 收集主网络命名空间下的全部可训练变量
        e1_params = sorted(e1_params, key=lambda v: v.name)  # 按变量名排序，保证两个网络逐一对齐
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]  # 收集 target 网络命名空间下的全部可训练变量
        e2_params = sorted(e2_params, key=lambda v: v.name)  # 同样按变量名排序

        self.update_ops = []  # 存放全部赋值操作
        for e1_v, e2_v in zip(e1_params, e2_params):  # 按顺序配对两个网络的同名变量
            op = e2_v.assign(e1_v)  # 生成"把主网络变量值赋给 target 网络变量"的计算图操作
            self.update_ops.append(op)  # 加入操作列表

    # 执行复制：在会话中运行全部赋值操作，完成一次参数同步
    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)  # 一次 run 执行所有 assign，把主网络参数拷贝到 target 网络
