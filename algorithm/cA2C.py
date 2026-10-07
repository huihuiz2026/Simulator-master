

# =============================================================================
# cA2C（contextual Advantage Actor-Critic，带上下文的优势演员-评论家算法）实现
# 出自 KDD 2018 论文《Efficient Large-Scale Fleet Management via Multi-Agent
# Deep Reinforcement Learning》，是论文中效果最好的调度算法。核心思想：
#   - critic（价值网络：估计当前状态的长期回报预期）给每个网格估值；
#   - actor（策略网络：输入状态、输出各动作的概率分布）决定把空闲司机派往哪里；
#   - 优势函数 A(s,a)=实际回报-critic 估值：衡量动作比平均好多少，
#     指导策略梯度（按"让好动作概率变大"的方向调整参数）更新；
#   - 全城网格共享同一套网络参数，状态里拼入全城上下文（对比 IDQN 每格一套网络）。
# 动作空间：6 个相邻网格方向 + 原地不动，共 7 个动作。
# 本文件包含：Estimator（网络与训练）、stateProcessor（状态预处理）、
# policyReplayMemory / ReplayMemory（经验回放池）、ModelParametersCopier（参数复制器）。
# 调用方式见 run/run_cA2C.py；代码为 TF1 session（会话）风格，经 tf_compat 兼容层跑在 TF2 上。
# =============================================================================

# 导入 NumPy 数值计算库并简写为 np：矩阵/数组的构造与运算全靠它
import numpy as np
# 经兼容层导入 TensorFlow 并简写为 tf：tf_compat 内部用 tensorflow.compat.v1 并关闭
# TF2 行为，让下面 TF1 风格的 session（会话：先建计算图、再执行）代码能跑在 TF2 上
from .tf_compat import tf
# 导入标准库 random（随机抽样）与 os（目录操作）
import random, os
# 从同目录的 alg_utility.py 导入全部公共组件：全连接层 fc、状态/奖励转换工具等
from .alg_utility import *
# 从标准库导入 deepcopy：完整复制对象（含底层数据），而非只复制引用
from copy import deepcopy

# ============ 网络类：一个 Estimator 同时容纳 critic 价值网络和 actor 策略网络 ============
# Estimator（估计器）：搭建并训练 cA2C 的两张网络——
# critic 价值网络（3 层全连接，输出每个网格状态的估值 V(s)）与
# actor 策略网络（3 层全连接，输出 7 个动作的概率分布），
# 两者参数同在本类的 scope 命名空间下，但各自有独立的损失和训练节点。
class Estimator:
    """ build value network
    """
    # 初始化：建好两张网络的计算图、TensorBoard 日志，并预计算邻居列表与有效动作掩码
    # 参数 sess：TensorFlow 会话（执行计算图的入口）
    # 参数 action_dim：动作维度 = 7（6 个邻居方向 + 原地不动）
    # 参数 state_dim：状态维度 = 3 x 有效网格数 + 144（上下文 + 时间 + 网格身份）
    # 参数 env：CityReal 城市仿真环境（提供网格拓扑与邻居关系）
    # 参数 scope：变量命名空间前缀，用于隔离不同 Estimator 的网络参数
    # 参数 summaries_dir：TensorBoard 日志根目录；None 表示不写日志
    def __init__(self,
                 sess,
                 action_dim,
                 state_dim,
                 env,
                 scope="estimator",
                 summaries_dir=None):
        self.sess = sess  # 保存会话引用，后续 sess.run() 执行计算图节点时使用
        self.n_valid_grid = env.n_valid_grids  # 有效网格（可调度网格）总数
        self.action_dim = action_dim  # 动作维度（6 个方向 + 原地不动 = 7）
        self.state_dim = state_dim  # 每个网格状态的维度
        self.M = env.M  # 城市地图行数
        self.N = env.N  # 城市地图列数
        self.scope = scope  # 本网络的命名空间前缀
        self.T = 144  # 一天的时间步数：144 步 x 10 分钟 = 24 小时
        self.env = env  # 保存环境引用（action() 里查网格邻居关系时用）

        # Writes Tensorboard summaries to disk
        self.summary_writer = None  # TensorBoard（TF 自带的训练可视化工具）日志写入器，先置空
        with tf.variable_scope(scope):  # 进入变量命名空间：此后创建的变量名都带 scope 前缀

            # Build the value function graph
            # with tf.variable_scope("value"):
            # 搭建 critic 价值网络（变量直接挂在 scope 下，没有再套子命名空间），返回价值损失
            value_loss = self._build_value_model()

            with tf.variable_scope("policy"):  # 在 "policy" 子命名空间下搭建 actor 策略网络
                actor_loss, entropy = self._build_mlp_policy()  # 返回策略损失和动作分布的熵

            # 合并损失 = 策略损失 + 0.5 x 价值损失 - 10 x 熵（熵正则：鼓励动作分布
            # 不过早收敛、保持探索）。注意：本文件实际训练走上面两个独立的 train_op，
            # 这个合并损失只是被构建出来，没有被任何优化器使用
            self.loss = actor_loss + .5 * value_loss - 10 * entropy


            # self.loss_gradients = tf.gradients(self.value_loss, tf.trainable_variables(scope=scope))
                                           # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

        # Summaries for Tensorboard
        # 合并价值网络的 TensorBoard 标量摘要（update_value 训练时随每次更新写入日志）
        self.summaries = tf.summary.merge([
            tf.summary.scalar("value_loss", self.value_loss),  # 记录价值损失（估值与目标的均方误差）
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),  # 记录估值输出的批量均值
            # tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

        # 合并策略网络的 TensorBoard 标量摘要（update_policy 训练时随每次更新写入日志）
        self.policy_summaries = tf.summary.merge([
            tf.summary.scalar("policy_loss", self.policy_loss),  # 记录策略损失
            tf.summary.scalar("adv", tf.reduce_mean(self.tfadv)),  # 记录优势值的批量均值
            tf.summary.scalar("entropy", self.entropy),  # 记录动作分布的熵（探索程度）
            # tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

        if summaries_dir:  # 若指定了日志目录，则创建 TensorBoard 事件文件写入器
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))  # 按命名空间拼子目录路径
            if not os.path.exists(summary_dir):  # 子目录不存在时
                os.makedirs(summary_dir)  # 递归创建目录
            self.summary_writer = tf.summary.FileWriter(summary_dir)  # 创建日志写入器

        # 预计算"邻居列表"：neighbors_list[i] = [网格 i 的 1 跳邻居在 target_grids 中的
        # 下标列表，再追加 i 自己（对应"原地不动"动作）]；下标同时就是状态向量里的位置
        self.neighbors_list = []
        for idx, node_id in enumerate(env.target_grids):  # idx：网格下标；node_id：网格的节点编号
            # 取该网格第 0 层（1 跳）有效邻居的节点编号列表
            neighbor_indices = env.nodes[node_id].layers_neighbors_id[0]  # index in env.nodes
            # 把每个邻居的节点编号换算成 target_grids 下标（即状态向量中的位置）
            neighbor_ids = [env.target_grids.index(env.nodes[item].get_node_index()) for item in neighbor_indices]
            neighbor_ids.append(idx)  # 末尾追加自己的下标（代表"原地不动"这个动作）
            # index in env.target_grids == index in state
            self.neighbors_list.append(neighbor_ids)  # 收集该网格的邻居下标列表

        # compute valid action mask.
        # 预计算三张"网格 x 动作"查找表：有效动作掩码 + 动作到目标网格的两套编号映射
        self.valid_action_mask = np.ones((self.n_valid_grid, self.action_dim))  # 掩码初始全 1（所有动作可用），不可用处置 0
        # 动作 -> 目标网格的节点编号（env.nodes 体系），最后一列（原地不动）指向自己
        self.valid_neighbor_node_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.nodes
        # 动作 -> 目标网格在 target_grids 中的下标（状态体系），最后一列（原地不动）指向自己
        self.valid_neighbor_grid_id = np.zeros((self.n_valid_grid, self.action_dim))  # id in env.target_grids
        for grid_idx, grid_id in enumerate(env.target_grids):  # 遍历每个有效网格（下标 + 节点编号）
            for neighbor_idx, neighbor in enumerate(self.env.nodes[grid_id].neighbors):  # 遍历 6 个方向上的邻居（边界方向为 None）
                if neighbor is None:  # 该方向出界或邻居网格无效
                    self.valid_action_mask[grid_idx, neighbor_idx] = 0  # 把该方向标记为不可选
                else:  # 该方向有有效邻居
                    node_index = neighbor.get_node_index()  # node_index in env.nodes
                    self.valid_neighbor_node_id[grid_idx, neighbor_idx] = node_index  # 记录该方向的邻居节点编号
                    self.valid_neighbor_grid_id[grid_idx, neighbor_idx] = env.target_grids.index(node_index)  # 记录该方向的邻居网格下标

            self.valid_neighbor_node_id[grid_idx, -1] = grid_id  # 最后一列（动作 6 = 原地不动）= 自己的节点编号
            self.valid_neighbor_grid_id[grid_idx, -1] = grid_idx  # 最后一列（动作 6 = 原地不动）= 自己的网格下标

    # 搭建 critic 价值网络：3 层全连接 + 1 个输出节点，估计每个网格状态的长期回报 V(s)；
    # 训练方式是回归——让输出尽量接近喂进来的 TD 目标 y（时序差分目标：
    # 即时奖励 + 折扣 x 下一状态估值，由外部 compute_targets 算好传入）
    def _build_value_model(self):

        # 状态输入占位符：placeholder 是计算图的输入接口，形状 [批量大小, state_dim]，
        # 运行会话时才喂入真实数据；同时绑定到 self.state 和局部名 X
        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # The TD target value
        # TD 目标占位符：价值网络的回归目标 y，形状 [批量, 1]（"标准答案"）
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")

        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")  # 学习率占位符（标量），训练时可动态调整

        # 3 layers feed forward network.
        # 3 层前馈全连接网络：fc() 是 alg_utility.py 里的全连接层封装
        # （计算"输入 x 权重 + 偏置"再过激活函数），ReLU 激活函数会把负值置 0
        l1 = fc(X, "l1", 128, act=tf.nn.relu)  # 第 1 层：128 个神经元
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)  # 第 2 层：64 个神经元
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)  # 第 3 层：32 个神经元
        # l1 = tf.layers.dense(X, 1024, tf.nn.sigmoid, trainable=trainable)
        # l2 = tf.layers.dense(l1, 512, tf.nn.sigmoid, trainable=trainable)
        # l3 = tf.layers.dense(l2, 32, tf.nn.sigmoid, trainable=trainable)
        # 输出层：1 个神经元、ReLU 激活 -> 每行状态一个估值 V(s) >= 0（奖励是订单价格，非负）
        self.value_output = fc(l3, "value_output", 1, act=tf.nn.relu)

        # self.losses = tf.square(self.y_pl - self.value_output)
        # 价值损失 = 估值与 TD 目标之差的平方的均值（均方误差，标准回归损失）
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))

        # 价值网络的训练节点：Adam 优化器（一种自适应学习率的梯度下降算法）
        # 按传入的学习率占位符最小化价值损失
        self.value_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.value_loss)

        return self.value_loss  # 返回价值损失张量（供 __init__ 合并损失用）

    # 搭建 actor 策略网络：与价值网络同构的 3 层全连接网络（但参数独立），
    # 输出 7 个动作的打分，经掩码 + softmax 变成概率分布；
    # 损失 = 优势加权的"选中动作负对数概率"，实现策略梯度更新
    def _build_mlp_policy(self):

        # 策略网络的状态输入占位符（与价值网络的状态格式相同，但占位符各自独立）
        self.policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="P")
        # 动作输入占位符：one-hot 编码（只有选中位置为 1、其余为 0）的矩阵 [批量, 7]
        self.ACTION = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="action")
        # 优势函数输入占位符：A(s,a)=实际回报-critic 估值，衡量动作比平均好多少
        self.tfadv = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')
        # 邻居掩码输入占位符：[批量, 7]，1=动作可选、0=动作被禁止（喂入后 masking 掉）
        self.neighbor_mask = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="neighbormask")
        # this mask filter invalid actions and those action smaller than current grid value.

        # 3 层全连接网络（结构与价值网络相同，但变量在 "policy" 命名空间下、参数独立）
        l1 = fc(self.policy_state, "l1", 128, act=tf.nn.relu)  # 第 1 层：128 个神经元
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)  # 第 2 层：64 个神经元
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)  # 第 3 层：32 个神经元

        # 输出层得到各动作的打分（logits，ReLU 激活后非负），整体再 +1：
        # 避免被掩码清零后所有打分都是 0，导致下一步取对数出现数值问题
        self.logits = logits = fc(l3, "logits", self.action_dim, act=tf.nn.relu) + 1  # avoid valid_logits are all zeros
        # 用邻居掩码把不可选动作的打分直接置 0
        self.valid_logits = logits * self.neighbor_mask

        # 手工实现的"带掩码 softmax"：先对打分取对数（被掩码的位置约为 log(1e-8)，
        # 是很小的负数），再 softmax（把一组数归一化成和为 1 的概率分布）——
        # 效果是被禁止的动作概率接近 0
        self.softmaxprob = tf.nn.softmax(tf.log(self.valid_logits + 1e-8))
        # 概率的对数版本 log(p)：策略梯度公式中需要 log(选中动作的概率)
        self.logsoftmaxprob = tf.nn.log_softmax(self.softmaxprob)

        # 每个动作的负对数概率 -log(p)；与 one-hot 动作相乘即只剩"选中动作的 -log(p)"
        self.neglogprob = - self.logsoftmaxprob * self.ACTION
        # actor 损失 = 优势加权的负对数概率（对动作维求和、对批量求均值）：
        # 优势为正（好动作）-> 最小化 -log(p) 等于提高该动作概率；优势为负则降低
        self.actor_loss = tf.reduce_mean(tf.reduce_sum(self.neglogprob * self.tfadv, axis=1))
        # 动作分布的熵 H = -sum(p*log(p))：值越大分布越"随机"、探索越充分
        self.entropy = - tf.reduce_mean(self.softmaxprob * self.logsoftmaxprob)

        # 策略总损失 = actor 损失 - 0.01 x 熵（熵正则项：奖励高熵，
        # 鼓励保持探索、不过早收敛到单一动作）
        self.policy_loss = self.actor_loss - 0.01 * self.entropy

        # 策略网络的训练节点：Adam 优化器最小化策略损失（即按策略梯度方向更新参数）
        self.policy_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.policy_loss)
        return self.actor_loss, self.entropy  # 返回 actor 损失和熵（供 __init__ 合并损失用）

    # 批量预测状态估值：把状态矩阵喂给价值网络，返回每行一个估值 V(s)
    def predict(self, s):
        # 在会话中执行价值输出节点：{占位符: 数据} 指定喂入内容
        value_output = self.sess.run(self.value_output, {self.state: s})

        return value_output  # 估值矩阵，形状 [len(s), 1]

    # cA2C 的决策入口：为全城每个网格的空闲司机生成调度动作，同时记录训练数据。
    # 流程：1) 价值网络给所有网格估值；2) 按估值收紧动作掩码（不值得去的方向不派车）；
    # 3) 策略网络算出各方向概率，为每个可调度司机独立采样一个方向；
    # 4) 汇总成 (起点, 终点, 司机数) 调度指令，并记下策略梯度训练所需数据
    # 参数 s：全部网格的状态矩阵 [有效网格数, state_dim]（每行一个网格）
    # 参数 context：每个网格可调度的空闲司机数（订单匹配后仍剩余的司机）
    # 参数 epsilon：传入但本方法内未使用（保留的接口参数）
    # 返回 7 项：调度动作列表、各网格动作概率、训练用状态/one-hot 动作/当前估值/掩码/目的网格下标
    def action(self, s, context, epsilon):
        """ Compute current action for all grids give states

        :param s: 504 x stat_dim,
        :return:
        """

        # 价值网络给所有网格批量估值，flatten 成一维（每个有效网格一个估值）
        value_output = self.sess.run(self.value_output, {self.state: s}).flatten()
        action_tuple = []  # 调度动作列表，元素为 (起点节点编号, 终点节点编号, 司机数)
        valid_prob = []  # 每个网格的动作概率行（供 compute_targets 计算价值网络的回归目标）

        # for training policy gradient.
        # 以下四个列表记录策略梯度训练所需的逐条样本
        action_choosen_mat = []  # 每条调度记录的 one-hot 动作向量
        policy_state = []  # 每条调度记录对应的网格状态行
        curr_state_value = []  # 每条调度记录出发网格的当前估值 V(s)
        next_state_ids = []  # 每条调度记录的目的网格下标（用于计算优势）

        # 从每行状态末尾的网格身份 one-hot 段（最后 n_valid_grid 列）还原出网格下标
        grid_ids = np.argmax(s[:, -self.n_valid_grid:], axis=1)

        # compute neighbor mask according to centralized value
        # 按估值动态收紧动作掩码：先深拷贝一份静态有效掩码（deepcopy 保证改动不影响原表）
        curr_neighbor_mask = deepcopy(self.valid_action_mask)
        for idx, grid_valid_idx in enumerate(grid_ids):  # 逐网格处理（idx：行号；grid_valid_idx：网格下标）
            # 取"自己 + 1 跳邻居"这些网格的估值
            valid_qvalues = value_output[self.neighbors_list[grid_valid_idx]]  # value of current and its nearby grids
            temp_qvalue = np.zeros(self.action_dim)  # 7 维临时数组：每个动作位填目的网格的估值
            # 把"自己+邻居"的估值按有效动作位填入（方向顺序与掩码一一对应，
            # 最后一列"原地不动"正好填自己的估值）
            temp_qvalue[curr_neighbor_mask[grid_valid_idx] > 0] = valid_qvalues
            temp_qvalue[temp_qvalue < temp_qvalue[-1]] = 0  # 把估值低于"留在本格"的方向打分清零
            # 再据此把这些方向从动态掩码中剔除（前一截方括号索引返回的是视图，
            # 赋值会写回 curr_neighbor_mask）：只保留估值不低于本格的方向，
            # 即不往"预期回报更差"的网格派车
            curr_neighbor_mask[grid_valid_idx][np.where(temp_qvalue < temp_qvalue[-1])] = 0
            if np.sum(curr_neighbor_mask[grid_valid_idx]) == 0:  # 若一个方向都不剩（本格估值最低）
                curr_neighbor_mask[grid_valid_idx] = self.valid_action_mask[grid_valid_idx]  # 回退为静态掩码，保证至少"原地不动"可选

        # compute policy probability.
        # 把状态 + 动态掩码喂给策略网络，得到每个网格的动作概率分布（被掩码方向概率约等于 0）
        action_probs = self.sess.run(self.softmaxprob, {self.policy_state: s,
                                                        self.neighbor_mask: curr_neighbor_mask})

        curr_neighbor_mask_policy = []  # 每条调度记录的掩码行（训练策略网络时喂回 neighbor_mask 占位符）
        # sample action.
        # 逐网格按概率采样动作：为每个可调度司机独立抽一个方向
        for idx, grid_valid_idx in enumerate(grid_ids):
            action_prob = action_probs[idx]  # 取出该网格的动作概率行

            # cast invalid action to zero, avlid numerical issue.
            action_prob[self.valid_action_mask[grid_valid_idx] == 0] = 0  # 把结构上无效的方向概率精确置 0（消除 softmax 残差）
            # 各网格的概率行都要记录（供价值网络算目标），无论是否有司机可派
            valid_prob.append(action_prob)   # action probability for state value function
            if int(context[idx]) == 0:  # 该网格没有可派出的空闲司机
                continue  # 跳过采样，处理下一个网格

            # 按概率分布为该网格的每个空闲司机独立抽一个动作（共抽 context[idx] 次），
            # 概率先除以总和再归一化，避免残差导致概率和略微超过 1
            curr_action_indices_temp = np.random.choice(self.action_dim, int(context[idx]),
                                                        p=action_prob/np.sum(action_prob))
            # num of drivers dispatched to nearby locations [2,3,2,3,1,3,3]
            # for numerically stable, avoid sum of action_prob > 1 with small value

            curr_action_indices = [0] * self.action_dim  # 每个方向的司机数计数器，初始全 0
            for kk in curr_action_indices_temp:  # 遍历每个司机的抽样结果
                curr_action_indices[kk] += 1  # 对应方向的司机数 +1

            start_node_id = self.env.target_grids[grid_valid_idx]  # 本网格的节点编号（调度起点）
            for curr_action_idx, num_driver in enumerate(curr_action_indices):  # 遍历 7 个方向的司机数
                if num_driver > 0:  # 该方向有司机要派
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx, curr_action_idx])  # 查表得到目标节点编号
                    if end_node_id != start_node_id:  # 目标不是自己（原地不动的司机无需生成调度指令）
                        action_tuple.append((start_node_id, end_node_id, num_driver))  # 记录调度动作 (起点, 终点, 司机数)

                    # book keeping for training
                    # 为训练做记录（注意：原地不动也会被记录——它同样是策略采样出的一个动作）
                    temp_a = np.zeros(self.action_dim)  # 构造 one-hot 动作向量
                    temp_a[curr_action_idx] = 1  # 在选中动作位置置 1
                    action_choosen_mat.append(temp_a)  # 记录 one-hot 动作
                    policy_state.append(s[idx])  # 记录该网格的状态行
                    curr_state_value.append(value_output[idx])  # 记录出发网格的当前估值
                    next_state_ids.append(self.valid_neighbor_grid_id[grid_valid_idx, curr_action_idx])  # 记录目的网格下标
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[idx])  # 记录该网格的动态掩码行

        # 返回 7 项：调度动作列表、各网格概率矩阵、训练用（状态、one-hot 动作、
        # 当前估值、动态掩码、目的网格下标）；其中估值和目的网格下标保持为普通列表
        return action_tuple, np.stack(valid_prob), \
               np.stack(policy_state), np.stack(action_choosen_mat), curr_state_value, \
               np.stack(curr_neighbor_mask_policy), next_state_ids  # 最后一项是目的网格下标列表

    # 计算优势函数（供策略网络训练用）：对每条调度记录算
    # A = 即时奖励(目的网格) + 折扣因子 x V(目的网格) - V(出发网格)，
    # 即"一步 TD 优势"——实际回报比 critic 当时的预期好多少，好则为正、差则为负
    # 参数 curr_state_value：每条记录出发网格的估值 V(s)（action() 里记录的）
    # 参数 next_state_ids：每条记录的目的网格下标
    # 参数 next_state：下一时刻全部网格的状态矩阵（每行对应一个网格，行序=网格序）
    # 参数 node_reward：每个有效网格的即时奖励（人均值）
    # 参数 gamma：折扣因子（0~1，未来回报打几折，越小越看重眼前）
    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, gamma):
        """for policy network"""
        advantage = []  # 优势值列表：每条调度记录一个
        node_reward = node_reward.flatten()  # 奖励展平成一维，便于按下标取值
        # 用价值网络给下一时刻每个网格估值（next_state 每行对应一个网格）
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()

        for idx, next_state_id in enumerate(next_state_ids):  # 逐条调度记录计算
            next_state_id = int(next_state_id)  # 目的网格下标转成 int
            # 一步 TD 优势：r(目的网格) + gamma x V(目的网格) - V(出发网格)
            temp_adv = node_reward[next_state_id] + gamma * qvalue_next[next_state_id] - curr_state_value[idx]
            advantage.append(temp_adv)  # 收集该条记录的优势值
        return advantage  # 返回优势列表（与调度记录一一对应）

    # 计算价值网络的回归目标（TD 目标）：对每个网格，按策略给各动作的概率，
    # 对"邻居即时奖励 + 折扣 x 邻居估值"加权求和——即按当前策略走一步的期望回报
    # 参数 valid_prob：每个网格的动作概率行（action() 返回的第 2 项）
    # 参数 next_state：下一时刻全部网格的状态矩阵
    # 参数 node_reward：每个有效网格的即时奖励（人均值）
    # 参数 gamma：折扣因子
    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        targets = []  # 每个网格一个回归目标
        node_reward = node_reward.flatten()  # 奖励展平成一维
        # 下一时刻每个网格的估值 V(s')（每行对应一个网格）
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()

        for idx in np.arange(self.n_valid_grid):  # 遍历全部有效网格
            grid_prob = valid_prob[idx][self.valid_action_mask[idx]>0]  # 取该网格在有效动作上的概率
            neighbor_grid_ids = self.neighbors_list[idx]  # 该网格的邻居（含自己）下标列表，与动作一一对应
            # TD 目标 = sum( 概率 x (邻居奖励 + gamma x 邻居估值) )：按策略动作分布加权的期望一步回报
            curr_grid_target = np.sum(grid_prob * (node_reward[neighbor_grid_ids] + gamma * qvalue_next[neighbor_grid_ids]))
            # assert np.sum(grid_prob) == 1 numerical issue.
            targets.append(curr_grid_target)  # 收集该网格的目标值

        return np.array(targets).reshape([-1, 1])  # 整理成 [网格数, 1] 列向量，与 y_pl 占位符形状一致

    # 价值网络的预热/初始化训练：执行一步 Adam 更新并返回损失
    # （可在正式训练前先用它拟合价值网络；run_cA2C 主流程未调用此方法）
    # 参数 s：批量状态；y：回归目标；learning_rate：学习率
    def initialization(self, s, y, learning_rate):
        sess = self.sess  # 取会话引用
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}  # 准备输入：状态、目标、学习率
        _, value_loss = sess.run([self.value_train_op, self.value_loss], feed_dict)  # 执行一次价值网络训练并取回损失
        return value_loss  # 返回本批损失

    # 策略网络的一次训练：把采样到的（状态, 动作, 优势, 掩码）批量喂给计算图，
    # 执行策略梯度更新（让高优势的动作概率变大、低优势的变小）
    # 参数 policy_state：调度记录的状态批量；advantage：优势值列向量；
    # 参数 action_choosen_mat：one-hot 动作批量；curr_neighbor_mask：动作掩码批量；
    # 参数 learning_rate：学习率；global_step：日志用的全局步数
    def update_policy(self, policy_state, advantage, action_choosen_mat, curr_neighbor_mask, learning_rate, global_step):
        sess = self.sess  # 取会话引用
        feed_dict = {self.policy_state: policy_state,  # 状态批量
                     self.tfadv: advantage,  # 优势值
                     self.ACTION: action_choosen_mat,  # one-hot 动作
                     self.neighbor_mask: curr_neighbor_mask,  # 动作掩码
                     self.loss_lr: learning_rate}  # 学习率
        summaries, _, loss = sess.run([self.policy_summaries, self.policy_train_op, self.policy_loss], feed_dict)  # 执行策略梯度更新，同时取回摘要与损失

        if self.summary_writer:  # 若配置了 TensorBoard 写入器
            self.summary_writer.add_summary(summaries, global_step)  # 把本步摘要写入日志
            self.summary_writer.flush()  # 立即刷盘
        return loss  # 返回本批策略损失

    # 价值网络的一次训练：向 TD 目标回归（最小化估值与目标的均方误差）
    # 参数 s：批量状态；y：TD 目标；learning_rate：学习率；global_step：日志步数
    def update_value(self, s, y, learning_rate, global_step):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, state_dim]
          a: Chosen actions of shape [batch_size, action_dim], 0, 1 mask
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        sess = self.sess  # 取会话引用
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}  # 准备输入：状态、TD 目标、学习率
        summaries, _, loss = sess.run([self.summaries, self.value_train_op, self.value_loss], feed_dict)  # 执行价值网络训练，同时取回摘要与损失

        if self.summary_writer:  # 若配置了 TensorBoard 写入器
            self.summary_writer.add_summary(summaries, global_step)  # 把本步摘要写入日志
            self.summary_writer.flush()  # 立即刷盘
        return loss  # 返回本批价值损失




# ================== 状态预处理器：把环境原始观测加工成网络的输入 ==================
# stateProcessor（状态处理器）：把仿真环境的全局状态 [空闲司机分布, 订单分布]
# 转换成"每个网格一行"的状态向量（cA2C 的输入），并做奖励、上下文的格式转换。
# 这里也体现了"上下文（contextual）"：每行状态都包含全城司机/订单分布 + 时间 + 自己的身份
class stateProcessor:
    """
        Process a raw global state into the states of grids.
    """

    # 参数 target_id_states：全局状态展平后各有效网格分量的下标
    #   （全局状态 = [MxN 司机分布, MxN 订单分布] 展平，每个网格占"编号"和"编号+MxN"两个下标）
    # 参数 target_grids：有效网格的节点编号列表（真实数据 504 个）
    # 参数 n_valid_grids：有效网格数量
    def __init__(self,
                 target_id_states,
                 target_grids,
                 n_valid_grids):
        # 有效网格在展平全局状态中的下标表（用于挑出司机数和订单数分量）
        self.target_id_states = target_id_states  # valid grid index for driver and order distribution.
        # 有效网格的节点编号列表（环境 env.nodes 体系）
        self.target_grids = target_grids   # valid grid id [22, 24, ...]  504 grids
        self.n_valid_grids = n_valid_grids  # 有效网格数量
        self.T = 144  # 一天的时间步数（144 x 10 分钟 = 24 小时）
        self.action_dim = 7  # 动作维度：6 个邻居方向 + 原地不动
        self.extend_state = True  # 标志位：是否使用扩展状态（本文件内未参与任何逻辑）

    # 从全局状态中挑出有效网格分量：输入 [司机分布, 订单分布]（各 MxN），
    # 返回长度 2 x 有效网格数的向量 = [各有效网格空闲司机数, 各有效网格订单数]
    def utility_conver_states(self, curr_state):
        curr_s = np.array(curr_state).flatten()  # 全局状态展平成一维数组
        curr_s_new = [curr_s[idx] for idx in self.target_id_states]  # 按预存下标挑出有效网格的司机数和订单数
        return np.array(curr_s_new)  # 转成 numpy 数组返回

    # 归一化状态：司机数段除以全城最大单格司机数、订单数段除以最大单格订单数，
    # 两段都压到 [0,1] 区间，让神经网络输入量级稳定、便于学习
    def utility_normalize_states(self, curr_s):
        max_driver_num = np.max(curr_s[:self.n_valid_grids])  # 全城最大的单格空闲司机数
        max_order_num = np.max(curr_s[self.n_valid_grids:])  # 全城最大的单格订单数
        if max_order_num == 0:  # 全城没有订单
            max_order_num = 1  # 除数置 1，避免除零
        if max_driver_num == 0:  # 全城没有空闲司机
            max_driver_num = 1  # 除数置 1，避免除零
        curr_s_new = np.zeros_like(curr_s)  # 新建同形状数组存放归一化结果
        curr_s_new[:self.n_valid_grids] = curr_s[:self.n_valid_grids] / max_driver_num  # 司机数段归一化
        curr_s_new[self.n_valid_grids:] = curr_s[self.n_valid_grids:] / max_order_num  # 订单数段归一化
        return curr_s_new  # 返回归一化后的状态

    # 从覆盖全部节点的奖励向量中挑出有效网格的奖励
    def utility_conver_reward(self, reward_node):
        reward_node_new = [reward_node[idx] for idx in self.target_grids]  # 按有效网格编号挑出对应奖励
        return np.array(reward_node_new)  # 转成 numpy 数组返回

    # 把环境返回的奖励重排成模型需要的格式：取有效网格的奖励，再除以该格空闲司机数，
    # 得到"平均每个司机的奖励"（司机多的网格分摊，避免奖励被司机数大的网格主导）
    # 参数 info：env.step 返回的信息 [node_reward(含邻居广播贡献), neighbor_reward]
    # 参数 curr_s：同一步经 utility_conver_states 处理的状态（提供司机数做分母）
    def reward_wrapper(self, info, curr_s):
        """ reformat reward from env to the input of model.
        :param info: [node_reward(including neighbors), neighbor_reward]
        :param curr_s:  processed by utility_conver_states, same time step as info.
        :return:
        """

        info_reward = info[0]  # 取第一部分：各节点奖励（含邻居广播的贡献）
        valid_nodes_reward = self.utility_conver_reward(info_reward[0])  # 挑出有效网格的奖励
        devide = curr_s[:self.n_valid_grids]  # 各有效网格的空闲司机数（做分母；注意这是 curr_s 的视图）
        devide[devide == 0] = 1  # 司机数为 0 的格置 1，避免除零
        # 逐格相除得到人均奖励
        valid_nodes_reward = valid_nodes_reward/devide  # averaged rewards for drivers arriving this grid
        return valid_nodes_reward  # 返回各有效网格的人均奖励

    # 计算调度上下文：从环境的"剩余司机/剩余订单"信息中取出各有效网格的剩余空闲司机数，
    # 即每个网格有多少司机没被订单匹配、可以被调度去别的网格
    # 参数 info：[剩余司机矩阵, 剩余订单矩阵]（各 MxN，来自 env 的订单预匹配步骤）
    def compute_context(self, info):
        # compute context
        context = info.flatten()  # 把 [剩余司机, 剩余订单] 展平成 [司机段(MN), 订单段(MN)] 的一维向量
        context = [context[idx] for idx in self.target_grids]  # 前半段按网格编号取值 -> 各有效网格的剩余司机数
        return context  # 返回每个有效网格可调度的空闲司机数

    # 把归一化全局状态扩展成"每个网格一行"的状态矩阵（cA2C 输入的最终形态）：
    # 每行 = [全城司机分布+订单分布(2n 维，所有行相同) | 时间 one-hot(144 维) | 自己的身份 one-hot(n 维)]
    # 参数 curr_s：归一化后的全局状态（长度 2n：n 个司机数 + n 个订单数）
    # 参数 curr_city_time：当前时间步（0~143）
    def to_grid_states(self, curr_s, curr_city_time):
        """ extend global state to all agents' state.

        :param curr_s:
        :param curr_city_time: curr_s time step
        :return:
        """
        T = self.T  # 时间步数局部别名（144）

        # curr_s = self.utility_conver_states(curr_state)
        time_one_hot = np.zeros((T))  # 时间 one-hot 向量：144 维全零
        time_one_hot[curr_city_time % T] = 1  # 当前时刻位置置 1（取模支持时间跨天回绕）
        onehot_grid_id = np.eye(self.n_valid_grids)  # 单位阵：第 i 行即网格 i 的身份 one-hot 向量

        # 状态矩阵 [有效网格数, 3n+144]：n=有效网格数
        s_grid = np.zeros((self.n_valid_grids, self.n_valid_grids * 3 + T))
        s_grid[:, :self.n_valid_grids * 2] = np.stack([curr_s] * self.n_valid_grids)  # 前 2n 列：每行填入同一份全城上下文（司机+订单分布）
        s_grid[:, self.n_valid_grids * 2:self.n_valid_grids * 2 + T] = np.stack([time_one_hot] * self.n_valid_grids)  # 中间 144 列：每行填入同一份时间 one-hot
        s_grid[:, -self.n_valid_grids:] = onehot_grid_id  # 最后 n 列：各网格自己的身份 one-hot（各行不同）

        return np.array(s_grid)  # 返回 [有效网格数, 3n+144] 的状态矩阵

    # 把每个网格的奖励向量整理成 [网格数, 1] 列向量（与价值网络 y_pl 占位符形状一致）
    def to_grid_rewards(self, node_reward):
        """
        :param node_reward: curr_city_time + 1 's reward
        :return:
        """
        return np.array(node_reward).reshape([-1, 1])  # 强制重排成 n 行 1 列

    # 把一串动作下标转成 one-hot 动作矩阵：第 i 行在第 action_neighbor_idx[i] 列置 1
    def to_action_mat(self, action_neighbor_idx):
        action_mat = np.zeros((len(action_neighbor_idx), self.action_dim))  # 全零矩阵 [样本数, 7]
        action_mat[np.arange(action_mat.shape[0]), action_neighbor_idx] = 1  # 花式索引：每行在对应动作列置 1
        return action_mat  # 返回 one-hot 动作矩阵


# ============ 经验回放池 1：策略网络的训练样本缓存 ============
# policyReplayMemory（策略回放池）：经验回放（replay）是强化学习常用技巧——
# 把交互采到的样本存起来，训练时从中随机抽小批量，打破样本间的时间相关性。
# 本池缓存策略网络所需的 (状态, one-hot 动作, 优势值, 动作掩码) 四元组
class policyReplayMemory:
    # 参数 memory_size：容量上限（超过后新数据随机覆盖旧数据）
    # 参数 batch_size：每次训练抽取的小批量大小
    def __init__(self, memory_size, batch_size):
        self.states = []  # 状态缓存
        # self.next_states = []
        self.neighbor_mask = []  # 动作掩码缓存
        self.actions = []  # one-hot 动作缓存
        # 奖励位缓存——本池中实际存的是优势值（advantage），不是原始奖励
        self.rewards = []  # advantages

        self.batch_size = batch_size  # 小批量大小
        self.memory_size = memory_size  # 容量上限
        self.current = 0  # 占位计数器（未使用）
        self.curr_lens = 0  # 当前缓存样本条数

    # 追加一批样本 (s, a, r, mask)：池空时直接存入；未超容量时沿样本维拼接；
    # 超过容量后在池内随机选一段等长位置整体覆盖（简单的旧数据淘汰策略）
    def add(self, s, a, r, mask):
        if self.curr_lens == 0:  # 池还是空的
            self.states = s  # 直接把这批数据作为初始状态缓存
            self.actions = a  # 初始动作缓存
            self.rewards = r  # 初始优势值缓存
            self.neighbor_mask = mask  # 初始掩码缓存
            self.curr_lens = self.states.shape[0]  # 用行数更新样本计数

        elif self.curr_lens <= self.memory_size:  # 未超容量：新样本拼接到末尾
            self.states = np.concatenate((self.states, s),axis=0)  # 拼接状态
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)  # 拼接掩码
            self.actions = np.concatenate((self.actions, a), axis=0)  # 拼接动作
            self.rewards = np.concatenate((self.rewards, r), axis=0)  # 拼接优势值
            self.curr_lens = self.states.shape[0]  # 更新样本计数
        else:  # 已超容量：随机覆盖池中一段等长旧样本
            new_sample_lens = s.shape[0]  # 本批新样本条数
            # random.seed(0)
            index = random.randint(0, self.curr_lens - new_sample_lens)  # 随机选覆盖起点（上界保证不越界）

            self.states[index:(index + new_sample_lens)] = s  # 用新状态覆盖该段旧数据
            self.actions[index:(index + new_sample_lens)] = a  # 覆盖动作
            self.rewards[index:(index + new_sample_lens)] = r  # 覆盖优势值
            self.neighbor_mask[index:(index + new_sample_lens)] = mask  # 覆盖掩码

    # 抽取一个小批量：样本数不超过 batch_size 时整池返回；否则无放回随机抽 batch_size 条
    def sample(self):

        if self.curr_lens <= self.batch_size:  # 样本不足：全部返回（优势值列表转成数组）
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        # random.seed(0)
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)  # 无放回随机抽 batch_size 个行号
        batch_s = self.states[indices]  # 按行号取状态
        batch_a = self.actions[indices]  # 按行号取 one-hot 动作
        batch_r = self.rewards[indices]  # 按行号取优势值
        batch_mask = self.neighbor_mask[indices]  # 按行号取掩码
        return [batch_s, batch_a, batch_r, batch_mask]  # 返回小批量 [状态, 动作, 优势, 掩码]

    # 清空回放池，恢复到初始空状态
    def reset(self):
        self.states = []  # 清空状态
        self.actions = []  # 清空动作
        self.rewards = []  # 清空优势值
        self.neighbor_mask = []  # 清空掩码
        self.curr_lens = 0  # 样本计数归零


# ============ 经验回放池 2：价值网络的训练样本缓存 ============
# ReplayMemory（价值回放池）：结构同 policyReplayMemory，但缓存的是
# (状态, 动作概率矩阵, TD 目标, 下一状态)；run_cA2C 实际只用第 1、3 项
# （状态和 TD 目标）来训练价值网络，动作与下一状态字段存了但未参与训练
class ReplayMemory:
    """ collect the experience and sample a batch for training networks.
        without time ordering
    """
    # 参数 memory_size：容量上限（超过后新数据随机覆盖旧数据）
    # 参数 batch_size：每次训练抽取的小批量大小
    def __init__(self, memory_size, batch_size):
        self.states = []  # 状态缓存
        self.next_states = []  # 下一状态缓存（存了但训练价值网络时未用）
        self.actions = []  # 动作缓存（实际存的是各网格的动作概率矩阵）
        self.rewards = []  # 奖励位缓存（实际存的是 TD 目标值）

        self.batch_size = batch_size  # 小批量大小
        self.memory_size = memory_size  # 容量上限
        self.current = 0  # 占位计数器（未使用）
        # 当前缓存样本条数
        self.curr_lens = 0  # current memory lens

    # 追加一批样本 (s, a, r, next_s)：池空时直接存入；未超容量时沿样本维拼接；
    # 超过容量后在池内随机选一段等长位置整体覆盖（与 policyReplayMemory.add 相同逻辑）
    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:  # 池还是空的
            self.states = s  # 直接把这批数据作为初始状态缓存
            self.actions = a  # 初始动作缓存
            self.rewards = r  # 初始目标值缓存
            self.next_states = next_s  # 初始下一状态缓存
            self.curr_lens = self.states.shape[0]  # 用行数更新样本计数

        elif self.curr_lens <= self.memory_size:  # 未超容量：新样本拼接到末尾
            self.states = np.concatenate((self.states, s),axis=0)  # 拼接状态
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)  # 拼接下一状态
            self.actions = np.concatenate((self.actions, a), axis=0)  # 拼接动作
            self.rewards = np.concatenate((self.rewards, r), axis=0)  # 拼接目标值
            self.curr_lens = self.states.shape[0]  # 更新样本计数
        else:  # 已超容量：随机覆盖池中一段等长旧样本
            new_sample_lens = s.shape[0]  # 本批新样本条数
            # random.seed(0)
            index = random.randint(0, self.curr_lens - new_sample_lens)  # 随机选覆盖起点（上界保证不越界）

            self.states[index:(index + new_sample_lens)] = s  # 用新状态覆盖该段旧数据
            self.actions[index:(index + new_sample_lens)] = a  # 覆盖动作
            self.rewards[index:(index + new_sample_lens)] = r  # 覆盖目标值
            self.next_states[index:(index + new_sample_lens)] = next_s  # 覆盖下一状态

    # 抽取一个小批量：样本数不超过 batch_size 时整池返回；否则无放回随机抽 batch_size 条
    def sample(self):

        if self.curr_lens <= self.batch_size:  # 样本不足：全部返回
            return [self.states, self.actions, self.rewards, self.next_states]
        # random.seed(0)
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)  # 无放回随机抽 batch_size 个行号
        batch_s = self.states[indices]  # 按行号取状态
        batch_a = self.actions[indices]  # 按行号取动作（实际是动作概率矩阵）
        batch_r = self.rewards[indices]  # 按行号取目标值
        batch_mask = self.next_states[indices]  # 按行号取下一状态（变量名叫 mask 是历史遗留，实为 next_states）
        return [batch_s, batch_a, batch_r, batch_mask]  # 返回小批量 [状态, 动作, 目标, 下一状态]

    # 清空回放池，恢复到初始空状态
    def reset(self):
        self.states = []  # 清空状态
        self.actions = []  # 清空动作
        self.rewards = []  # 清空目标值
        self.next_states = []  # 清空下一状态
        self.curr_lens = 0  # 样本计数归零



# ============ 参数复制器：把一个网络的权重复制给另一个网络 ============
# ModelParametersCopier（模型参数复制器）：预先建好一组"赋值"操作节点，
# 调用 make() 时执行它们，即可把 estimator1 的全部参数同步给 estimator2。
# 典型用途是同步目标网络（target network：定期拷贝在线网络参数以稳定训练）；
# cA2C 没有目标网络所以 run_cA2C 未使用，IDQN/cDQN 的脚本会用到
class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """

    # 参数 estimator1：参数来源网络（把它的权重复制出去）
    # 参数 estimator2：参数去向网络（接收 estimator1 的权重）
    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        # 收集 estimator1 命名空间下的全部可训练变量（权重/偏置），按变量名排序保证两边配对
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)  # 按变量名排序
        # 同样收集 estimator2 的可训练变量并排序
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)  # 按变量名排序

        self.update_ops = []  # 复制操作节点列表
        for e1_v, e2_v in zip(e1_params, e2_params):  # 排序后同名变量逐对配对
            op = e2_v.assign(e1_v)  # 建立"把 e1 的当前值赋给 e2"的操作节点（只建图，不执行）
            self.update_ops.append(op)  # 收集该复制操作

    # 在会话中执行全部赋值操作，完成一次参数同步
    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)  # 一次性运行所有 assign 节点
