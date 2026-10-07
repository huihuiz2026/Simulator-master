# 导入 pickle（Python 对象序列化/反序列化模块，用于读数据和保存结果）和 sys（用于修改模块搜索路径）
import pickle, sys
# 导入 pathlib.Path 类：以面向对象方式处理文件路径，跨平台兼容
from pathlib import Path
# 把本文件所在目录的上一级（即项目根目录 Simulator-master）插入 sys.path 最前面，
# 保证无论从哪个目录运行脚本，都能找到根目录下的 simulator、algorithm 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# （已注释）原先单独导入 simulator.utilities，其中的 mkdir_p 等工具已由下方两个星号导入提供
# from simulator.utilities import *
# 星号导入 algorithm.alg_utility：引入 TensorFlow 兼容层 tf（让 TF1 session 风格代码跑在 TF2 上）、
# np（NumPy）、collision_action（统计对向调度冲突）等工具
from algorithm.alg_utility import *
# 星号导入 simulator.envs：引入 CityReal 城市仿真环境类，以及 np、time、mkdir_p 等名字
from simulator.envs import *


################## Load data ###################################
# 外部数据根目录（仓库中不含这些数据文件，运行时需自行准备，此处为加载外部数据）
dir_prefix = "/mnt/research/linkaixi/AllData/dispatch/"
# 当前时间字符串，格式"年月日_时-分"，用作本次实验目录名（time 由上面的星号导入引入）
current_time = time.strftime("%Y%m%d_%H-%M")
# 拼接本次实验的日志目录路径
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
# 创建日志目录（已存在则不报错；mkdir_p 来自 simulator.utilities）
mkdir_p(log_dir)
# 打印日志目录路径
print("log dir is {}".format(log_dir))

# 仿真参数数据目录
data_dir = dir_prefix + "dispatch_realdata/data_for_simulator/"
# 订单时长分布：空列表占位（本脚本使用预载入的真实订单，不走随机生成订单流程）
order_time_dist = []
# 订单价格分布：空列表占位（同上）
order_price_dist = []
# 以下逐个读取外部数据文件（pickle 反序列化；encoding='latin1' 用于兼容 Python 2
# 时代保存的旧格式 pickle 文件），各文件内容见行尾注释
mapped_matrix_int = pickle.load(open(data_dir+"mapped_matrix_int.pkl", 'rb'), encoding='latin1')  # 地图矩阵：-100 为无效网格，正整数为有效网格编号
order_num_dist = pickle.load(open(data_dir+"order_num_dist", 'rb'), encoding='latin1')  # 订单数量分布：144 个时刻、各网格生成订单数的泊松均值
idle_driver_dist_time = pickle.load(open(data_dir+"idle_driver_dist_time", 'rb'), encoding='latin1')  # 全城空闲司机总数分布：144 个时刻的 [均值, 标准差]
idle_driver_location_mat = pickle.load(open(data_dir+"idle_driver_location_mat", 'rb'), encoding='latin1')  # 期望空闲司机位置矩阵：144 时刻 x 有效网格数
target_ids = pickle.load(open(data_dir+"target_grid_id.pkl", 'rb'), encoding='latin1')  # 有效网格编号列表（仅加载，后续主流程未直接使用）
onoff_driver_location_mat = pickle.load(open(data_dir + "onoff_driver_location_mat", 'rb'), encoding='latin1')  # 司机上下线控制矩阵：144 时刻 x 有效网格 x [均值, 标准差]（正偏上线、负偏下线）
order_filename = dir_prefix + "dispatch_realdata/orders/all_orders_target"  # 真实订单数据文件路径（约 4 周的订单）
order_real = pickle.load(open(order_filename, 'rb'), encoding='latin1')  # 真实订单数组：每条 [起点网格, 终点网格, 起始时刻, 时长(步), 价格]
# 从地图矩阵取出行数 M 和列数 N
M, N = mapped_matrix_int.shape
# 打印提示：数据加载完成
print("finish load data")


################## Initialize env ###################################
# 邻域边数：6 表示六边形邻域（每个网格最多 6 个邻居方向）
n_side = 6
# 折扣因子 γ（强化学习里衡量未来回报重要性的系数，0.9 表示下一步的回报打 9 折；本脚本实际用的是后面小写的 gamma，此常量未使用）
GAMMA = 0.9
# 订单/调度可跨越的最大网格层数 l_max=9（同时限定订单时长最多 9 步）
l_max = 9

# 用真实数据构造 CityReal 城市仿真环境（一天 144 步、每步 10 分钟），参数依次为：
# 地图矩阵、订单数量分布参数、空闲司机总数分布、空闲司机位置矩阵、
# 订单时长分布（空占位）、订单价格分布（空占位）、最大跨层数 l_max、行数 M、列数 N、
# 邻域边数 n_side、订单采样概率 1/28（4 周数据平均抽出 1 天的订单量）、
# 预载入的真实订单数组、司机上下线控制矩阵
env = CityReal(mapped_matrix_int, order_num_dist,
               idle_driver_dist_time, idle_driver_location_mat,
               order_time_dist, order_price_dist,
               l_max, M, N, n_side, 1/28.0, order_real, onoff_driver_location_mat)

# 重新拼接日志目录（与前面的值相同，重复赋值）
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)

# 状态矩阵形状为 (2, M, N)：[0] 是空闲司机分布、[1] 是订单分布；展平后前 M*N 个元素
# 属于司机部分、后 M*N 个属于订单部分。target_grids 是有效网格编号（0 ~ M*N-1），
# 加上 M*N 即得到同一批网格在展平状态"订单部分"中的下标
temp = np.array(env.target_grids) + env.M * env.N
# 拼接两个下标列表 = [各有效网格司机数下标] + [各有效网格订单数下标]，
# 供 stateProcessor 从展平状态中一次取出有效网格的司机/订单分布
target_id_states = env.target_grids + temp.tolist()

# （已注释）旧代码：直接展平状态并用 utility_conver_states 抽取有效网格分布，现改由 stateProcessor 完成
# curr_s = np.array(env.reset_clean()).flatten()  # [0] driver dist; [1] order dist
# curr_s = utility_conver_states(curr_s, target_id_states)
# 打印分隔提示：一天的订单已生成完毕
print("******************* Finish generating one day order **********************")





# 打印分隔提示：开始训练（文案写的是 Deep SARSA，本脚本实际训练 IDQN：
# Independent DQN，即每个网格一个独立的 DQN 智能体；DQN = Deep Q Network 深度 Q 网络，
# 用神经网络近似 Q 函数——Q(s,a) 表示状态 s 下采取动作 a 的长期累积回报期望）
print("******************* Starting training Deep SARSA **********************")
# 星号导入 algorithm.IDQN 的算法组件：Estimator（Q 网络封装）、stateProcessor（状态预处理器）、
# ReplayMemory（经验回放池）、ModelParametersCopier（网络参数复制器）
from algorithm.IDQN import *

# 最大迭代数（实际训练循环用下方字面量 25 轮，此值仅用于计算随机种子偏移；行尾英文为原注释）
MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
# 是否绘图的开关（本脚本未使用）
is_plot_figure = False
# 每个 episode（一轮从初始状态到结束的完整交互，这里一轮 = 仿真完整一天）的起始城市时刻
city_time_start = 0
# episode 长度：144 步 x 10 分钟 = 一整天
EP_LEN = 144
# 全局步数计数器（用于统计和 TensorBoard 日志）
global_step = 0
# 结束时刻 = 起始时刻 + 144（本脚本未使用）
city_time_end = city_time_start + EP_LEN
# 大写 ε 常量（本脚本未使用，实际探索率取自下方 epsilons 衰减表）
EPSILON = 0.9
# 折扣因子 γ（用于计算学习目标 r + γ·maxQ，含义同上面的 GAMMA）
gamma = 0.9
# 学习率（神经网络参数每次梯度更新的步长）
learning_rate = 1e-2

# 上一轮 episode 的奖励（占位变量，未使用；变量名拼写沿用原代码）
prev_epsiode_reward = 0
# 跨所有 episode 累计的每步奖励
all_rewards = []
# 每步订单接单率记录（注意：跨所有 episode 累积，每轮并不清空）
order_response_rates = []
# Q 值表总和记录（表格型 Q-learning 算法的遗留变量，未使用）
value_table_sum = []
# 每轮 episode 的总奖励（当天成交总额 GMV）
episode_rewards = []
# 每轮 episode 的对向调度冲突总数
episode_conflicts_drivers = []
# 每轮 episode 记录的平均订单接单率
order_response_rate_episode = []
# 每轮 episode 的调度司机总数
episode_dispatched_drivers = []
# 一天的时间步总数（每步 10 分钟）
T = 144
# 动作维度：6 个邻居方向 + 原地不动
action_dim = 7
# 每个网格样本的状态特征维度 = 司机分布 + 订单分布 + 时刻 one-hot + 网格 one-hot
# （one-hot：只有一位为 1、其余为 0 的编码向量）
state_dim = env.n_valid_grids * 3 + T

# （已注释）重置 TensorFlow 默认计算图（清空已有图再重建）
# tf.reset_default_graph()
# 创建 TF1 风格会话（session：TensorFlow 1.x 中执行计算图的运行环境，
# 所有张量运算都要在 session 里运行；本项目经 tf_compat 兼容层以 v1 模式跑在 TF2 下）
sess = tf.Session()
# 固定 TensorFlow 全局随机种子为 1（让网络初始化等随机行为可复现）
tf.set_random_seed(1)
# 创建在线 Q 网络q_estimator：3 层全连接网络，输入状态输出 7 个动作的 Q 值；
# summaries_dir 指定 TensorBoard（TensorFlow 自带的训练过程可视化工具）日志目录
q_estimator = Estimator(sess, action_dim,
                        state_dim,
                        env,
                        scope="q_estimator",
                        summaries_dir=log_dir)

# 创建目标网络 target_estimator：结构与在线网络相同（target network：参数定期从在线网络
# 复制过来的另一份 Q 网络，专门用来计算学习目标，避免"追着自己的尾巴跑"导致训练不稳定）
target_estimator = Estimator(sess, action_dim, state_dim, env, scope="target_q")
# 在会话中初始化所有全局变量（即两个网络的权重）
sess.run(tf.global_variables_initializer())
# 参数复制器：预先构建"把 q_estimator 的权重赋值给 target_estimator"的计算图操作
estimator_copy = ModelParametersCopier(q_estimator, target_estimator)
# 经验回放池（experience replay：把交互得到的转移样本 (s, a, r, s') 缓存起来，
# 训练时随机抽一小批，打破相邻样本间的相关性）；容量 100 万条，每批采样 3000 条
replay = ReplayMemory(memory_size=1e+6, batch_size=3000)
# 状态预处理器：负责全局状态与"每网格一条样本"之间的转换、奖励处理等
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)

# 模型保存器（用于把训练好的网络权重存成检查点文件）
saver = tf.train.Saver()
# 记录各轮 episode 使用的随机种子
save_random_seed = []
# 声明的总训练轮数（实际循环用的是下方字面量 25，此变量未使用）
N_ITER_RUNS = 25
# 随机种子偏移量（用于下方种子计算）
temp_value = 10
# 初始空闲司机数量的缩放比例（1 = 按原始数量，不缩放）
RATIO = 1
# 探索率 ε 的起始值（ε-greedy 策略：以概率 ε 随机选动作去探索、以 1-ε 选当前最优动作去利用）
EPSILON_start = 0.5
# 探索率 ε 的最终值
EPSILON_end = 0.1
# ε 衰减步数：用 15 轮从起始值线性降到最终值
epsilon_decay_steps=15
# 生成 ε 衰减表：15 个值从 0.5 线性递减到 0.1，训练前期多探索、后期多利用
epsilons = np.linspace(EPSILON_start, EPSILON_end, epsilon_decay_steps)
# 外层训练循环：共 25 轮 episode（episode 即一轮从初始状态到结束的完整交互，这里一轮 = 仿真完整一天）
for n_iter in np.arange(25):
    # 本轮随机种子 = 轮次 + MAX_ITER - temp_value（即 40~64），保证每轮结果可复现且互不相同
    RANDOM_SEED = n_iter + MAX_ITER - temp_value
    # 把该随机种子写入仿真环境（控制环境内部的随机数行为）
    env.reset_randomseed(RANDOM_SEED)
    # 记录本轮使用的随机种子
    save_random_seed.append(RANDOM_SEED)
    # 状态/动作/奖励批缓冲（本脚本逐条存入经验回放池，这三个列表占位未使用）
    batch_s, batch_a, batch_r = [], [], []
    # 本轮逐步奖励记录（GMV：Gross Merchandise Value，成交订单总金额）
    batch_reward_gmv = []
    # 本轮奖励累计变量（占位未使用，拼写沿用原代码）
    epsiode_reward = 0
    # 被调度司机数计数（占位未使用，实际用 curr_num_actions 统计）
    num_dispatched_drivers = 0

    # 重置环境（下方英文为原注释）
    # reset env
    # 订单生成模式标志：1 表示重置时重新抽样生成一天的订单
    is_regenerate_order = 1
    # 重置环境开始新的一天：清空全部司机与订单、按采样概率 1/28 从 4 周真实订单中抽出一天的订单、
    # 按 ratio=1 初始化空闲司机总数、从 city_time_start=0 时刻开始；
    # 返回初始状态 curr_state = [空闲司机分布矩阵, 订单分布矩阵]（均为 MxN）
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    # 预匹配：模拟每个网格用自身及邻居的空闲司机去抵消订单，
    # 返回 [剩余空闲司机, 剩余订单] 两个 MxN 矩阵
    info = env.step_pre_order_assigin(curr_state)
    # 取出各有效网格的剩余空闲司机数（作为每个网格待分配调度的司机数量）
    context = stateprocessor.compute_context(info)
    # 从展平的初始状态中抽取有效网格的 [司机分布, 订单分布]（长度 = 2 x 有效网格数）
    curr_s = stateprocessor.utility_conver_states(curr_state)
    # 归一化：司机部分除以其最大值、订单部分除以其最大值，数值压到 0~1
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    # 构造"每个网格一条样本"的初始状态矩阵 s_grid（行数 = 有效网格数，列数 = state_dim）：
    # [归一化司机分布, 归一化订单分布, 时刻 one-hot(144), 网格 one-hot]（行尾 t0、s0 为原代码标记：初始时刻、初始状态）
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    # 以下三个列表是本轮 episode 的统计容器（下方英文为原注释）
    # record rewards to update the value table
    episodes_immediate_rewards = []
    # 本轮逐步的对向调度冲突数
    num_conflicts_drivers = []
    # 本轮逐步的调度司机数
    curr_num_actions = []
    # 本轮探索率：前 15 轮按衰减表取值，之后固定为 0.1
    epsilon = epsilons[n_iter] if n_iter < 15 else EPSILON_end
    # 内层主循环：推进 EP_LEN+1 = 145 步（多出的 1 步用于把最后一步也配成完整转移样本）
    for ii in np.arange(EP_LEN + 1):

        # 输入当前状态，输出调度动作（下方英文为原注释）
        # INPUT: state,  OUTPUT: action
        # 用 Q 网络选动作：对每个网格，按 ε-greedy 概率把该网格剩余的空闲司机
        # 分配到 7 个动作（6 个邻居方向 + 原地不动）上；返回值依次为：
        # qvalues（各网格样本的 7 个 Q 值）、action_idx（动作目标节点 id）、
        # action_idx_valid（目标网格在 target_grids 中的下标）、action_neighbor_idx（动作方向编号）、
        # action_tuple（(起点网格, 终点网格, 司机数) 的调度列表）、action_starting_gridids（各起点网格的不同动作数）
        qvalues, action_idx, action_idx_valid, action_neighbor_idx, \
        action_tuple, action_starting_gridids = q_estimator.action(s_grid, context, epsilon)
        # a0

        # 推进一个时间步：执行调度动作 action_tuple（把空闲司机移往相邻网格）、匹配订单并计算
        # 奖励 r（= 匹配订单价格之和）、generate_order=2 使用预抽好的一天订单序列；
        # 返回下一状态、奖励和 info = [各网格奖励, 预匹配上下文]
        # ONE STEP: r0
        next_state, r, info = env.step(action_tuple, 2)

        # 各网格的即时奖励：从 info 中取各有效网格的订单收入，再除以当前各网格司机数（即人均奖励）
        # r0
        immediate_reward = stateprocessor.reward_wrapper(info, curr_s)

        # 动作 one-hot 矩阵（行数 = 本步动作数，列数 = 7）：每行标记该样本选了哪个动作
        # a0
        action_mat = stateprocessor.to_action_mat(action_neighbor_idx)

        # 训练用状态：把每个网格的状态行按其动作数复制扩展，得到与动作一一对应的样本（行数 = 动作数）
        # s0
        s_grid_train = stateprocessor.to_grid_state_for_training(s_grid, action_starting_gridids)

        # 每个训练样本的下一状态：[下一时刻归一化司机/订单分布, 下一时刻 one-hot, 动作终点网格 one-hot]
        # s1
        s_grid_next = stateprocessor.to_grid_next_states(s_grid_train, next_state, action_idx_valid, env.city_time)

        # 保存转移样本到经验回放池（下方英文为原注释）
        # Save transition to replay memory
        # 第 0 步还没有"上一步"的配对信息，无法组成完整转移样本 (s, a, r, s')，跳过存储
        if ii != 0:
            # 取上一步各动作终点网格的人均奖励，与本步动作样本对齐（下方英文为原注释）
            # r1, c0
            r_grid = stateprocessor.to_grid_rewards(action_idx_valid_prev, immediate_reward)
            # TD 目标（temporal difference target）= 即时奖励 + 折扣因子 gamma x 目标网络
            # 对下一状态最大 Q 值的估计，即 r + γ·max_a' Q_target(s', a')
            targets_batch = r_grid + gamma * target_estimator.predict(s_grid_next_prev)

            # 把上一步的 (状态, 动作, 学习目标, 下一状态) 作为一条转移样本存入经验回放池（下方英文为原注释）
            # s0, a0, r1
            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid_next_prev)

        # 暂存本步的训练状态，供下一轮循环配对成转移样本
        state_mat_prev = s_grid_train
        # 暂存本步的动作矩阵
        action_mat_prev = action_mat
        # 暂存本步的剩余司机上下文
        context_prev = context
        # 暂存本步的下一状态
        s_grid_next_prev = s_grid_next
        # 暂存本步的动作终点网格下标
        action_idx_valid_prev = action_idx_valid

        # 用 step 返回的 info[1]（预匹配上下文）更新各有效网格的剩余空闲司机数（下方英文为原注释）
        # c1
        context = stateprocessor.compute_context(info[1])
        # 从下一状态抽取有效网格的 [司机分布, 订单分布]（下方英文为原注释）
        # s1
        curr_s = stateprocessor.utility_conver_states(next_state)
        # 对下一状态做归一化
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        # 构造下一时刻的"每网格一条样本"状态矩阵（行尾 t0、s0 为原代码标记）
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

        # 从经验回放池抽样小批量更新 Q 网络（方法 1；下方英文为原注释）
        # Sample a minibatch from the replay memory and update q network training method1
        # 经验回放池非空才进行训练
        if replay.curr_lens != 0:
            # 每个时间步做 20 次小批量参数更新
            for _ in np.arange(20):
                # 从回放池随机采样一批（最多 3000 条）转移样本
                fetched_batch = replay.sample()
                # 解包：状态 / 动作 one-hot / 学习目标 / 下一状态
                mini_s, mini_a, mini_target, mini_next_s = fetched_batch
                # 对在线 Q 网络做一次梯度下降（最小化预测 Q(s,a) 与目标之间的平方误差）
                q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
                # 全局步数 +1
                global_step += 1

        # 记录统计数据（下方两行英文为原注释）
        # Perform gradient descent update
        # book keeping
        # 全局步数 +1
        global_step += 1
        # 把本步奖励计入全程累计奖励
        all_rewards.append(r)
        # 把本步奖励计入本轮 GMV
        batch_reward_gmv.append(r)
        # 记录本步订单接单率（该列表跨所有 episode 累积）
        order_response_rates.append(env.order_response_rate)
        # 统计本步的对向调度冲突数（同时出现 A→B 与 B→A 的调度记 1 次冲突）
        num_conflicts_drivers.append(collision_action(action_tuple))
        # 本步调度的司机总数（action_tuple 每项第 3 个元素是该动作移动的司机数；无动作则记 0）
        curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
        # 记录本步调度司机数
        curr_num_actions.append(curr_num_action)

    # 本轮总奖励 = 逐步奖励求和（去掉第 0 个记录值）
    episode_reward = np.sum(batch_reward_gmv[1:])
    # 存入每轮总奖励列表
    episode_rewards.append(episode_reward)
    # 平均接单率：对累计的所有逐步记录求均值（去掉第 0 个记录值；order_response_rates 跨轮累积）
    n_iter_order_response_rate = np.mean(order_response_rates[1:])
    # 存入每轮接单率列表
    order_response_rate_episode.append(n_iter_order_response_rate)
    # 本轮对向调度冲突总数（去掉最后一个记录值）
    episode_conflicts_drivers.append(np.sum(num_conflicts_drivers[:-1]))
    # 本轮调度司机总数（去掉最后一个记录值）
    episode_dispatched_drivers.append(np.sum(curr_num_actions[:-1]))

    # 打印本轮统计：轮次、总奖励、接单率、冲突数、调度司机数
    print("iteration {} ********* reward {} order{} conflicts {} drivers {}".format(n_iter, episode_reward,
                                                                                             order_response_rate_episode[-1],
                                                                                             episode_conflicts_drivers[-1],
                                                                                             episode_dispatched_drivers[-1]))

    # 把各项统计结果用 pickle 序列化保存到日志目录的 results.pkl（"wb" 表示二进制写模式），
    # 列表元素依次为：每轮总奖励 / 每轮接单率 / 各轮随机种子 / 每轮冲突总数 / 每轮调度司机总数
    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers,
                 episode_dispatched_drivers], open(log_dir + "results.pkl", "wb"))

    # 若是第 24 轮（0 起数的最后一轮），保存结果后直接退出循环，不再做目标网络更新和存模型
    if n_iter == 24:
        break  # 跳出外层训练循环（最后一轮不再更新目标网络、不再保存模型）

    # （已注释）另一种训练方式：每轮结束后集中抽样 4000 次更新 Q 网络
    # # training method 2.
    # for _ in np.arange(4000):
    #     fetched_batch = replay.sample()
    #     mini_s, mini_a, mini_target, mini_next_s = fetched_batch
    #     q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
    #     global_step += 1

    # 更新目标 Q 网络（下方英文为原注释）
    # update target Q network
    # 执行参数复制：把在线网络 q_estimator 的最新权重复制到目标网络 target_estimator
    estimator_copy.make(sess)


    # 把当前网络权重保存为检查点文件 model.ckpt（checkpoint：可从中恢复继续训练或做评估）
    saver.save(sess, log_dir+"model.ckpt")
