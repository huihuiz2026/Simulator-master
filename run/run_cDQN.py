
# 导入 pickle（Python 对象序列化模块，用于读取/保存 .pkl 数据文件）和
# sys（用于把项目根目录加入模块搜索路径）
import pickle, sys
# 导入 pathlib.Path 类：以面向对象方式处理文件路径，跨平台兼容
from pathlib import Path
# 把本文件所在目录的上一级（项目根目录 Simulator-master）插入 sys.path 最前面，
# 保证无论从哪个目录运行脚本，都能找到根目录下的 algorithm 和 simulator 两个包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# from simulator.utilities import *
# （上一行为被注释掉的旧导入；所需工具实际经由下面两个 import * 引入）
# 导入算法公共工具模块：提供 TensorFlow 兼容层 tf（让 TF1 代码跑在 TF2 下）、
# 对开冲突统计 collision_action、状态抽取 utility_conver_states 等
from algorithm.alg_utility import *
# 导入城市仿真环境模块：提供 CityReal 类；模块内 import 的 time、np、mkdir_p
# 等名字也随 * 一并进入当前命名空间
from simulator.envs import *

# ===================== 第一部分：加载外部数据 =====================
################## Load data ###################################
# 外部数据根目录（作者服务器上的绝对路径；仓库中不含这些数据文件，需自行准备）
dir_prefix = "/mnt/research/linkaixi/AllData/dispatch/"
# 当前时间字符串（格式：年月日_时-分），用作本次实验的目录名，区分不同的运行
current_time = time.strftime("%Y%m%d_%H-%M")
# 拼接本次实验的日志目录：模型与结果文件都会保存在这里
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
# 递归创建日志目录（已存在时不报错）
mkdir_p(log_dir)
# 打印日志目录位置，便于运行后查找输出文件
print("log dir is {}".format(log_dir))


# 仿真器输入数据所在的子目录（外部数据）
data_dir = dir_prefix + "dispatch_realdata/data_for_simulator/"
# 订单时长分布参数：置空列表占位——本脚本使用预生成的真实订单，
# 不需要"随机生成订单"所用的分布参数
order_time_dist = []
# 订单价格分布参数：同上，仅占位
order_price_dist = []
# 加载城市地图矩阵（-100 表示无效网格，正整数是有效网格编号）；
# 这些 pkl 文件是 Python 2 时代的旧格式，需指定 encoding='latin1' 才能在 Python 3 读取
mapped_matrix_int = pickle.load(open(data_dir+"mapped_matrix_int.pkl", 'rb'), encoding='latin1')
# 加载各时刻各网格的订单数量分布参数（每个时刻一个 {网格编号: [均值]} 字典）
order_num_dist = pickle.load(open(data_dir+"order_num_dist", 'rb'), encoding='latin1')
# 加载各时刻全城空闲司机总数的分布 [均值, 标准差]（共 144 个时刻）
idle_driver_dist_time = pickle.load(open(data_dir+"idle_driver_dist_time", 'rb'), encoding='latin1')
# 加载各时刻各有效网格空闲司机数量矩阵（144 x 有效网格数）
idle_driver_location_mat = pickle.load(open(data_dir+"idle_driver_location_mat", 'rb'), encoding='latin1')
# 加载目标区域（调度范围）的网格编号列表
target_ids = pickle.load(open(data_dir+"target_grid_id.pkl", 'rb'), encoding='latin1')
# 加载司机上下线控制矩阵：每时刻每网格 [上线数-下线数的均值, 标准差]
onoff_driver_location_mat = pickle.load(open(data_dir + "onoff_driver_location_mat", 'rb'), encoding='latin1')
# 预生成的真实订单文件路径（一天的订单序列）
order_filename = dir_prefix + "dispatch_realdata/orders/all_orders_target"
# 加载真实订单数组：每条为 [起点网格, 终点网格, 起始时刻, 时长(步), 价格]
order_real = pickle.load(open(order_filename, 'rb'), encoding='latin1')
# 从地图矩阵取出行数 M 和列数 N
M, N = mapped_matrix_int.shape
# 打印提示：外部数据全部加载完成
print("finish load data")


# ===================== 第二部分：初始化仿真环境 =====================
################## Initialize env ###################################
# 邻域边数：6 表示六边形邻域（每个网格最多有 6 个邻居方向）
n_side = 6
# 折扣因子常量（未来奖励折算到当前时的折扣比例；本脚本后续实际使用的是下面的小写 gamma）
GAMMA = 0.9
# 最大调度/订单跨越的网格层数
l_max = 9

# 用加载的数据构造 CityReal 城市仿真环境，参数依次为：地图矩阵、订单数量分布、
# 空闲司机总数分布、空闲司机位置矩阵、订单时长分布（空占位）、订单价格分布（空占位）、
# 最大距离 l_max、行数 M、列数 N、邻域边数、订单采样概率、预生成订单数组、司机上下线矩阵；
# 其中采样概率 1/28 表示 reset 时每条真实订单以 1/28 的概率被抽入当天的订单序列
env = CityReal(mapped_matrix_int, order_num_dist,               # 地图矩阵、订单数量分布
               idle_driver_dist_time, idle_driver_location_mat,  # 司机总数分布、司机位置矩阵
               order_time_dist, order_price_dist,                # 订单时长、价格分布（空占位）
               l_max, M, N, n_side, 1/28.0, order_real, onoff_driver_location_mat)  # 其余参数


# 重新拼接一遍日志目录（与开头相同，冗余赋值，不改变结果）
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)


# 目标网格编号加 M*N 偏移：reset 返回的状态是 [空闲司机分布, 订单分布] 两个 MxN
# 矩阵展平后前后拼接的向量——前 M*N 个元素是司机数、后 M*N 个是订单数，
# 因此目标网格在"订单段"中的下标 = 网格编号 + M*N
temp = np.array(env.target_grids) + env.M * env.N
# 拼接"司机段下标 + 订单段下标"，得到目标网格在展开状态向量中的成组下标
target_id_states = env.target_grids + temp.tolist()


# 重置环境：清空司机与订单、按采样概率抽出一天的订单、初始化司机分布，
# 返回初始状态 [空闲司机分布矩阵, 订单分布矩阵]；再展平成一维向量
# （行尾英文注释：[0] 是司机分布、[1] 是订单分布）
curr_s = np.array(env.reset_clean()).flatten()  # [0] driver dist; [1] order dist
# 按下标列表 target_id_states 抽取目标网格的 [空闲司机数, 订单数]
# （全局状态 -> 目标区域状态）
curr_s = utility_conver_states(curr_s, target_id_states)
# 打印提示：一天的订单已生成完毕
print("******************* Finish generating one day order **********************")



# 打印提示：开始训练（cDQN 本质上是 deep expected SARSA——"期望 SARSA"的
# 深度网络版本：训练目标按各动作概率加权求期望，见 cDQN.py 顶部英文说明）
print("******************* Starting training Deep SARSA **********************")
# 导入 contextual DQN 算法组件：Estimator（Q 值网络）、stateProcessor（状态处理器）、
# ReplayMemory（经验回放池）、ModelParametersCopier（网络参数复制器）
from algorithm.cDQN import *

# 常量 50：并非训练轮数，而是随机种子的偏移基数（实际训练轮数由下面 N_ITER_RUNS 决定）
MAX_ITER = 50
# 是否绘图的开关（本脚本未使用，占位）
is_plot_figure = False
# 每轮（episode，即"一轮完整交互"：从初始状态跑到一天结束的整个过程）的起始时刻
city_time_start = 0
# 每轮的仿真步数：144 步 x 10 分钟 = 24 小时（一整天）
EP_LEN = 144
# 全局训练步数计数器（用作 TensorBoard 日志的横轴；TensorBoard 是 TF 自带的
# 训练过程可视化工具）
global_step = 0
# 每轮的结束时刻 = 起始 + 144（本脚本后续未使用，占位）
city_time_end = city_time_start + EP_LEN
# 旧版探索系数（本脚本实际使用下面的 epsilons 衰减序列，此常量未再使用）
EPSILON = 0.8
# 折扣因子 gamma=0.9：未来奖励折算到当前时的折扣比例（越远的收益越打折扣）
gamma = 0.9
# 学习率：网络参数每次更新时的步长
learning_rate = 1e-3

# 上一轮的累计奖励（占位变量，后续未使用；变量名中的拼写 epsiode 为原文如此）
prev_epsiode_reward = 0
# 每步调度车辆数的记录列表（占位，每轮开头会被重建）
curr_num_actions = []
# 记录所有步的环境总奖励（GMV：成交订单价格之和）
all_rewards = []
# 记录每一轮的平均订单接单率（成功匹配订单数 / 总订单数）
order_response_rate_episode = []
# 占位列表（本脚本未使用）
value_table_sum = []
# 记录每一轮的累计奖励
episode_rewards = []
# 记录每一轮的"对开冲突"调度总数（同时出现 a->b 与 b->a 的动作对数）
episode_conflicts_drivers = []
# 记录每一轮逐步的接单率明细
record_all_order_response_rate = []

# 一天的总步数（同时决定状态中时间 one-hot 编码的维度；
# one-hot：只有一位为 1 其余为 0 的向量编码）
T = 144
# 动作维度 = 7：6 个邻居方向 + 1 个原地不动
action_dim = 7
# 状态维度 = 3 x 有效网格数 + 144：全局空闲司机分布 + 全局订单分布（2 份）
# + 每个网格自己的 one-hot 编号（1 份）+ 144 维时间 one-hot
state_dim = env.n_valid_grids * 3 + T


# tf.reset_default_graph()
# 创建 TensorFlow 会话（TF1 风格：计算图先建好、由会话负责执行；
# 此处经 tf_compat 兼容层在 TF2 下运行）
sess = tf.Session()
# 固定 TensorFlow 的随机种子，使网络参数初始化等随机操作可复现
tf.set_random_seed(1)

# 创建主 Q 网络（Estimator）：输入各网格的状态、输出每个网格的价值估计
# （Q 值：从该状态出发预期能累积多少奖励），scope 指定变量命名空间，
# summaries_dir 指定 TensorBoard 摘要日志目录
q_estimator = Estimator(sess, action_dim,        # 会话对象、动作维度
                        state_dim,               # 状态维度
                        env,                     # 环境对象（读取网格与邻居结构）
                        scope="q_estimator",     # 变量命名空间：主网络
                        summaries_dir=log_dir)   # TensorBoard 日志目录

# 创建 target 目标网络（target network：结构与主网络相同、参数定期从主网络
# 复制过来的"慢速副本"，用它计算训练目标可避免"追逐移动目标"造成的震荡）
target_estimator = Estimator(sess, action_dim, state_dim, env, scope="target_q")
# 初始化计算图中所有 TensorFlow 变量（网络权重等）
sess.run(tf.global_variables_initializer())
# 构造参数复制器：预先建好"把 q_estimator 的参数赋值给 target_estimator"的操作
estimator_copy = ModelParametersCopier(q_estimator, target_estimator)
# 创建经验回放池（replay buffer：存放交互得到的 [状态, 动作, 目标值, 下一状态]
# 样本，训练时随机抽取一批，以打破样本间的时间相关性）；容量 1e6，批大小 3000
replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
# 创建状态处理器：负责全局状态的抽取/归一化/扩展成逐网格状态、奖励格式转换、
# 上下文（每格需调度的司机数）计算等前后处理
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)

# 司机规模比例：1 表示 reset 时按全量初始化空闲司机数
RATIO = 1
# 创建 TensorFlow 模型保存器，用于把训练好的网络参数存盘
saver = tf.train.Saver()


# 打印提示：开始 contextual deep Q learning 训练
print("Start training contextual deep Q learning. ")
# 记录每一轮使用的随机种子，便于复现
save_random_seed = []
# 总训练轮数：25 轮（每轮仿真一整天）
N_ITER_RUNS = 25
# 随机种子偏移量（与 MAX_ITER 一起决定每轮的种子值）
temp_value = 10
# epsilon-greedy 探索率的起始值（epsilon-greedy：以概率 epsilon 随机选动作"探索"，
# 否则选当前最优动作"利用"）
EPSILON_start = 0.5
# 探索率的最终值
EPSILON_end = 0.1
# 探索率从起始值衰减到最终值所用的轮数
epsilon_decay_steps=15
# 生成线性衰减的探索率序列：前 15 轮从 0.5 均匀降到 0.1
epsilons = np.linspace(EPSILON_start, EPSILON_end, epsilon_decay_steps)
# 记录每一轮的调度司机总数
episode_dispatched_drivers = []
# 训练主循环：共 25 轮，每轮 = 一个 episode（完整仿真一天 144 步）
for n_iter in np.arange(N_ITER_RUNS):
    # 本轮随机种子 = 轮序号 + 50 - 10，控制本轮订单抽样与司机生成的随机性
    RANDOM_SEED = n_iter + MAX_ITER - temp_value
    # 把环境的随机数发生器重置为本轮种子，保证实验可复现
    env.reset_randomseed(RANDOM_SEED)
    # 记录本轮种子
    save_random_seed.append(RANDOM_SEED)
    # 批量数据占位（每轮开头清空；batch_s/batch_r 后由 replay.sample 的返回值
    # 覆盖，batch_a 未使用）
    batch_s, batch_a, batch_r = [], [], []
    # 本轮逐步的 GMV 奖励记录（用于最后累加出本轮总奖励）
    batch_reward_gmv = []
    # 本轮奖励累计（占位，后面直接用 np.sum 重新计算）
    epsiode_reward = 0
    # 调度司机计数（占位，后续未使用）
    num_dispatched_drivers = 0

    # reset env
    # （原注释：重置环境）
    # 以下为已注释的旧逻辑：按轮数决定是否重新生成订单（现已固定为总是重新生成）
    # if n_iter % 1 == 0:
    #     is_regenerate_order = 1
    # else:
    #     is_regenerate_order = 0
    # 每轮都重新抽取一天的订单（=1 表示 reset 时按 1/28 概率重新抽样真实订单）
    is_regenerate_order = 1
    # 重置环境：清空司机和订单、重新生成一天订单、按比例 ratio 初始化空闲司机分布，
    # 城市时间回到 city_time_start（0）；返回初始状态 [司机分布, 订单分布]
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    # 订单"预分配"（函数名 assigin 为 assign 的原文拼写）：模拟每个网格的订单先被
    # 本格及邻居的空闲司机抢单，返回抢单后剩余的 [空闲司机分布, 订单分布]
    info = env.step_pre_order_assigin(curr_state)
    # 从预分配结果提取上下文 context：每个目标网格上"本格订单消化后仍空闲、
    # 可被调度去别处"的司机数（即该网格要派出的司机数量）
    context = stateprocessor.compute_context(info)
    # 抽取目标网格的 [空闲司机数, 订单数]
    curr_s = stateprocessor.utility_conver_states(curr_state)
    # 归一化：司机段除以最大司机数、订单段除以最大订单数，避免数值间差异过大
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    # 把全局归一化状态扩展成"每个网格一份样本"的状态矩阵（有效网格数 x state_dim），
    # 每行 = [全局司机分布, 全局订单分布, 时刻 one-hot, 本网格编号 one-hot]
    # （行尾英文注释：t0 时刻、s0 状态）
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    # record rewards to update the value table
    # （原注释：记录奖励用于更新价值表）
    # 占位列表（本脚本未使用）
    episodes_immediate_rewards = []
    # 本轮逐步接单率记录
    order_response_rates = []
    # 已注释的旧探索率公式（行尾中文为原文：测试阶段不改变 epsilon）
    # epsilon = EPSILON * (1 - np.max([n_iter, temp_value+5]) / N_ITER_RUNS)  #testing staget 不改变epsilon
    # 本轮探索率：前 15 轮取衰减序列中的对应值，之后固定为 0.1
    epsilon = epsilons[n_iter] if n_iter < 15 else EPSILON_end
    # 本轮逐步的对开冲突数记录
    num_conflicts_drivers = []
    # 本轮逐步的调度司机数记录
    curr_num_actions = []
    # 内层循环：仿真一整天的 144 步（+1 使循环多跑一步——第 0 步只保存状态与动作、
    # 不存样本，这样一天的 144 个动作都能配上"后续奖励与状态"凑成完整转移样本）
    for ii in np.arange(EP_LEN + 1):
        # 选动作：用主 Q 网络根据状态 s_grid 与上下文 context 生成调度动作——
        # 对每个网格：取其邻居及自身的价值估计构造 7 维动作价值，价值低于"原地不动"
        # 的方向被置零；按 epsilon-greedy 生成动作概率（保留方向平分 epsilon 概率，
        # 最优方向额外加上 1-epsilon），再用多项式抽样把该格"待调度司机数"分配到
        # 各方向；返回动作列表 [(起点网格, 终点网格, 司机数)] 与各网格动作概率矩阵
        # INPUT: state,  OUTPUT: action
        # （原注释：输入状态、输出动作）
        action_tuple, valid_action_prob_mat = q_estimator.action(s_grid, context, epsilon)

        # a0
        # （原注释标记：本步选出的动作记作 a0）

        # ONE STEP: r0
        # （原注释：推进一步、得到奖励 r0）
        # 环境执行一步：先执行调度动作，再匹配订单计算奖励（成交订单价格之和），
        # 然后时间 +1、调度司机到达目的地、生成新订单、司机上下线；
        # generate_order=2 表示使用预生成的固定订单序列（而非随机生成）；
        # 返回下一状态、本步总奖励 r、逐格奖励信息 info
        next_state, r, info = env.step(action_tuple, 2)

        # r0
        # （原注释标记：本步奖励记作 r0）
        # 把 info 中的逐格奖励转成"平均每司机奖励"：每个网格的奖励除以该格
        # 调度前的空闲司机数（为 0 时按 1 处理，避免除零）
        immediate_reward = stateprocessor.reward_wrapper(info, curr_s)

        # Save transition to replay memory
        # （原注释：把转移样本存入经验回放池）
        # 跳过第 0 步：此时还没有"上一步"保存的状态/动作，无法组成完整转移
        if ii != 0:
            # r1, c0
            # （原注释：奖励 r1、上下文 c0）
            # 把逐格奖励整理成列向量（有效网格数 x 1）
            r_grid = stateprocessor.to_grid_rewards(immediate_reward)
            # s0, a0, r1
            # （原注释：由 s0、a0、r1 计算目标值）
            # 用 target 目标网络计算每个网格的 TD 目标（时序差分目标 =
            # 即时奖励 + 折扣后的未来价值）：取上一步动作概率最高的邻居方向，
            # 目标 = 该方向网格的本步奖励 + gamma * s1 中该网格的价值估计
            targets_batch = target_estimator.compute_targets(action_mat_prev, s_grid, r_grid, gamma)
            # 把转移样本 (s0, a0 的概率, TD 目标, s1) 存入经验回放池
            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)

        # 保存当前状态与动作概率，供下一轮迭代组成转移样本
        state_mat_prev = s_grid
        action_mat_prev = valid_action_prob_mat

        # s1
        # （原注释标记：下一状态记作 s1）
        # 处理下一状态：抽取目标网格数值 -> 归一化 -> 扩展成逐网格状态矩阵
        curr_s = stateprocessor.utility_conver_states(next_state)
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        # （行尾英文注释：t0 时刻、s0 状态——指当前时间步的状态）
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

        # c1
        # （原注释标记：下一时刻的上下文记作 c1）
        # 用 info 的第二部分（step 内部算好的"抢单后剩余司机/订单"矩阵）计算
        # 下一时刻各网格需调度的司机数
        context = stateprocessor.compute_context(info[1])

        # training method 2
        # （原注释：训练方式 2；以下整块已被注释停用——
        # 原意是每步从回放池抽小批次、边交互边更新网络 20 次）
        # Sample a minibatch from the replay memory and update q network
        # if replay.curr_lens != 0:
        #     for _ in np.arange(20):
        #         batch_s, _, batch_r, _ = replay.sample()
        #         iloss = q_estimator.update(batch_s, batch_r, 1e-3, global_step)
        #         global_step += 1
        # print "******** city time {} *********".format(env.city_time)

        # Perform gradient descent update
        # （原注释：执行梯度下降更新；真正的更新在本轮结束后的批量训练中
        # 进行，此处只做计数）
        # book keeping
        # （原注释：记账——以下为本步统计记录）
        # 全局步数 +1
        global_step += 1
        # 记录本步的环境总奖励
        all_rewards.append(r)
        # 记录本步 GMV 奖励（供本轮汇总）
        batch_reward_gmv.append(r)
        # 记录本步订单接单率
        order_response_rates.append(env.order_response_rate)
        # 统计本步实际调度的司机总数：对每个动作元组取第 3 个元素（司机数）求和；
        # 无动作时记 0
        curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
        # 记录本步调度司机数
        curr_num_actions.append(curr_num_action)
        # 统计本步"对开冲突"数（同时出现 a->b 与 b->a 的调度对数）
        num_conflicts_drivers.append(collision_action(action_tuple))

    # training method 1
    # （原注释：训练方式 1——一轮仿真结束后集中批量训练）
    # 批量训练循环：用本轮回放池中收集的样本共训练 4000 个批次
    for _ in np.arange(4000):
        # 从经验回放池随机采样一个批次（3000 条转移样本）
        batch_s, _, batch_r, _ = replay.sample()
        # 用该批次更新主 Q 网络：最小化网络输出与 TD 目标之间的均方误差，
        # 学习率 1e-3；返回本批的损失值
        iloss = q_estimator.update(batch_s, batch_r, 1e-3, global_step)
        # 全局步数 +1
        global_step += 1

        # update target Q network
        # （原注释：更新 target Q 网络）
        # 已注释的"每 70 步定期更新一次 target 网络"的旧写法：
    #     if (global_step + 1) % 70 == 0:
    # 批量训练结束后，把主 Q 网络的全部参数一次性复制到 target 目标网络
    # （注意：此行在 for 循环之外，每轮只复制一次）
    estimator_copy.make(sess)

    # 本轮总奖励 = 逐步 GMV 奖励之和（去掉第 0 步的记录）
    episode_reward = np.sum(batch_reward_gmv[1:])
    # 记入每轮总奖励列表
    episode_rewards.append(episode_reward)
    # 本轮平均接单率（去掉第 0 步的记录）
    n_iter_order_response_rate = np.mean(order_response_rates[1:])
    # 记入每轮平均接单率列表
    order_response_rate_episode.append(n_iter_order_response_rate)
    # 保存本轮逐步接单率明细
    record_all_order_response_rate.append(order_response_rates)
    # 本轮对开冲突总数（去掉最后一个记录）
    episode_conflicts_drivers.append(np.sum(num_conflicts_drivers[:-1]))
    # 本轮调度司机总数（去掉最后一个记录）
    episode_dispatched_drivers.append(np.sum(curr_num_actions[:-1]))


    # 打印本轮统计：轮次、总奖励、平均接单率、调度司机数、冲突数、探索率
    print("******** iteration {} ********* reward {}, order_response_rate {} number drivers {}, conflicts {}, epsilon {}".format(n_iter, episode_reward,  # 轮次、总奖励
                                                                                                        n_iter_order_response_rate,  # 本轮平均接单率
                                                                                                         episode_dispatched_drivers[-1],  # 本轮调度司机总数
                                                                                                        episode_conflicts_drivers[-1],  # 本轮对开冲突总数
                                                                                                             epsilon))  # 本轮探索率


    # 把训练统计结果（各轮总奖励/平均接单率/随机种子/冲突数/调度司机数）
    # 序列化保存到 results.pkl 文件
    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers, episode_dispatched_drivers], open(log_dir + "results.pkl", "wb"))

    # 保存当前模型参数到 model.ckpt（TensorFlow 检查点文件，每轮覆盖保存一次）
    saver.save(sess, log_dir+"model.ckpt")
