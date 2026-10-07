# -*- coding: utf-8 -*-
# （上一行声明文件编码为 UTF-8，是 Python 2 时代的写法，Python 3 下可省略）
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
# 导入文件复制函数 copyfile（本脚本中未实际使用，占位导入）
from shutil import copyfile

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


# 打印提示：开始训练深度 actor-critic（演员-评论家算法：actor 演员=策略网络，
# 负责输出动作的概率分布；critic 评论家=价值网络，负责评估状态价值并指导 actor 改进）
print("******************* Starting training Deep actor critic **********************")
# 导入 contextual A2C 算法组件：Estimator（内含 actor/critic 双网络）、stateProcessor
# （状态处理器）、ReplayMemory 与 policyReplayMemory（两个经验回放池）
from algorithm.cA2C import *



# 常量 50：并非训练轮数，而是随机种子的偏移基数（实际轮数由下方 for 循环决定）
MAX_ITER = 50
# 是否绘图的开关（本脚本未使用，占位）
is_plot_figure = False
# 每轮（episode，即"一轮完整交互"：从初始状态跑到一天结束的整个过程）的起始时刻
city_time_start = 0
# 每轮的仿真步数：144 步 x 10 分钟 = 24 小时（一整天）
EP_LEN = 144

# 每轮的结束时刻 = 起始 + 144（本脚本后续未使用，占位）
city_time_end = city_time_start + EP_LEN
# 探索率参数：会传入 action，但 cA2C 的 action 内部并未使用它
# （动作改由 actor 网络输出的概率分布直接采样得到）
epsilon = 0.5
# 折扣因子 gamma=0.9：未来奖励折算到当前时的折扣比例（越远的收益越打折扣）
gamma = 0.9
# 学习率：网络参数每次更新时的步长
learning_rate = 1e-3

# 上一轮的累计奖励（占位变量，后续未使用；变量名中的拼写 epsiode 为原文如此）
prev_epsiode_reward = 0

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
# 创建算法主体 Estimator（contextual A2C）：内部同时构建 value 网络
# （critic：输入状态、输出各网格的价值估计）与 policy 网络（actor：输入状态、
# 输出各调度方向的动作概率分布），两网络参数在所有网格智能体间共享——
# 这是论文中效果最好的方法
q_estimator = Estimator(sess, action_dim,        # 会话对象、动作维度
                        state_dim,               # 状态维度
                        env,                     # 环境对象（读取网格与邻居结构）
                        scope="q_estimator",     # 变量命名空间
                        summaries_dir=log_dir)   # TensorBoard 日志目录


# 初始化计算图中所有 TensorFlow 变量（网络权重等）
sess.run(tf.global_variables_initializer())

# 创建值网络（critic）的经验回放池（replay buffer：存放交互得到的
# [状态, 动作, 目标值, 下一状态] 样本，训练时随机抽取一批以打破时间相关性）；
# 容量 1e6，批大小 3000
replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
# 创建策略网络（actor）的经验回放池：存放 [状态, 所选动作, 优势, 邻居掩码]；
# 容量 1e6，批大小 3000
policy_replay = policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
# 创建状态处理器：负责全局状态的抽取/归一化/扩展成逐网格状态、奖励格式转换、
# 上下文（每格需调度的司机数）计算等前后处理
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)


# "是否恢复已有模型"的占位标志（本脚本并未实现恢复逻辑，未使用）
restore = True
# 创建 TensorFlow 模型保存器，用于把训练好的网络参数存盘
saver = tf.train.Saver()


# 以下两行为已注释的调试记录开关（原意是记录每步的状态与动作，现已停用）：
# record_curr_state = []
# record_actions = []
# 记录每一轮使用的随机种子，便于复现
save_random_seed = []
# 记录每一轮的调度司机总数
episode_dispatched_drivers = []
# 值网络（critic）的训练步数计数器（用于 TensorBoard 日志的横轴）
global_step1 = 0
# 策略网络（actor）的训练步数计数器
global_step2 = 0
# 司机规模比例：1 表示 reset 时按全量初始化空闲司机数
RATIO = 1
# 训练主循环：共 25 轮（np.arange(25)），每轮 = 一个 episode（完整仿真一天 144 步）
for n_iter in np.arange(25):
    # 本轮随机种子 = 轮序号 + 50 - 10，控制本轮订单抽样与司机生成的随机性
    RANDOM_SEED = n_iter + MAX_ITER - 10
    # 把环境的随机数发生器重置为本轮种子，保证实验可复现
    env.reset_randomseed(RANDOM_SEED)
    # 记录本轮种子
    save_random_seed.append(RANDOM_SEED)
    # 批量数据占位（每轮开头清空；后续由 replay.sample 的返回值覆盖）
    batch_s, batch_a, batch_r = [], [], []
    # 本轮逐步的 GMV 奖励记录（用于最后累加出本轮总奖励）
    batch_reward_gmv = []
    # 本轮奖励累计（占位，后面直接用 np.sum 重新计算）
    epsiode_reward = 0
    # 调度司机计数（占位，后续未使用）
    num_dispatched_drivers = 0

    # reset env
    # （原注释：重置环境）
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
    # 本轮逐步的对开冲突数记录
    num_conflicts_drivers = []
    # 本轮逐步的调度司机数记录
    curr_num_actions = []
    # 本轮逐步接单率记录
    order_response_rates = []
    # 内层循环：仿真一整天的 144 步（+1 使循环多跑一步——第 0 步只保存状态与动作、
    # 不存样本，这样一天的 144 个动作都能配上"后续奖励与状态"凑成完整转移样本）
    for ii in np.arange(EP_LEN + 1):
        # 已注释的调试代码：记录每步的原始状态
        # record_curr_state.append(curr_state)
        # 选动作：先用 critic（价值网络）评估各网格及其邻居的价值，把价值低于
        # "原地不动"的方向从邻居掩码中排除；再由 actor（策略网络）对剩余方向输出
        # softmax 概率分布，按该分布为每个网格的"待调度司机数"（context）逐个抽取
        # 目的地；共返回 7 个结果：
        #   action_tuple          调度动作列表 [(起点网格, 终点网格, 司机数)]
        #   valid_action_prob_mat 各网格的动作概率矩阵（供计算值网络目标）
        #   policy_state          每个已执行动作对应的网格状态（actor 训练输入）
        #   action_choosen_mat    每个已执行动作方向的 one-hot 编码
        #   curr_state_value      每个动作起始网格的价值估计（计算优势用）
        #   curr_neighbor_mask    每个动作对应的邻居掩码（屏蔽无效/低价值方向）
        #   next_state_ids        每个动作的目的地网格编号
        # INPUT: state,  OUTPUT: action
        # （原注释：输入状态、输出动作）
        action_tuple, valid_action_prob_mat, policy_state, action_choosen_mat, \
        curr_state_value, curr_neighbor_mask, next_state_ids = q_estimator.action(s_grid, context, epsilon)
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
            # s0, a0, r1  for value newtwork
            # （原注释：供值网络使用；newtwork 为原文拼写）
            # 用 critic 计算每个网格的 TD 目标（时序差分目标 = 即时奖励 + 折扣后的
            # 未来价值）：把各邻居方向的（本步奖励 + gamma*下一状态价值）按上一步
            # 动作概率加权求和，得到期望目标值（期望 SARSA 风格）
            targets_batch = q_estimator.compute_targets(action_mat_prev, s_grid, r_grid, gamma)

            # advantage for policy network.
            # （原注释：供策略网络使用的优势）
            # 计算优势（advantage：动作的实际回报与状态平均价值之差，衡量该动作比
            # "平均水平"好多少）：每个已执行动作的优势 = 目的地网格的本步奖励
            # + gamma * 目的地网格在下一状态的价值 - 起始网格的当前价值估计
            advantage = q_estimator.compute_advantage(curr_state_value_prev, next_state_ids_prev,  # 上一步各动作的起始格价值、目的地编号
                                                      s_grid, r_grid, gamma)  # 下一状态、本步逐格奖励、折扣因子

            # 把值网络样本 (s0, a0 概率, TD 目标, s1) 存入 critic 的经验回放池
            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)
            # 把策略样本 (状态, 所选动作 one-hot, 优势, 邻居掩码) 存入 actor 的经验回放池
            policy_replay.add(policy_state_prev, action_choosen_mat_prev, advantage, curr_neighbor_mask_prev)

        # for updating value network
        # （原注释：以下保存供值网络更新用的上一时刻状态与动作概率）
        state_mat_prev = s_grid
        action_mat_prev = valid_action_prob_mat

        # for updating policy net
        # （原注释：以下保存供策略网络更新用的上一时刻动作记录）
        action_choosen_mat_prev = action_choosen_mat
        curr_neighbor_mask_prev = curr_neighbor_mask
        policy_state_prev = policy_state
        # for computing advantage
        # （原注释：以下保存供计算优势用的上一时刻价值与目的地编号）
        curr_state_value_prev = curr_state_value
        next_state_ids_prev = next_state_ids

        # s1
        # （原注释标记：下一状态记作 s1）
        # 更新原始状态引用，并处理下一状态：抽取目标网格数值 -> 归一化 ->
        # 扩展成逐网格状态矩阵
        curr_state = next_state
        curr_s = stateprocessor.utility_conver_states(next_state)
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        # （行尾英文注释：t0 时刻、s0 状态——指当前时间步的状态）
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

        # c1
        # （原注释标记：下一时刻的上下文记作 c1）
        # 用 info 的第二部分（step 内部算好的"抢单后剩余司机/订单"矩阵）计算
        # 下一时刻各网格需调度的司机数
        context = stateprocessor.compute_context(info[1])

        # training method 1.
        # （原注释：训练方式 1；以下整块已被注释停用——原意是每步从策略回放池
        # 抽小批次、边交互边更新策略网络 30 次）
        # #    # Sample a minibatch from the replay memory and update q network
        # if replay.curr_lens != 0:
        #     # update policy network
        #     for _ in np.arange(30):
        #         batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
        #         q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,
        #                                   global_step2)
        #         global_step2 += 1

        # Perform gradient descent update
        # （原注释：执行梯度下降更新；真正的更新在本轮结束后的批量训练中进行，
        # 此处只做计数）
        # book keeping
        # （原注释：记账——以下为本步统计记录）
        # 值网络训练步数 +1
        global_step1 += 1
        # 策略网络训练步数 +1
        global_step2 += 1
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

    # 打印本轮统计：轮次、总奖励、平均接单率、调度司机数、冲突数
    print("******** iteration {} ********* reward {}, order_response_rate {} number drivers {}, conflicts {}".format(n_iter, episode_reward,  # 轮次、总奖励
                                                                                                       n_iter_order_response_rate,  # 本轮平均接单率
                                                                                                     episode_dispatched_drivers[-1],  # 本轮调度司机总数
                                                                                                    episode_conflicts_drivers[-1]))  # 本轮对开冲突总数

    # 把训练统计结果（各轮总奖励/平均接单率/随机种子/冲突数/调度司机数）
    # 序列化保存到 results.pkl 文件
    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers,  # 各轮统计量列表
                 episode_dispatched_drivers], open(log_dir + "results.pkl", "wb"))  # 写入结果文件
    # 第 24 轮（最后一轮）跑完并保存结果后直接跳出循环、不再进行网络更新——
    # 相当于用最终模型做了一次纯评估
    if n_iter == 24:
        break

    # update value network
    # （原注释：更新值网络）
    # 值网络（critic）批量训练：用本轮回放池收集的样本共训练 4000 个批次，
    # 最小化价值输出与 TD 目标之间的均方误差，学习率 1e-3
    for _ in np.arange(4000):
        # 从值网络经验回放池随机采样一个批次（3000 条样本）
        batch_s, _, batch_r, _ = replay.sample()
        # 更新 critic：让价值估计向 TD 目标拟合；返回本批的损失值
        iloss = q_estimator.update_value(batch_s, batch_r, 1e-3, global_step1)
        # 值网络训练步数 +1
        global_step1 += 1

    # training method 2
    # （原注释：训练方式 2）
    # update policy network
    # （原注释：更新策略网络）
    # 策略网络（actor）批量训练：用本轮策略样本共训练 4000 个批次——
    # 策略梯度法：优势为正时增大所选动作的概率、为负时减小，
    # 并带熵正则项鼓励探索
    for _ in np.arange(4000):
        # 从策略回放池随机采样一个批次（状态/动作/优势/邻居掩码）
        batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
        # 更新 actor：按优势调整动作概率；batch_r.reshape([-1, 1]) 把优势
        # 从一维数组变成 n x 1 的列向量，以匹配网络的输出形状
        q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,  # 状态、优势、动作 one-hot、邻居掩码、学习率
                                  global_step2)  # 策略网络训练步数（用于 TensorBoard 日志）
        # 策略网络训练步数 +1
        global_step2 += 1



    # 保存当前模型参数到 model.ckpt（TensorFlow 检查点文件，每轮覆盖保存一次）
    saver.save(sess, log_dir+"model.ckpt")
    # 当随机种子为 54（即第 14 轮：14 + 50 - 10 = 54）时，额外保存一份
    # "进入测试阶段前"的模型备份 model_before_testing.ckpt
    if RANDOM_SEED == 54:
        saver.save(sess, log_dir + "model_before_testing.ckpt")
