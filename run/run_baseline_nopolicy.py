
# 导入 pickle（Python 对象序列化/反序列化模块，用于读数据和保存结果）和 sys（用于修改模块搜索路径）
import pickle, sys
# 导入 pathlib.Path 类：以面向对象方式处理文件路径，跨平台兼容
from pathlib import Path
# 把本文件所在目录的上一级（即项目根目录 Simulator-master）插入 sys.path 最前面，
# 保证无论从哪个目录运行脚本，都能找到根目录下的 simulator 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 星号导入 simulator.envs 模块：引入 CityReal 城市仿真环境类，
# 以及该模块内部转手引入的 np（NumPy）、time（时间处理）、mkdir_p（创建目录）等名字
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

# 仿真参数数据目录（对应 2017-07-24 至 2017-08-20 的真实数据）
data_dir = dir_prefix + "dispatch_realdata/data_for_simulator2017-07-24_2017-08-20/"
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
order_filename = dir_prefix + "dispatch_realdata/order_new_2017-07-24_2017-08-20/all_orders_target"  # 真实订单数据文件路径（约 4 周的订单）
order_real = pickle.load(open(order_filename, 'rb'), encoding='latin1')  # 真实订单数组：每条 [起点网格, 终点网格, 起始时刻, 时长(步), 价格]
# 从地图矩阵取出行数 M 和列数 N
M, N = mapped_matrix_int.shape
# 打印提示：数据加载完成
print("finish load data")


################## Initialize env ###################################
# 邻域边数：6 表示六边形邻域（每个网格最多 6 个邻居方向）
n_side = 6
# 折扣因子 GAMMA（强化学习里衡量未来回报重要性的系数，0.9 表示下一步的回报打 9 折；本无策略基线未使用）
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
# 再次确保日志目录存在（幂等操作，已存在则不报错）
mkdir_p(log_dir)

# 状态矩阵形状为 (2, M, N)：[0] 是空闲司机分布、[1] 是订单分布；展平后前 M*N 个元素
# 属于司机部分、后 M*N 个属于订单部分。target_grids 是有效网格编号（0 ~ M*N-1），
# 加上 M*N 即得到同一批网格在展平状态"订单部分"中的下标
temp = np.array(env.target_grids) + env.M * env.N
# 拼接两个下标列表 = [各有效网格司机数下标] + [各有效网格订单数下标]，
# 可用于从展平状态中一次取出有效网格的司机/订单分布（本基线未使用，供学习脚本使用）
target_id_states = env.target_grids + temp.tolist()

# 打印分隔提示：一天的订单已生成完毕
print("******************* Finish generating one day order **********************")



# 打印分隔提示：开始运行无策略基线
print("******************* Starting runing no policy baseline **********************")

# 最大迭代数（本基线不训练，该值仅用于给随机种子加偏移；行尾英文为原注释）
MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
# 是否绘图的开关（本脚本未使用）
is_plot_figure = False
# 每个 episode（一轮从初始状态到结束的完整交互，这里一轮 = 仿真完整一天）的起始城市时刻
city_time_start = 0
# episode 长度：144 步 x 10 分钟 = 一整天
EP_LEN = 144
# 全局步数计数器（统计用）
global_step = 0
# 结束时刻 = 起始时刻 + 144（本脚本未使用）
city_time_end = city_time_start + EP_LEN
# 探索率 epsilon（ε-greedy 策略参数：以概率 ε 随机选动作、以 1-ε 选最优动作；本基线不学习，未使用）
epsilon = 0.5
# 折扣因子 gamma（含义同上面的 GAMMA；本基线未使用）
gamma = 0.9
# 学习率（神经网络参数每次更新的步长；本基线未使用）
learning_rate = 1e-3

# 上一轮 episode 的奖励（占位变量，未使用；变量名拼写沿用原代码）
prev_epsiode_reward = 0
# 每步调度司机数的记录列表（无策略基线不调度，占位未使用）
curr_num_actions = []
# 跨所有 episode 累计的每步奖励
all_rewards = []
# 每轮 episode 的平均订单接单率
order_response_rate_episode = []
# Q 值表总和记录（表格型 Q-learning 算法的遗留变量，未使用）
value_table_sum = []
# 每轮 episode 的总奖励（当天成交总额）
episode_rewards = []
# 调度冲突司机数记录（无调度，占位未使用）
num_conflicts_drivers = []
# 每轮 episode 的空闲司机数总和
driver_numbers_episode = []
# 每轮 episode 的订单数总和
order_numbers_episode = []

# 一天的时间步总数（每步 10 分钟）
T = 144
# 动作维度：6 个邻居方向 + 原地不动（本基线未使用）
action_dim = 7
# 每个网格样本的状态特征维度 = 司机分布 + 订单分布 + 网格 one-hot + 时刻 one-hot（本基线未使用）
state_dim = env.n_valid_grids * 3 + T

# 记录每轮 episode 逐步的接单率曲线
record_all_order_response_rate = []


# 函数：从预匹配上下文 info 中抽取各有效网格的剩余空闲司机数
# 参数 target_grids：有效网格编号列表；参数 info：形如 (2, M, N) 的数组（[0] 剩余司机、[1] 剩余订单）
def compute_context(target_grids, info):

    # 展平成一维数组：前 M*N 个元素是各网格剩余空闲司机数，后 M*N 个是剩余订单数
    context = info.flatten()
    # 按有效网格编号取出各网格的剩余空闲司机数
    context = [context[idx] for idx in target_grids]
    # 返回各有效网格的剩余空闲司机数列表
    return context

# 初始空闲司机数量的缩放比例（1 = 按原始数量，不缩放）
RATIO = 1

# 打印提示：开始运行
print("Start Running ")
# 记录各轮 episode 使用的随机种子
save_random_seed = []
# 记录各轮 episode 的可用（未匹配）空闲司机总数
episode_avaliables_vehicles = []
# 外层循环：跑 10 轮 episode（episode 即一轮从初始状态到结束的完整交互，这里一轮 = 仿真完整一天），
# 每轮用不同的随机种子，相当于重复仿真 10 天
for n_iter in np.arange(10):
    # 本轮随机种子 = 轮次 + MAX_ITER + 5（即 55~64），保证每轮结果可复现且互不相同
    RANDOM_SEED = n_iter + MAX_ITER + 5
    # 把该随机种子写入仿真环境（控制环境内部的随机数行为）
    env.reset_randomseed(RANDOM_SEED)
    # 记录本轮使用的随机种子
    save_random_seed.append(RANDOM_SEED)
    # 状态/动作/奖励批缓冲（无策略基线不收集训练样本，占位未使用）
    batch_s, batch_a, batch_r = [], [], []
    # 本轮逐步奖励记录（GMV：Gross Merchandise Value，成交订单总金额）
    batch_reward_gmv = []
    # 本轮奖励累计变量（占位未使用，拼写沿用原代码）
    epsiode_reward = 0
    # 被调度司机数计数（无策略基线不调度，占位未使用）
    num_dispatched_drivers = 0

    # 本轮逐步的空闲司机总数
    driver_numbers = []
    # 本轮逐步的订单总数
    order_numbers = []
    # 订单生成模式标志：1 表示重置时重新抽样生成一天的订单
    is_regenerate_order = 1
    # 重置环境开始新的一天：清空全部司机与订单、按采样概率 1/28 从 4 周真实订单中抽出一天的订单、
    # 按 ratio=1 初始化空闲司机总数、从 city_time_start=0 时刻开始；
    # 返回初始状态 curr_state = [空闲司机分布矩阵, 订单分布矩阵]（均为 MxN）
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    # 记录初始空闲司机总数（state[0] 是司机分布矩阵）
    driver_numbers.append(np.sum(curr_state[0]))
    # 记录初始订单总数（state[1] 是订单分布矩阵）
    order_numbers.append(np.sum(curr_state[1]))
    # 预匹配：模拟每个网格用自身及邻居的空闲司机去抵消订单，
    # 返回 [剩余空闲司机, 剩余订单] 两个 MxN 矩阵
    info = env.step_pre_order_assigin(curr_state)
    # 取出各有效网格的剩余空闲司机数（即还没被订单占用的可用司机）
    context = compute_context(env.target_grids, np.array(info))

    # 本轮逐步的即时奖励记录（后续未参与计算；下方英文为原注释）
    # record rewards to update the value table
    episodes_immediate_rewards = []
    # 本轮逐步的订单接单率（成功匹配订单数 / 当步总订单数）
    order_response_rates = []
    # 本轮逐步的可用（未匹配）空闲司机总数
    available_drivers = []
    # 内层主循环：推进 EP_LEN+1 = 145 步，仿真一整天（多出的 1 步沿用学习脚本的写法）
    for ii in np.arange(EP_LEN + 1):
        # 记录当前时刻各有效网格剩余空闲司机总数
        available_drivers.append(np.sum(context))
        # ONE STEP: r0
        # 推进一个时间步：调度动作传空列表（不移动任何司机，即"无策略"基线），
        # generate_order=2 表示使用 reset 时抽好的一天订单序列（而非随机生成）；
        # 返回下一状态、本步奖励 r（= 匹配订单价格之和，即当天 GMV）和 info
        next_state, r, info = env.step([], 2)
        # 记录本步之后的空闲司机总数
        driver_numbers.append(np.sum(next_state[0]))
        # 记录本步之后的订单总数
        order_numbers.append(np.sum(next_state[1]))

        # info[1] 是 step 内部算好的预匹配上下文，据此更新各有效网格的剩余空闲司机数
        context = compute_context(env.target_grids, np.array(info[1]))
        # Perform gradient descent update
        # book keeping
        # 全局步数 +1
        global_step += 1
        # 把本步奖励计入全程累计奖励
        all_rewards.append(r)
        # 把本步奖励计入本轮 GMV
        batch_reward_gmv.append(r)
        # 记录本步订单接单率
        order_response_rates.append(env.order_response_rate)

    # 本轮总奖励 = 逐步奖励求和（去掉第 0 个记录值）
    episode_reward = np.sum(batch_reward_gmv[1:])
    # 存入每轮总奖励列表
    episode_rewards.append(episode_reward)
    # 本轮空闲司机数总和（去掉最后一个记录值）
    driver_numbers_episode.append(np.sum(driver_numbers[:-1]))
    # 本轮订单数总和（去掉最后一个记录值）
    order_numbers_episode.append(np.sum(order_numbers[:-1]))
    # 本轮可用司机数总和（去掉最后一个记录值）
    episode_avaliables_vehicles.append(np.sum(available_drivers[:-1]))
    # 本轮平均订单接单率（去掉第 0 个记录值后求均值）
    n_iter_order_response_rate = np.mean(order_response_rates[1:])
    # 存入每轮平均接单率列表
    order_response_rate_episode.append(n_iter_order_response_rate)
    # 保存本轮逐步接单率曲线
    record_all_order_response_rate.append(order_response_rates)

    # 打印本轮统计：轮次、总奖励（GMV）、平均接单率、可用司机总数
    print("******** iteration {} ********* reward {}, order response rate {} available vehicle {}".format(n_iter,
                                                                                                          episode_reward,
                                                                                        n_iter_order_response_rate,
                                                                                        episode_avaliables_vehicles[-1]))

    # 把各项统计结果用 pickle 序列化保存到日志目录的 results.pkl（"wb" 表示二进制写模式），
    # 列表元素依次为：每轮总奖励 / 每轮平均接单率 / 各轮随机种子 / 每轮空闲司机数总和 /
    # 每轮订单数总和 / 每轮可用司机数总和
    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed,
                 driver_numbers_episode, order_numbers_episode, episode_avaliables_vehicles], open(log_dir + "results.pkl", "wb"))


# 输出平均每个时间步的可用空闲司机数（各轮可用司机总数的均值再除以一天的 144 步）
print("averaged available vehicles per time step: {}".format(np.mean(episode_avaliables_vehicles)/144.0))
