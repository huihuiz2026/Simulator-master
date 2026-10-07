
# 导入 sys 模块：用于把项目根目录加入模块搜索路径，保证 simulator 包能被导入
import sys
# 导入 pathlib.Path 类：以面向对象方式处理文件路径，跨平台兼容
from pathlib import Path

# 导入 NumPy 数值计算库并简写为 np：用于构建矩阵和数组运算
import numpy as np

# 把本文件所在目录的上一级（即项目根目录 Simulator-master）插入 sys.path 最前面，
# 这样无论从哪个目录运行脚本，都能找到根目录下的 simulator 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 从 simulator/envs.py 导入 CityReal 类：城市仿真环境（网格地图 + 司机 + 订单 + 匹配调度）
from simulator.envs import CityReal


def build_example_env():
    """用人工构造的数据搭建一个 3x3 网格小城市的 CityReal 仿真环境"""

    # 城市地图：3x3 矩阵，-100 表示无效网格（不可通行、不产生订单），
    # 其余正整数是有效网格的编号（本例共 8 个有效网格）
    mapped_matrix_int = np.array([[1, -100, 3], [5, 4, 2], [6, 7, 8]])
    # 从地图形状取出行数 M 和列数 N（这里均为 3），创建环境时使用
    M, N = mapped_matrix_int.shape
    # 订单数量分布参数：每个时刻一个字典 {网格编号: [泊松均值]}，
    # 供随机生成订单的流程使用；本例主循环走预生成订单模式，此参数仅作占位
    order_num_dist = []
    # 有效网格数量：9 格中有 1 格为 -100 无效格，故为 8
    num_valid_grid = 8
    # 144 个时刻 x 8 个有效网格的矩阵：记录每个时刻各网格应有的空闲司机数量，
    # 供"按网格"统计司机数的模式使用（本例默认全局模式，作占位）
    idle_driver_location_mat = np.zeros((144, 8))

    # 遍历一天中的 144 个时间步（每步 10 分钟，共 24 小时）
    for ii in np.arange(144):
        # 为当前时刻建立 "网格编号 -> 订单数量分布参数" 的字典
        time_dict = {}
        # 遍历地图上全部 M*N=9 个网格编号（含无效格，统一占位）
        for jj in np.arange(M*N):  # num of grids
            # 该网格每步生成订单数的泊松均值设为 2（[2] 即 [mu]）
            time_dict[jj] = [2]
        # 把当前时刻的分布参数字典追加进总列表
        order_num_dist.append(time_dict)
        # 当前时刻：8 个有效网格各配置 2 个空闲司机
        idle_driver_location_mat[ii, :] = [2] * num_valid_grid

    # 全城空闲司机总数的目标分布：每个时刻 [均值, 标准差]，
    # 环境初始化/reset 时按正态分布 N(10, 1) 采样当天初始的空闲司机总数
    idle_driver_dist_time = [[10, 1] for _ in np.arange(144)]

    # 邻域边数：6 表示六边形邻域（每个网格最多有 6 个邻居方向）
    n_side = 6
    # 最大调度距离：空闲司机重调度/订单跨网格最多跨越 2 层网格
    l_max = 2
    # 订单时长的概率分布：时长 10~90 分钟（即 1~9 步）共 9 档，
    # 概率依次为 0.2、0.2、0.15、0.15、0.1、0.1、0.05、0.04、0.01（总和为 1）
    order_time = [0.2, 0.2, 0.15,
                  0.15,  0.1,  0.1,
                  0.05, 0.04,  0.01]
    # 各时长档位对应的订单价格分布 [均值, 标准差]：
    # 例如时长 10 分钟的订单价格 ~ N(10.17, 3.34)，时长 90 分钟 ~ N(79.40, 44.39)
    order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                   [15.02, 6.90],  # mean and std of order price when duration is 20 min
                   [23.22, 11.63],  # 时长 30 分钟档
                   [32.14, 16.20],  # 时长 40 分钟档
                   [40.99, 20.69],  # 时长 50 分钟档
                   [49.94, 25.61],  # 时长 60 分钟档
                   [58.98, 31.69],  # 时长 70 分钟档
                   [68.80, 37.25],  # 时长 80 分钟档
                   [79.40, 44.39]]  # 时长 90 分钟档

    # 预生成的"真实订单"列表：每条为 [起点网格, 终点网格, 起始时刻, 时长(步), 价格]
    # 起点/终点用地图中的网格编号（1~8）表示；起点为 9、10 等图外编号的订单
    # 属于"城外进入订单"——司机送客进入城市，订单完成时司机出现在终点网格
    order_real = []
    # 司机上下线控制矩阵：每个时刻是 8 个有效网格（按 target_node_ids 顺序）的
    # [均值, 标准差] 列表；每步对每个网格采样 N(均值, 标准差) 并四舍五入：
    # >0 则该网格上线相应数量的司机，<0 则下线相应数量的司机
    onoff_driver_location_mat = []
    # 遍历 144 个时间步，逐时刻填充订单列表和上下线矩阵
    for tt in np.arange(144):
        # 为时刻 tt 追加 9 条订单：如 [1, 5, tt, 1, 13.2] 表示
        # 网格 1 -> 网格 5、tt 时刻发起、时长 1 步（10 分钟）、价格 13.2
        order_real += [[1, 5, tt, 1, 13.2], [2, 6, tt, 1, 13.2], [3, 1, tt, 1, 13.2],
                       [5, 3, tt, 1, 13.2], [6, 2, tt, 1, 13.2], [7, 9, tt, 1, 13.2],
                       [9, 10, tt, 1, 13.2], [10, 6, tt, 1, 13.2], [9, 7, tt, 1, 13.2]]
        # 时刻 tt 各网格的 [上线数-下线数] 均值与标准差：
        # 第 1 格均值 -0.625（略偏下线），其余均值约 +0.09（略偏上线），
        # 整体效果是城市司机数量随时间缓慢净增长
        onoff_driver_location_mat.append([[-0.625,       2.92350389],
                                         [0.09090909,  1.46398452],
                                         [0.09090909,  2.36596622],
                                         [0.09090909, 2.36596622],
                                         [0.09090909, 1.46398452],
                                         [0.09090909, 1.46398452],
                                         [0.09090909, 1.46398452],
                                         [0.09090909, 1.46398452],
                                         [0.09090909, 1.46398452]])
    # 用以上人工数据构造 CityReal 城市仿真环境，参数依次为：
    # 地图矩阵、订单数量分布、空闲司机数分布、空闲司机位置矩阵、
    # 订单时长分布、订单价格分布、最大调度距离、行数 M、列数 N、邻域边数、
    # 订单采样概率（1 表示预生成订单全部选用）、预生成订单数组、司机上下线矩阵
    env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 1, np.array(order_real), np.array(onoff_driver_location_mat))
    # 返回构造好的环境对象
    return env


def running_example():
    """无调度策略基线：在示例环境中不调度空车地跑完一天，统计订单接单率"""

    # 构造示例城市环境
    env = build_example_env()

    # 重置环境：清空司机与订单、按采样概率载入一天的订单序列、初始化司机分布，
    # 返回初始状态 state = [空闲司机分布矩阵, 订单分布矩阵]（均为 MxN）
    state = env.reset_clean()
    # 记录每个时间步的订单接单率（成功匹配的订单数 / 当步总订单数）
    order_response_rates = []
    # 仿真步数计数器
    T = 0
    # 总仿真步数：144 步 x 10 分钟 = 24 小时
    max_iter = 144
    # 仿真主循环：模拟一整天
    while T < max_iter:
        # 可选逻辑（已注释）：每 5 步重置一次环境并按比例生成订单
        # if T % 5 == 0:
        #     state = env.reset_clean(generate_order=2)
        # 调度动作列表：空列表表示不移动任何空闲司机（即"无策略"基线；
        # 学习算法会输出形如 [司机ID, 目标网格] 的调度对填入此处）
        dispatch_action = []
        # 推进一个时间步：执行调度动作 -> 匹配订单并计算奖励 -> 时间+1 ->
        # 生成新订单 -> 司机上下线 -> 返回下一状态；
        # generate_order=2 表示使用预生成的固定订单序列（而非随机生成）
        state, reward, _ = env.step(dispatch_action, generate_order=2)

        # 打印本步的接单率（city_time 在 step 内已 +1，故 -1 还原为本步所处的时刻）
        print("City time {}: Order response rate: {}".format(env.city_time-1, env.order_response_rate))
        # 把本步接单率记入列表，供最后求全天平均
        order_response_rates.append(env.order_response_rate)

        # 打印三个司机统计量：状态矩阵中的空闲司机总数、逐网格累加的空闲司机数、
        # 系统注册的司机总数（含正在送客和已下线的司机）；前两者应相等
        print(("idle driver: {} == {} total num of drivers: {}".format(np.sum(state[0]),
                                                                      np.sum(env.get_observation_driver_state()),
                                                                      len(env.drivers))))

        # 一致性断言：状态中的空闲司机总数必须等于环境记录的空闲司机数，
        # 校验模拟器内部状态没有出现不一致（漏数/重复计数）
        assert np.sum(state[0]) == env.n_drivers

        # 仿真步数 +1
        T += 1
    # 仿真结束：打印全天 144 步的平均订单接单率
    print(np.mean(order_response_rates))


# 作为脚本直接运行时才执行以下代码（被其他模块导入时不执行）
if __name__ == "__main__":
    # 调用主流程：开始一天的仿真并输出每步接单率与全天平均接单率
    running_example()
