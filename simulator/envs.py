# 导入 os、sys、random、time 四个标准库（本文件实际很少直接用到，主要为兼容保留）
import os, sys, random, time
# 导入日志模块 logging：仿真过程中的调试/运行信息通过它输出
import logging

# 从同包 objects.py 导入全部公开对象：Node（网格节点）、Driver（司机）、Order（订单）等类
from .objects import *
# 从同包 utilities.py 导入工具函数：一二维编号互换 ids_2dto1d/ids_1dto2d、
# 邻居列表 get_neighbor_list、分层邻居 get_layers_neighbors、时间序列 datetime_range 等
from .utilities import *
# from algorithm import *

# current_time = time.strftime("%Y%m%d_%H-%M")
# log_dir = "/nfs/private/linkaixiang_i/data/dispatch_simulator/experiments/"+current_time + "/"
# mkdir_p(log_dir)
# logging.basicConfig(filename=log_dir +'logger_env.log', level=logging.INFO)

# 创建本模块的日志器（__name__ 即模块名 "simulator.envs"）
logger = logging.getLogger(__name__)
# 日志级别设为 DEBUG：输出最详细的日志
logger.setLevel(logging.DEBUG)
# 创建控制台日志处理器：把日志打印到标准输出（屏幕）
logger_ch = logging.StreamHandler()
# 控制台处理器同样输出 DEBUG 及以上级别
logger_ch.setLevel(logging.DEBUG)
# 设置日志格式："时间[级别][行号:函数名]||消息"，时间精确到秒（下方两行为格式参数）
logger_ch.setFormatter(logging.Formatter(
    '%(asctime)s[%(levelname)s][%(lineno)s:%(funcName)s]||%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
# 把控制台处理器挂载到日志器，日志配置完成
logger.addHandler(logger_ch)
# 全局默认随机种子 0；CityReal 构造时会拷贝为实例属性 RANDOM_SEED
RANDOM_SEED = 0  # unit test use this random seed.

# ======================= CityReal：城市仿真环境类 =======================
# 真实城市由 MxN 个网格组成；本类管理全部网格节点、司机与订单，
# 对外提供 reset_clean（重置一天）/ step（推进一步）接口，是强化学习算法调用的"环境"
class CityReal:
    '''A real city is consists of M*N grids '''

    # 构造函数：搭建城市网格结构，保存全部仿真参数
    # 参数 mapped_matrix_int：MxN 地图矩阵，-100=无效网格，其余正整数=网格编号
    # 参数 order_num_dist：144 个时刻的订单数量分布，每项为 {节点索引: [泊松均值]} 字典
    # 参数 idle_driver_dist_time：144 个时刻全城空闲司机总数的 [均值, 标准差]（global 模式用）
    # 参数 idle_driver_location_mat：144 x 有效网格数矩阵，各时刻每格空闲司机数（逐格模式用）
    # 参数 order_time_dist：订单时长 1~9 步（10~90 分钟）各自的出现概率
    # 参数 order_price_dist：各时长档位的订单价格 [均值, 标准差]
    # 参数 l_max：订单/调度最多可跨越的网格层数；M、N：地图行、列数；n_side：邻域边数（6=六边形）
    # 参数 probability：从 real_orders 采样订单的概率 p；real_orders：真实历史订单数组
    # 参数 onoff_driver_location_mat：144 x 有效网格数 x 2，每格上下线净数量的 [均值, 标准差]
    # 参数 global_flag："global" 按全城总数控制司机数，否则按逐格矩阵；time_interval：每步分钟数
    def __init__(self, mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat, order_time_dist, order_price_dist,
                 l_max, M, N, n_side, probability=1.0/28, real_orders="", onoff_driver_location_mat="",
                 global_flag="global", time_interval=10):
        """
        :param mapped_matrix_int: 2D matrix: each position is either -100 or grid id from order in real data.
        :param order_num_dist: 144 [{node_id1: [mu, std]}, {node_id2: [mu, std]}, ..., {node_idn: [mu, std]}]
                            node_id1 is node the index in self.nodes
        :param idle_driver_dist_time: [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle drivers in
        the city at each time
        :param idle_driver_location_mat: 144 x num_valid_grids matrix.
        :param order_time_dist: [ 0.27380797,..., 0.00205766] The probs of order duration = 1 to 9
        :param order_price_dist: [[10.17, 3.34],   # mean and std of order's price, order durations = 10 minutes.
                                   [15.02, 6.90],  # mean and std of order's price, order durations = 20 minutes.
                                   ...,]
        :param onoff_driver_location_mat: 144 x 504 x 2: 144 total time steps, num_valid_grids = 504.
        mean and std of online driver number - offline driver number
        onoff_driver_location_mat[t] = [[-0.625       2.92350389]  <-- Corresponds to the grid in target_node_ids
                                        [ 0.09090909  1.46398452]
                                        [ 0.09090909  2.36596622]
                                        [-1.2         2.05588586]...]
        :param M:
        :param N:
        :param n_side:
        :param time_interval:
        :param l_max: The max-duration of an order
        :return:
        """
        # City.__init__(self, M, N, n_side, time_interval)
        # 城市地图行数
        self.M = M  # row numbers
        # 城市地图列数
        self.N = N  # column numbers
        # 先为全部 M*N 个网格各建一个 Node（节点编号从 0 开始）——注意：该列表随后
        # 会在下方被重置为全 None，再由 construct_node_real 只为有效网格重建，
        # 因此这一步属于会被覆盖的过渡操作（历史遗留）
        self.nodes = [Node(i) for i in range(M * N)]  # a list of nodes: node id start from 0
        # 司机注册表：driver_id -> Driver 实例；记录注册总数（含在线、送客中、离线司机）
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        # 当前"在线且空闲"的司机数：只统计可接单司机，不含送客中和离线司机
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        # 当前离线司机总数
        self.n_offline_drivers = 0  # total number of offline drivers.
        # 过渡步骤：先在 MxN 全网格上连接一遍邻居（结果随后会被覆盖）
        self.construct_map_simulation(M, N, n_side)
        # 城市当前时间步（一天共 n_intervals=144 步，每步 10 分钟）
        self.city_time = 0
        # self.idle_driver_distribution = np.zeros((M, N))
        # 一天的总时间步数：1440 分钟整除每步 10 分钟 = 144
        self.n_intervals = 1440 // time_interval
        # 城市总网格数 M*N（含无效网格）
        self.n_nodes = self.M * self.N
        # 邻域边数：6 表示六边形邻域（每个网格最多 6 个邻居方向）
        self.n_side = n_side
        # 当前时间步的订单接单率（完成订单数/总订单数；无订单时记 -1）
        self.order_response_rate = 0

        # 把模块级随机种子拷贝为实例属性（可被 reset_randomseed 修改）
        self.RANDOM_SEED = RANDOM_SEED

        # 最大调度/订单跨越层数：订单时长与空车调度最多跨 l_max 层网格（从 1 起）
        self.l_max = l_max  # Start from 1. The max number of layers an order can across.
        # 校验：l_max 不得超过行数、列数减 1（保证分层邻居仍落在地图内）
        assert l_max <= M-1 and l_max <= N-1
        # 校验：l_max 限定在 1~9（对应忽略时长小于 10 分钟或大于 1.5 小时的订单）
        assert 1 <= l_max <= 9   # Ignore orders less than 10 minutes and larger than 1.5 hours

        # 有效网格的节点索引列表（先置空，由 construct_node_real 按地图扫描顺序填充）
        self.target_grids = []
        # 有效网格数量（先置 0，由 construct_node_real 统计）
        self.n_valid_grids = 0  # num of valid grid
        # 把节点列表整体重置为 M*N 个 None：之后只有地图编号 >= 0 的有效格
        # 才会在 construct_node_real 中真正创建 Node，无效格保持 None 占位
        self.nodes = [None for _ in np.arange(self.M * self.N)]
        # 依据地图矩阵重建节点：创建有效格节点、构建分层邻居、统计有效格信息
        self.construct_node_real(mapped_matrix_int)
        # 保存原始地图矩阵（-100=无效格，正整数=网格编号）供后续映射使用
        self.mapped_matrix_int = mapped_matrix_int

        # 在有效节点之间重新建立邻居连接（覆盖此前 construct_map_simulation 的结果）
        self.construct_map_real(n_side)
        # 保存订单数量分布参数（144 个时刻的 {节点索引: [泊松均值]} 字典列表）
        self.order_num_dist = order_num_dist
        # 订单数量采样使用的分布名称（默认泊松分布）
        self.distribution_name = "Poisson"
        # 保存全城空闲司机总数的时间分布（global 模式：每时刻 [均值, 标准差]）
        self.idle_driver_dist_time = idle_driver_dist_time
        # 保存逐格空闲司机数量矩阵（144 x 有效网格数）
        self.idle_driver_location_mat = idle_driver_location_mat

        # 订单时长分布：截取前 l_max 档并重新归一化，使概率和为 1
        self.order_time_dist = order_time_dist[:l_max]/np.sum(order_time_dist[:l_max])
        # 订单价格分布：各时长档位的 [均值, 标准差] 列表
        self.order_price_dist = order_price_dist

        # 有效节点的索引列表：按网格编号升序排列（target_grids 则按地图扫描顺序）
        target_node_ids = []
        # 提取地图中全部正整数网格编号并升序排序
        target_grids_sorted = np.sort(mapped_matrix_int[np.where(mapped_matrix_int > 0)])
        # 逐个编号定位它在地图中的位置并换算成节点索引
        for item in target_grids_sorted:
            # np.where 找到该编号所在的行 x、列 y（每个编号在图中唯一）
            x, y = np.where(mapped_matrix_int == item)
            # 二维坐标按行优先展开成一维节点索引后追加
            target_node_ids.append(ids_2dto1d(x, y, M, N)[0])
        # 保存按网格编号升序的有效节点索引列表
        self.target_node_ids = target_node_ids
        # store valid note id. Sort by number of orders emerged. descending.

        # 网格编号 -> 节点索引 的映射字典（把订单中的网格编号翻译成节点索引）
        self.node_mapping = {}
        # 构建 node_mapping：为每个有效网格编号记录其节点索引
        self.construct_mapping()

        # 真实历史订单数据（约 4 周量），每条为 [起点网格编号, 终点网格编号, 起始时刻, 时长(步), 价格]
        self.real_orders = real_orders  # 4 weeks' data
        # [[92, 300, 143, 2, 13.2],...] origin grid, destination grid, start time, end time, price.


        # 订单采样概率 p：reset 时按它对 real_orders 做伯努利采样得到一天订单
        self.p = probability   # sample probability
        # 一天 144 个时刻的 "HHMM" 时间字符串列表：从 2017-09-01 00:00 起、
        # 每 time_interval 分钟取一个、到次日 0:00 之前为止（用于与真实数据时刻对齐）
        self.time_keys = [int(dt.strftime('%H%M')) for dt in
                          datetime_range(datetime(2017, 9, 1, 0), datetime(2017, 9, 2, 0),
                                        timedelta(minutes=time_interval))]
        # 一天的订单容器：外层 144 个时刻、内层为该时刻的订单五元组（由 bootstrap 填充）
        self.day_orders = []  # one day's order.

        # 司机上下线控制矩阵：144 x 有效网格数 x 2，每格 [均值, 标准差]，
        # 每步正态采样后四舍五入：正值上线、负值下线
        self.onoff_driver_location_mat = onoff_driver_location_mat

        # Stats
        # 统计量：本步全网格累计上线的司机数
        self.all_grids_on_number = 0  # current online # drivers.
        # 统计量：本步全网格累计下线的司机数
        self.all_grids_off_number = 0


        # "城外进城订单"计数矩阵：144 时刻 x 有效网格数——起点在图外的订单不进入
        # 订单列表，而是在"起始时刻+时长"给终点格记一个进城司机（见 utility_bootstrap_oneday_order）
        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))
        # 司机数量控制模式："global" 按全城总数控制，其他值按逐格矩阵控制
        self.global_flag = global_flag
        # 邻居奖励广播的分层折扣权重：第 1/2/3 层邻居分别为 1、e^-1、e^-2
        # （可用 utility_set_neighbor_weight 重新设置）
        self.weights_layers_neighbors = [1.0, np.exp(-1), np.exp(-2)]


    # 在模拟的 MxN 全网格地图上为每个节点连接邻居（过渡用途，真实地图用 construct_map_real）
    # 参数 M、N：地图行、列数；n：n 边形邻域边数（6=六边形）
    def construct_map_simulation(self, M, N, n):
        """Connect node to its neighbors based on a simulated M by N map
            :param M: M row index matrix
            :param N: N column index matrix
            :param n: n - sided polygon
        """
        # 遍历全部网格位置（含 None 占位）
        for idx, current_node in enumerate(self.nodes):
            # 只处理真实存在的节点
            if current_node is not None:
                # 一维节点索引转回 (行 i, 列 j)
                i, j = ids_1dto2d(idx, M, N)
                # 按六边形/四边形邻域规则连接邻居（越界方向置 None）
                current_node.set_neighbors(get_neighbor_list(i, j, M, N, n, self.nodes))

    # 构建 node_mapping：地图网格编号 -> 节点索引 的映射字典
    def construct_mapping(self):
        """
        :return:
        """
        # 取出地图矩阵中全部正整数网格编号（按行扫描顺序）
        target_grid_id = self.mapped_matrix_int[np.where(self.mapped_matrix_int>0)]
        # 第 k 个网格编号与第 k 个有效节点索引配对（两者按同一扫描顺序排列，一一对应）
        for g_id, n_id in zip(target_grid_id, self.target_grids):
            # 写入映射：网格编号 g_id 对应节点索引 n_id
            self.node_mapping[g_id] = n_id

    # 依据真实地图初始化节点：只为地图编号 >= 0 的有效格创建 Node，
    # 为每个有效节点构建 l_max 层邻居索引，最后记录 target_grids 与有效格数
    def construct_node_real(self, mapped_matrix_int):
        """ Initialize node, only valid node in mapped_matrix_in will be initialized.
        """
        # 找出地图矩阵中所有 >= 0（即非 -100 无效）位置的行、列坐标
        row_inds, col_inds = np.where(mapped_matrix_int >= 0)

        # 有效节点索引列表（从 0 开始）
        target_ids = []  # start from 0.
        # 第一遍扫描：为每个有效位置创建 Node 并记下其一维索引
        for x, y in zip(row_inds, col_inds):
            # (行 x, 列 y) 转成一维节点索引
            node_id = ids_2dto1d(x, y, self.M, self.N)
            # 在节点列表对应位置创建节点（节点编号从 0 开始）
            self.nodes[node_id] = Node(node_id)  # node id start from 0.
            # 记录该有效节点索引
            target_ids.append(node_id)

        # 第二遍扫描：为每个有效节点构建 l_max 层邻居索引（逐层去重，剔除越界/无效格）
        for x, y in zip(row_inds, col_inds):
            # 二维坐标转一维节点索引
            node_id = ids_2dto1d(x, y, self.M, self.N)
            # 计算该节点 1~l_max 层邻居的节点索引并存入节点属性
            self.nodes[node_id].get_layers_neighbors(self.l_max, self.M, self.N, self)

        # 保存有效节点索引列表（按地图扫描顺序）
        self.target_grids = target_ids
        # 记录有效网格数量
        self.n_valid_grids = len(target_ids)
        
    # 在真实地图的有效节点之间建立邻居连接（n_side=6 时为六边形 6 邻域，越界/无效方向为 None）
    def construct_map_real(self, n_side):
        """Build node connection. 
        """
        # 遍历全部网格位置
        for idx, current_node in enumerate(self.nodes):
            # 一维索引转 (行 i, 列 j)
            i, j = ids_1dto2d(idx, self.M, self.N)
            # 只处理有效节点
            if current_node is not None:
                # 为该节点设置邻居列表（长度即 n_side，无效方向为 None）
                current_node.set_neighbors(get_neighbor_list(i, j, self.M, self.N, n_side, self.nodes))

    # 初始化各节点的订单生成器：为每个有效节点设置订单数量分布（泊松/高斯）及其参数
    # 参数 distribution_all：每个节点的分布名称列表；dis_paras_all：对应的分布参数列表
    def initial_order_random(self, distribution_all, dis_paras_all):
        """ Initialize order distribution
        :param distribution: 'Poisson', 'Gaussian'
        :param dis_paras:     lambda,    mu, sigma
        """
        # 遍历全部节点
        for idx, node in enumerate(self.nodes):
            # 跳过无效格
            if node is not None:
                # 为该节点创建订单数量分布生成器（泊松用均值，高斯用 mu/sigma）
                node.order_distribution(distribution_all[idx], dis_paras_all[idx])

    # 获取全局观测状态：返回 2xMxN 数组——通道 0 为各格空闲司机数、通道 1 为各格订单数
    def get_observation(self):
        # 初始化 2xMxN 的全零状态数组
        next_state = np.zeros((2, self.M, self.N))
        # 遍历全部节点
        for _node in self.nodes:
            # 跳过无效格（保持 0）
            if _node is not None:
                # 节点一维索引转 (行, 列)
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                # 通道 0 填入该格空闲司机数
                next_state[0, row_id, column_id] = _node.idle_driver_num
                # 通道 1 填入该格当前订单数
                next_state[1, row_id, column_id] = _node.order_num

        # 返回堆叠状态 [空闲司机分布, 订单分布]
        return next_state

    # 统计全城"在线且空闲"的司机总数：逐节点累加各格的 idle_driver_num
    def get_num_idle_drivers(self):
        """ Compute idle drivers
        :return:
        """
        # 空闲司机计数器清零
        temp_n_idle_drivers= 0
        # 遍历全部节点
        for _node in self.nodes:
            # 跳过无效格
            if _node is not None:
                # 累加该格的空闲司机数
                temp_n_idle_drivers += _node.idle_driver_num
        # 返回全城空闲司机总数
        return temp_n_idle_drivers

    # 获取空闲司机分布（MxN）：不依赖节点的 idle_driver_num 计数器，
    # 而是逐个遍历每格 drivers 字典、统计"在线且未在送客"的司机数（可用来交叉校验计数一致性）
    def get_observation_driver_state(self):
        """ Get idle driver distribution, computing #drivers from node.
        :return:
        """
        # 初始化 MxN 全零矩阵
        next_state = np.zeros((self.M, self.N))
        # 遍历全部节点
        for _node in self.nodes:
            # 跳过无效格
            if _node is not None:
                # 节点索引转 (行, 列)
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                # 逐个司机检查状态并计数，填入对应网格
                next_state[row_id, column_id] = _node.get_idle_driver_numbers_loop()

        # 返回空闲司机分布矩阵
        return next_state

    # 重设实例随机种子（影响后续 np.random.seed 的取值）
    def reset_randomseed(self, random_seed):
        # 更新实例随机种子
        self.RANDOM_SEED = random_seed

    # 旧版重置方法：只允许在城市时间 0 调用——按分布初始化司机数量与订单数量分布，
    # 生成第 0 步订单后返回初始观测（主流程请优先使用 reset_clean）
    def reset(self):
        """ Return initial observation: get order distribution and idle driver distribution

        """

        # 缓存行数 M
        _M = self.M
        # 缓存列数 N
        _N = self.N
        # 断言：只有城市时间 0（尚未开始仿真）时才能调用本方法
        assert self.city_time == 0
        # initialization drivers according to the distribution at time 0
        # 按时刻 0 的全城空闲司机分布采样目标司机数
        num_idle_driver = self.utility_get_n_idle_drivers_real()
        # 通过上/下线控制把空闲司机数调整到目标值（旧版控制逻辑）
        self.step_driver_online_offline_control(num_idle_driver)

        # generate orders at first time step
        # 为 M*N 个网格准备分布名称列表（全部为 self.distribution_name，即泊松）
        distribution_name = [self.distribution_name]*(_M*_N)
        # 取当前时刻的订单分布参数字典 {节点索引: [mu]}
        distribution_param_dictionary = self.order_num_dist[self.city_time]
        # 分布参数列表初始化为 0（无效格保持 0）
        distribution_param = [0]*(_M*_N)
        # 遍历参数字典，把各节点的参数搬进列表
        for key, value in list(distribution_param_dictionary.items()):
            # 高斯分布：参数为 (均值 mu, 标准差 sigma)
            if self.distribution_name == 'Gaussian':
                # 解包均值与标准差
                mu, sigma = value
                # 写入该节点的分布参数
                distribution_param[key] = mu, sigma
            # 泊松分布：参数为均值 mu（lambda）
            elif self.distribution_name == 'Poisson':
                # 取出均值参数
                mu = value[0]
                # 写入该节点的分布参数
                distribution_param[key] = mu
            else:
                # 未知分布名称：打印错误提示
                print("Wrong distribution")

        # 为每个有效节点创建订单数量分布生成器
        self.initial_order_random(distribution_name, distribution_param)
        # 按生成器随机生成当前时刻的订单
        self.step_generate_order_real()

        # 返回初始观测 [空闲司机分布, 订单分布]
        return self.get_observation()

    # 新版重置方法：可在任意时刻调用——①清空全部司机与各格订单；
    # ②generate_order==1 时按概率 p 从 real_orders 采样装满一天订单 day_orders；
    # ③按目标分布放置初始司机（global 模式按全城总数，否则按逐格矩阵）；
    # ④返回初始观测（2xMxN：[空闲司机分布, 订单分布]）
    # 参数 generate_order：1=重新采样一天订单，其他值=沿用现有 day_orders；
    # 参数 ratio：global 模式下司机目标数的缩放比例；city_time：可选的起始时刻
    def reset_clean(self, generate_order=1, ratio=1, city_time=""):
        """ 1. bootstrap oneday's order data.
            2. clean current drivers and orders, regenerate new orders and drivers.
            can reset anytime
        :return:
        """
        # 显式传入 city_time 时，把城市时间重设为该值（从指定时刻开始仿真）
        if city_time != "":
            self.city_time = city_time

        # clean orders and drivers
        # 重建空司机注册表（清空全部司机）
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        # 空闲司机数清零
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        # 离线司机数清零
        self.n_offline_drivers = 0  # total number of offline drivers.
        # 遍历全部节点，逐格清空订单与司机记录
        for node in self.nodes:
            # 跳过无效格
            if node is not None:
                # 清空该格的订单列表、订单数、司机表及各项计数
                node.clean_node()

        # Generate one day's order.
        # generate_order==1：按概率 p 采样真实订单，装满一天的 day_orders
        if generate_order == 1:
            self.utility_bootstrap_oneday_order()

        # Init orders of current time step
        # 当前时刻在一天 144 步中的位置（取模以支持跨天）
        moment = self.city_time % self.n_intervals
        # 把 day_orders 中该时刻的订单逐条装入对应起点网格
        self.step_bootstrap_order_real(self.day_orders[moment])

        # Init current driver distribution
        # global 模式：采样全城空闲司机总数并乘缩放比例作为目标
        if self.global_flag == "global":
            # 采样目标空闲司机总数 N(均值, 标准差) 并四舍五入
            num_idle_driver = self.utility_get_n_idle_drivers_real()
            # 按比例缩放目标司机数
            num_idle_driver = int(num_idle_driver * ratio)
        else:
            # 逐格模式：目标数取 idle_driver_location_mat 当时刻各格数量之和
            # （放置司机时按各格缺口加权，见 utility_add_driver_real_new_offlinefirst）
            num_idle_driver = self.utility_get_n_idle_drivers_nodewise()
        # 通过上/下线（优先唤醒离线司机）把空闲司机数调整到目标
        self.step_driver_online_offline_control_new(num_idle_driver)
        # 返回初始观测 [空闲司机分布, 订单分布]
        return self.get_observation()

    # 收集当前离线司机的编号列表（遍历 drivers 注册表，挑出 online 为 False 的司机）
    def utility_collect_offline_drivers_id(self):
        """count how many drivers are offline
        :return: offline_drivers: a list of offline driver id
        """
        # 离线司机计数器
        count = 0 # offline driver num
        # 离线司机编号列表
        offline_drivers = []   # record offline driver id
        # 遍历司机注册表
        for key, _driver in list(self.drivers.items()):
            # 在线状态为 False 即离线（含送客到图外退出、被调度下线等情形）
            if _driver.online is False:
                # 离线计数 +1
                count += 1
                # 记录该离线司机的编号
                offline_drivers.append(_driver.get_driver_id())
        # 返回离线司机编号列表
        return offline_drivers

    # 逐格模式的目标空闲司机数：取 idle_driver_location_mat 当时刻各格数量之和
    def utility_get_n_idle_drivers_nodewise(self):
        """ compute idle drivers.
        :return:
        """
        # 当前时刻在一天中的位置（取模跨天）
        time = self.city_time % self.n_intervals
        # 各有效网格的目标空闲数求和
        idle_driver_num = np.sum(self.idle_driver_location_mat[time])
        # 取整后返回目标空闲司机总数
        return int(idle_driver_num)


    # 增加若干空闲司机（旧版，不优先唤醒离线司机）：先算每个有效网格
    # "历史目标数 - 当前空闲数"的缺口，按缺口比例加权随机选格、创建全新司机放入
    # 参数 num_added_driver：要新增的司机数
    def utility_add_driver_real_new(self, num_added_driver):
        # 取当前空闲司机分布（观测通道 0，MxN 矩阵）
        curr_idle_driver_distribution = self.get_observation()[0]
        # 展平后按 target_node_ids（网格编号升序）顺序重排，与历史矩阵的列对齐
        curr_idle_driver_distribution_resort = np.array(
            [int(curr_idle_driver_distribution.flatten()[index]) for index in
             self.target_node_ids])

        # 当时刻各有效网格的目标空闲司机数（历史分布）
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        # 各格缺口 = 历史目标数 - 当前空闲数
        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort
        # 缺口非正的格置 0（已达标或超员的格不再放人）
        idle_diff[np.where(idle_diff <= 0)] = 0

        # 以缺口归一化为概率，从有效节点中随机抽取 num_added_driver 个放置格
        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff/float(np.sum(idle_diff)))

        # 当前司机注册总数（新司机编号从该值起连续分配）
        n_total_drivers = len(self.drivers)
        # 逐个创建司机并放入抽中的网格
        for ii, node_id in enumerate(node_ids):
            # 新司机编号 = 注册总数 + 序号
            added_driver_id = n_total_drivers + ii
            # 注册新司机实例
            self.drivers[added_driver_id] = Driver(added_driver_id)
            # 把司机的位置指向该节点
            self.drivers[added_driver_id].set_position(self.nodes[node_id])
            # 把司机加入该节点的司机表（节点空闲数 +1）
            self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])

        # 城市空闲司机计数增加新增数量
        self.n_drivers += num_added_driver

    # 增加空闲司机（离线优先版，reset_clean 使用）：与 utility_add_driver_real_new
    # 一样按各格缺口加权选格，但每个格先唤醒该格已有的离线司机，无离线司机时才创建新司机
    # 参数 num_added_driver：要新增的司机数
    def utility_add_driver_real_new_offlinefirst(self, num_added_driver):

        # curr_idle_driver_distribution = self.get_observation()[0][np.where(self.mapped_matrix_int > 0)]
        # 当前空闲司机分布（MxN 矩阵）
        curr_idle_driver_distribution = self.get_observation()[0]
        # 展平后按 target_node_ids（网格编号升序）顺序重排，与历史矩阵的列对齐
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index]) for index in
                                                         self.target_node_ids])

        # 当时刻各有效网格的目标空闲司机数（历史分布）
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        # 各格缺口 = 历史目标数 - 当前空闲数
        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort
        # 缺口非正的格置 0
        idle_diff[np.where(idle_diff <= 0)] = 0

        # 所有格都无缺口：无需加人，直接返回
        if float(np.sum(idle_diff)) == 0:
            return
        # 固定随机种子，保证采样可复现
        np.random.seed(self.RANDOM_SEED)
        # 以缺口归一化为概率，随机抽取 num_added_driver 个放置格
        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff/float(np.sum(idle_diff)))

        # 逐个处理要新增的司机
        for ii, node_id in enumerate(node_ids):

            # 该格存在离线司机：优先唤醒一个（不新建对象）
            if self.nodes[node_id].offline_driver_num > 0:
                # 唤醒该格一个离线司机（节点空闲数 +1、离线数 -1）
                self.nodes[node_id].set_offline_driver_online()
                # 城市空闲司机数 +1
                self.n_drivers += 1
                # 城市离线司机数 -1
                self.n_offline_drivers -= 1
            else:

                # 该格无离线司机：创建全新司机（编号取当前注册总数）
                n_total_drivers = len(self.drivers)
                # 新司机编号
                added_driver_id = n_total_drivers
                # 注册新司机实例
                self.drivers[added_driver_id] = Driver(added_driver_id)
                # 设置其位置为该节点
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                # 把司机加入该节点的司机表
                self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
                # 城市空闲司机数 +1
                self.n_drivers += 1

    # 向指定网格增加若干空闲司机（逐格版）：优先唤醒该格的离线司机，没有时才创建新司机
    # 参数 node_id：目标节点索引；num_added_driver：要增加的司机数
    def utility_add_driver_real_nodewise(self, node_id, num_added_driver):


        # 还有待增加的司机就继续循环
        while num_added_driver > 0:
            # 该格有离线司机：唤醒一个
            if self.nodes[node_id].offline_driver_num > 0:
                # 唤醒该格一个离线司机（节点空闲数 +1、离线数 -1）
                self.nodes[node_id].set_offline_driver_online()
                # 城市空闲司机数 +1
                self.n_drivers += 1
                # 城市离线司机数 -1
                self.n_offline_drivers -= 1
            else:

                # 无离线司机：创建新司机，编号取当前注册总数
                n_total_drivers = len(self.drivers)
                # 新司机编号
                added_driver_id = n_total_drivers
                # 注册新司机实例
                self.drivers[added_driver_id] = Driver(added_driver_id)
                # 设置其位置为该节点
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                # 把司机加入该节点的司机表
                self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
                # 城市空闲司机数 +1
                self.n_drivers += 1
            # 剩余待增加数 -1
            num_added_driver -= 1

    # 把指定网格的若干空闲司机下线（随机挑选；该格已无空闲司机时提前终止）
    # 参数 node_id：目标节点索引；n_drivers_to_off：要下线的司机数
    def utility_set_drivers_offline_real_nodewise(self, node_id, n_drivers_to_off):

        # 还有待下线司机就继续循环
        while n_drivers_to_off > 0:
            # 该格仍有空闲司机
            if self.nodes[node_id].idle_driver_num > 0:
                # 随机把该格一个空闲司机设为离线
                self.nodes[node_id].set_idle_driver_offline_random()
                # 城市空闲司机数 -1
                self.n_drivers -= 1
                # 城市离线司机数 +1
                self.n_offline_drivers += 1
                # 剩余待下线数 -1
                n_drivers_to_off -= 1
                # 全网累计下线统计 +1
                self.all_grids_off_number += 1
            else:
                # 该格已无空闲司机，终止循环
                break

    # 全城下线若干空闲司机（加权版）：计算各格"当前空闲数 - 历史目标数"的盈余，
    # 按盈余加权随机选格，把选中的格里一个空闲司机下线
    # 参数 n_drivers_to_off：要下线的司机数
    def utility_set_drivers_offline_real_new(self, n_drivers_to_off):


        # 当前空闲司机分布（观测通道 0）
        curr_idle_driver_distribution = self.get_observation()[0]
        # 展平后按 target_node_ids（网格编号升序）顺序重排，与历史矩阵对齐
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index])
                                                         for index in self.target_node_ids])

        # historical idle driver distribution
        # 当时刻各有效网格的目标空闲司机数（历史分布）
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        # diff of curr idle driver distribution and history
        # 各格盈余 = 当前空闲数 - 历史目标数
        idle_diff = curr_idle_driver_distribution_resort - idle_driver_distribution.astype(int)
        # 盈余非正的格置 0（未超过历史水平的格不参与下线）
        idle_diff[np.where(idle_diff <= 0)] = 0

        # 可下线上限：盈余 >= 0 的那些格的当前空闲司机总数
        n_drivers_can_be_off = int(np.sum(curr_idle_driver_distribution_resort[np.where(idle_diff >= 0)]))
        # 要下线数超过上限时截断到上限
        if n_drivers_to_off > n_drivers_can_be_off:
            n_drivers_to_off = n_drivers_can_be_off

        # 全城盈余总数
        sum_idle_diff = np.sum(idle_diff)
        # 盈余为 0：无可下线司机，直接返回
        if sum_idle_diff == 0:

            return
        # 固定随机种子，保证采样可复现
        np.random.seed(self.RANDOM_SEED)
        # 以盈余归一化为概率，随机抽取 n_drivers_to_off 个下线格
        node_ids = np.random.choice(self.target_node_ids, size=[n_drivers_to_off],
                                    p=idle_diff / float(sum_idle_diff))

        # 逐个处理抽中的格
        for ii, node_id in enumerate(node_ids):
            # 该格仍有空闲司机时才下线一个
            if self.nodes[node_id].idle_driver_num > 0:
                # 随机下线该格一个空闲司机
                self.nodes[node_id].set_idle_driver_offline_random()
                # 城市空闲司机数 -1
                self.n_drivers -= 1
                # 城市离线司机数 +1
                self.n_offline_drivers += 1
                # 剩余待下线数 -1（仅作记录）
                n_drivers_to_off -= 1


    # 采样生成一天的订单：对 real_orders 的每条订单按概率 p 做伯努利采样——
    # 起点和终点都在图外的订单直接丢弃；起点在图外的"进城订单"不进订单列表，
    # 而是记入 out_grid_in_orders[起始+时长, 终点格]（模拟城外司机送客进城，
    # 之后由 step_driver_status_control 按它把新司机放进终点格）；其余按起始时刻装入 day_orders
    def utility_bootstrap_oneday_order(self):

        # 真实订单总数
        num_all_orders = len(self.real_orders)
        # 每条订单以概率 p 做 0/1 伯努利采样，取值为 1 的即被选中
        index_sampled_orders = np.where(np.random.binomial(1, self.p, num_all_orders) == 1)
        # 取出被选中的订单
        one_day_orders = self.real_orders[index_sampled_orders]

        # 清零"城外进城订单"计数矩阵：144 时刻 x 有效网格数
        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))

        # 一天订单容器：外层按 144 个时刻分桶
        day_orders = [[] for _ in np.arange(self.n_intervals)]
        # 遍历每条被选中的订单
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]
            # 起始时刻（一天内的步数下标）
            start_time = int(iorder[2])
            # 起点与终点都不在地图编号中：无法落到任何格，跳过
            if iorder[0] not in self.node_mapping and iorder[1] not in self.node_mapping:
                continue
            # 起点网格编号转节点索引，图外编号映射为 -100
            start_node = self.node_mapping.get(iorder[0], -100)
            # 终点网格编号转节点索引，图外编号映射为 -100
            end_node = self.node_mapping.get(iorder[1], -100)
            # 订单时长（步）
            duration = int(iorder[3])
            # 订单价格
            price = iorder[4]


            # 起点在图外：该订单由城外司机承接——不进入订单列表，
            # 而在"起始时刻+时长"（司机送客到达时刻）给终点格记 1 个进城司机
            if start_node == -100:
                # 终点格在 target_grids 中的列位置
                column_index = self.target_grids.index(end_node)
                # 到达时刻取模一天步数，该终点格的进城司机计数 +1
                self.out_grid_in_orders[(start_time + duration) % self.n_intervals, column_index] += 1
                # 本条订单处理完毕，不装入 day_orders
                continue

            # 正常订单：按起始时刻装入对应桶（五元组：起点、终点、起始时刻、时长、价格）
            day_orders[start_time].append([start_node, end_node, start_time, duration, price])
        # 保存一天的订单列表，供回放模式使用
        self.day_orders = day_orders

    # 司机状态推进：①让送客到期的司机在订单终点复位为空闲；
    # ②按 out_grid_in_orders 把"城外送客进城"的新司机放进终点格
    def step_driver_status_control(self):
        # Deal with orders finished at time T=1, check driver status. finish order, set back to off service
        # 遍历全部注册司机，检查各自订单是否到期
        for key, _driver in list(self.drivers.items()):
            # 到期司机：位置设为订单终点、结束服务恢复空闲，并计入终点格
            _driver.status_control_eachtime(self)
        # 当前时刻在一天中的位置
        moment = self.city_time % self.n_intervals
        # 当时刻各格的"进城司机"数量
        orders_to_on_drivers = self.out_grid_in_orders[moment, :]
        # 遍历每个有效格
        for idx, item in enumerate(orders_to_on_drivers):
            # 该格有进城司机时，向该格添加等量司机
            if item != 0:
                # 该格对应的节点索引
                node_id = self.target_grids[idx]
                # 向该格添加 item 个司机（优先唤醒离线司机）
                self.utility_add_driver_real_nodewise(node_id, int(item))

    # 逐格司机上下线控制：每个有效格按 onoff 矩阵的 [均值, 标准差] 采样一个正态数
    # 并四舍五入——正数=该格上线相应数量司机，负数=下线相应数量司机，0=不变
    def step_driver_online_offline_nodewise(self):
        """ node wise control driver online offline
        :return:
        """
        # 当前时刻在一天中的位置
        moment = self.city_time % self.n_intervals
        # 当时刻各格的上下线分布参数（每行 [均值, 标准差]）
        curr_onoff_distribution = self.onoff_driver_location_mat[moment]

        # 本步全网累计上线数清零
        self.all_grids_on_number = 0
        # 本步全网累计下线数清零
        self.all_grids_off_number = 0
        # 遍历每个有效节点（按 target_node_ids 顺序）
        for idx, target_node_id in enumerate(self.target_node_ids):
            # 该格的均值（正=净上线，负=净下线）
            curr_mu    = curr_onoff_distribution[idx, 0]
            # 该格的标准差
            curr_sigma = curr_onoff_distribution[idx, 1]
            # 采样一个正态数并四舍五入取整，作为本格净上下线数量
            on_off_number = np.round(np.random.normal(curr_mu, curr_sigma, 1)[0]).astype(int)

            # 正数：该格上线相应数量的司机
            if on_off_number > 0:
                # 向该格添加司机（优先唤醒离线司机）
                self.utility_add_driver_real_nodewise(target_node_id, on_off_number)
                # 累计上线统计 +该数量
                self.all_grids_on_number += on_off_number
            # 负数：该格下线相应数量的司机
            elif on_off_number < 0:
                # 把该格 |数量| 个空闲司机下线
                self.utility_set_drivers_offline_real_nodewise(target_node_id, abs(on_off_number))
            # 0：本格司机数量不变
            else:
                pass

    # 新版上下线控制（reset_clean 使用）：把空闲司机总数调整到目标值——
    # 需要加人时走"离线优先"的添加（先唤醒离线司机），需要减人时按各格盈余加权下线
    # 参数 n_idle_drivers：期望的空闲司机数
    def step_driver_online_offline_control_new(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        # 收集当前离线司机的编号列表
        offline_drivers = self.utility_collect_offline_drivers_id()
        # 更新离线司机总数
        self.n_offline_drivers = len(offline_drivers)

        # 目标数大于当前空闲数：补足差额（优先唤醒离线司机）
        if n_idle_drivers > self.n_drivers:

            # 添加差额数量的司机
            self.utility_add_driver_real_new_offlinefirst(n_idle_drivers - self.n_drivers)

        # 目标数小于当前空闲数：下线差额数量的司机
        elif n_idle_drivers < self.n_drivers:
            # 按各格盈余加权下线差额司机
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        else:
            # 数量恰好相等，无需调整
            pass

    # 旧版上下线控制（reset 使用）：目标高于当前时先逐个唤醒离线司机、
    # 仍不足时再按缺口加权创建新司机；目标低于当前时按盈余加权下线
    # 参数 n_idle_drivers：期望的空闲司机数
    def step_driver_online_offline_control(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        # 收集当前离线司机的编号列表
        offline_drivers = self.utility_collect_offline_drivers_id()
        # 更新离线司机总数
        self.n_offline_drivers = len(offline_drivers)
        # 目标数大于当前空闲数：需要上线更多司机
        if n_idle_drivers > self.n_drivers:
            # bring drivers online.
            # 只要还没达到目标就继续补人
            while self.n_drivers < n_idle_drivers:
                # 还有离线司机可唤醒
                if self.n_offline_drivers > 0:
                    # 逐个唤醒离线司机，直到达标或离线耗尽
                    for ii in np.arange(self.n_offline_drivers):
                        # 唤醒一个离线司机
                        self.drivers[offline_drivers[ii]].set_online()
                        # 空闲司机数 +1
                        self.n_drivers += 1
                        # 离线司机数 -1
                        self.n_offline_drivers -= 1
                        # 已达目标数则提前退出唤醒循环
                        if self.n_drivers == n_idle_drivers:
                            break

                # 离线司机耗尽仍不足：按各格缺口加权创建新司机补足
                self.utility_add_driver_real_new(n_idle_drivers - self.n_drivers)

        # 目标数小于当前空闲数：按各格盈余加权下线司机
        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        else:
            # 数量相等，无需调整
            pass

    # 采样全城目标空闲司机总数（global 模式）：取当前时刻 idle_driver_dist_time 的
    # [均值, 标准差] 采样正态数并四舍五入；随机种子固定为 city_time，同时刻结果可复现
    def utility_get_n_idle_drivers_real(self):
        """ control the number of idle drivers in simulator;
        :return:
        """
        # 当前时刻在一天中的位置
        time = self.city_time % self.n_intervals
        # 取该时刻的 [均值, 标准差]
        mean, std = self.idle_driver_dist_time[time]
        # 以 city_time 作为随机种子
        np.random.seed(self.city_time)
        # 采样一个正态数并四舍五入取整返回
        return np.round(np.random.normal(mean, std, 1)[0]).astype(int)

    # 设置邻居奖励广播的分层折扣权重（默认 [1, e^-1, e^-2]）
    def utility_set_neighbor_weight(self, weights):
        # 覆盖实例的分层邻居权重
        self.weights_layers_neighbors = weights

    # 随机生成新订单（generate_order==1 的 step 路径）：为每个有效节点按其订单数量
    # 分布（泊松/高斯）采样订单数；时长按 order_time_dist 抽取、价格按对应档位
    # N(均值, 标准差) 抽取、终点从时长可达的分层邻居中随机选取
    def step_generate_order_real(self):
        # generate order at t + 1
        # 遍历全部节点
        for node in self.nodes:
            # 跳过无效格
            if node is not None:
                # 该节点的一维索引
                node_id = node.get_node_index()
                # generate orders start from each node
                # 随机种子 = 节点索引 + 当前时刻（同一节点同一时刻可复现）
                random_seed = node.get_node_index() + self.city_time
                # 生成该节点本时刻的订单（数量与属性见函数头注释）
                node.generate_order_real(self.l_max, self.order_time_dist, self.order_price_dist,
                                         self.city_time, self.nodes, random_seed)

    # 回放订单（generate_order!=1 的 step 路径）：把 day_orders 中指定时刻的订单
    # 逐条装入起点网格；终点在图外时终点节点记为 None（接单司机送客后即离线退出市场）
    # 参数 day_orders_t：某一时刻的订单列表（五元组）
    def step_bootstrap_order_real(self, day_orders_t):
        # 遍历该时刻的每条订单
        for iorder in day_orders_t:
            # 起点节点索引
            start_node_id = iorder[0]
            # 终点节点索引（可能是 -100 等图外值）
            end_node_id = iorder[1]
            # 起点节点对象
            start_node = self.nodes[start_node_id]

            # 终点是有效格：取终点节点对象
            if end_node_id in self.target_grids:
                end_node = self.nodes[end_node_id]
            else:
                # 终点在图外：记为 None（司机送客到图外后离开市场）
                end_node = None
            # 在起点格登记订单（起始时刻用当前 city_time，时长与价格取自订单）
            start_node.add_order_real(self.city_time, end_node, iorder[3], iorder[4])

    # 旧版订单匹配（不带邻居广播）：逐节点先清理到期订单，再在本格内匹配
    # min(订单数, 空闲司机数) 单；奖励 = 全部匹配订单的价格之和；
    # 同时更新全局接单率 order_response_rate（无订单时记 -1）
    def step_assign_order(self):

        # 总奖励累加器（R_{t+1}）
        reward = 0  # R_{t+1}
        # 全部节点订单总数
        all_order_num = 0
        # 已完成（成功匹配）订单总数
        finished_order_num = 0
        # 遍历全部节点
        for node in self.nodes:
            # 跳过无效格
            if node is not None:
                # 清理该格过期未接订单与已完成订单
                node.remove_unfinished_order(self.city_time)
                # 本格匹配订单，返回 [本格奖励, 订单总数, 完成数]
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time, self)
                # 累加奖励
                reward += reward_node
                # 累加订单总数
                all_order_num += all_order_num_node
                # 累加完成订单数
                finished_order_num += finished_order_num_node
        # 有订单时：接单率 = 完成数 / 总数
        if all_order_num != 0:
            self.order_response_rate = finished_order_num/float(all_order_num)
        else:
            # 无订单时接单率记 -1
            self.order_response_rate = -1
        # 返回总奖励
        return reward

    # 新版订单匹配（带邻居广播，step 使用）：第一轮逐格本格匹配 min(订单数, 空闲司机数)，
    # 第二轮对仍有剩单的格向其单向邻居"广播（向相邻网格借用空闲司机）"借司机接单；
    # 总奖励 = 全部匹配订单价格之和；返回 (总奖励, [node_reward, neighbor_reward])，
    # 其中广播借司机所得的奖励记入出借方邻居格的 neighbor_reward
    def step_assign_order_broadcast_neighbor_reward_update(self):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """

        # 各格"本格匹配"奖励数组（长度 = 网格总数 M*N，无效格恒为 0）
        node_reward = np.zeros((len(self.nodes)))
        # 各格"借邻居司机匹配"奖励数组
        neighbor_reward = np.zeros((len(self.nodes)))
        # First round broadcast
        # 总奖励累加器（R_{t+1}）
        reward = 0  # R_{t+1}
        # 全部订单数
        all_order_num = 0
        # 已完成订单数
        finished_order_num = 0
        # 第一轮：逐格本格匹配
        for node in self.nodes:
            # 跳过无效格
            if node is not None:

                # 本格匹配：返回 [奖励, 订单总数, 完成数]（本格空闲司机接本格订单）
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time, self)
                # 累加总奖励
                reward += reward_node
                # 累加订单总数
                all_order_num += all_order_num_node
                # 累加完成订单数
                finished_order_num += finished_order_num_node
                # 本格奖励记入 node_reward
                node_reward[node.get_node_index()] += reward_node
        # Second round broadcast
        # 第二轮：对仍有剩单的格广播借邻居司机
        for node in self.nodes:
            # 跳过无效格
            if node is not None:
                # 只处理还有未接订单的格（此时本格已无空闲司机）
                if node.order_num != 0:
                    # 广播匹配：向邻居借司机接单，返回 [新增奖励, 新增完成数]，
                    # 出借司机的邻居格所得奖励直接累加进 neighbor_reward
                    reward_node_broadcast, finished_order_num_node_broadcast \
                        = node.simple_order_assign_broadcast_update(self, neighbor_reward)
                    # 累加广播新增奖励
                    reward += reward_node_broadcast
                    # 累加广播新增完成数
                    finished_order_num += finished_order_num_node_broadcast

        # 逐格总奖励 = 本格匹配奖励 + 该格出借司机所得的广播奖励
        node_reward = node_reward + neighbor_reward
        # 有订单时：接单率 = 完成数 / 总数
        if all_order_num != 0:
            self.order_response_rate = finished_order_num/float(all_order_num)
        else:
            # 无订单时记 -1
            self.order_response_rate = -1

        # 返回总奖励与 [本格奖励数组, 邻居广播奖励数组]
        return reward, [node_reward, neighbor_reward]

    # 清理全部有效格的到期订单：移除超过等待时间仍未被接的订单和已完成订单
    def step_remove_unfinished_orders(self):
        # 遍历全部节点
        for node in self.nodes:
            # 跳过无效格
            if node is not None:
                # 清理该格过期/已完成订单
                node.remove_unfinished_order(self.city_time)

    # 预匹配供需盈余统计（构成 step 返回 info 中的 context）：基于下一状态计算每格
    # "净剩司机"与"净剩订单"；若某格有剩单，则模拟向其邻居借司机互相抵扣，
    # 返回对齐后的 [剩司机分布, 剩订单分布]（2xMxN），供算法感知各格供需缺口
    # 参数 next_state：get_observation 返回的 2xMxN 状态
    def step_pre_order_assigin(self, next_state):

        # 各格净剩空闲司机 = max(空闲司机数 - 订单数, 0)（供过于求的格）
        remain_drivers = next_state[0] - next_state[1]
        remain_drivers[remain_drivers < 0] = 0

        # 各格净剩订单 = max(订单数 - 空闲司机数, 0)（供不应求的格）
        remain_orders = next_state[1] - next_state[0]
        remain_orders[remain_orders < 0] = 0

        # 全城无剩单或无剩司机：无需抵扣，直接拼成 2xMxN 返回
        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:
            # 拼接 [剩司机分布, 剩订单分布]
            context = np.array([remain_drivers, remain_orders])
            return context

        # 展平成一维数组，便于按节点索引访问
        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        # 遍历全部节点做邻居抵扣
        for node in self.nodes:
            # 跳过无效格
            if node is not None:
                # 当前节点索引
                curr_node_id = node.get_node_index()
                # 该格还有剩单才需要向邻居借司机
                if remain_orders_1d[curr_node_id] != 0:
                    # 逐个邻居尝试借司机抵扣剩单
                    for neighbor_node in node.neighbors:
                        # 邻居存在才可用
                        if neighbor_node is not None:
                            # 邻居节点索引
                            neighbor_id = neighbor_node.get_node_index()
                            # 当前格剩单数 a
                            a = remain_orders_1d[curr_node_id]
                            # 邻居格剩司机数 b
                            b = remain_drivers_1d[neighbor_id]
                            # 两者互相抵扣：剩单取 max(a-b, 0)
                            remain_orders_1d[curr_node_id] = max(a-b, 0)
                            # 剩司机取 max(b-a, 0)
                            remain_drivers_1d[neighbor_id] = max(b-a, 0)
                        # 本格剩单已清零则提前结束邻居循环
                        if remain_orders_1d[curr_node_id] == 0:
                            break

        # 抵扣后的剩司机、剩订单各自还原为 MxN，拼成 2xMxN 上下文
        context = np.array([remain_drivers_1d.reshape(self.M, self.N),
                   remain_orders_1d.reshape(self.M, self.N)])
        # 返回供需盈余上下文
        return context

    # 执行调度动作：处理形如 (起点格, 终点格, 人数) 的调度列表——被调司机先从起点格
    # 移除并暂时离线（本步不能接单），下一时间步由 step_add_dispatched_drivers 在终点格
    # 重新上线；终点无效（编号 <0 或节点为 None）时被调司机直接下线离开市场；
    # 终点不是起点邻居（不相邻）则抛异常
    # 参数 dispatch_actions：调度动作列表，每项 (start_node_id, end_node_id, num_of_drivers)
    # 返回 save_remove_id：[(终点节点索引, 司机编号), ...]，供到达时恢复
    def step_dispatch_invalid(self, dispatch_actions):
        """ If a
        :param dispatch_actions:
        :return:
        """
        # 收集被调走司机的 (终点, 司机编号) 对
        save_remove_id = []
        # 遍历每个调度动作
        for action in dispatch_actions:

            # 解包动作三元组：起点格、终点格、调度人数
            start_node_id, end_node_id, num_of_drivers = action
            # 起点无效或调度人数为 0：跳过该动作
            if self.nodes[start_node_id] is None or num_of_drivers == 0:
                continue  # not a feasible action

            # 起点空闲司机不足时，把人数截断为实际可用数
            if self.nodes[start_node_id].get_driver_numbers() < num_of_drivers:
                num_of_drivers = self.nodes[start_node_id].get_driver_numbers()

            # 目标格编号 < 0：视为调度出城，被调司机直接下线
            if end_node_id < 0:
                # 逐个把起点的空闲司机下线
                for _ in np.arange(num_of_drivers):
                    # 随机选一个空闲司机下线
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    # 城市空闲司机数 -1
                    self.n_drivers -= 1
                    # 城市离线司机数 +1
                    self.n_offline_drivers += 1
                    # 全网下线统计 +1
                    self.all_grids_off_number += 1
                # 处理下一个动作
                continue

            # 目标格是无效格（None）：同样视为调度出城，司机下线
            if self.nodes[end_node_id] is None:
                # 逐个把起点的空闲司机下线
                for _ in np.arange(num_of_drivers):
                    # 随机选一个空闲司机下线
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    # 城市空闲司机数 -1
                    self.n_drivers -= 1
                    # 城市离线司机数 +1
                    self.n_offline_drivers += 1
                    # 全网下线统计 +1
                    self.all_grids_off_number += 1
                # 处理下一个动作
                continue

            # 目标格不是起点格的邻居（不相邻）：非法调度，抛出异常
            if self.nodes[end_node_id] not in self.nodes[start_node_id].neighbors:
                raise ValueError('City:step(): not a feasible dispatch')


            # 正常调度：把 num_of_drivers 个空闲司机从起点格移出、暂时离线在途
            for _ in np.arange(num_of_drivers):
                # t = 1 dispatch start, idle driver decrease
                # 从起点格随机移除一个空闲司机（节点空闲数 -1）
                remove_driver_id = self.nodes[start_node_id].remove_idle_driver_random()
                # 记录 (目的地, 司机编号)，供下一时间步恢复
                save_remove_id.append((end_node_id, remove_driver_id))
                # 司机位置暂时置空（在途状态）
                self.drivers[remove_driver_id].set_position(None)
                # 司机进入"调度在途"的离线状态
                self.drivers[remove_driver_id].set_offline_for_start_dispatch()
                # 城市空闲司机数 -1
                self.n_drivers -= 1

        # 返回被调走司机的 (目的地, 编号) 列表
        return save_remove_id

    # 调度司机到达：把上一时间步被调走的司机放到目的地格并恢复在线空闲，
    # 完成一次空车重调度（t 步发出、t+1 步到达可用）
    # 参数 save_remove_id：step_dispatch_invalid 返回的 [(目的地, 司机编号), ...]
    def step_add_dispatched_drivers(self, save_remove_id):
        # drivers dispatched at t, arrived at t + 1
        # 遍历每个被调司机的 (目的地, 编号)
        for destination_node_id, arrive_driver_id in save_remove_id:
            # 司机位置设为目的节点
            self.drivers[arrive_driver_id].set_position(self.nodes[destination_node_id])
            # 恢复在线状态（结束调度在途）
            self.drivers[arrive_driver_id].set_online_for_finish_dispatch()
            # 把司机加入目的地节点的司机表（节点空闲数 +1）
            self.nodes[destination_node_id].add_driver(arrive_driver_id, self.drivers[arrive_driver_id])
            # 城市空闲司机数 +1
            self.n_drivers += 1

    # 城市时间 +1，并把新时间同步写入全部注册司机（供其状态判断使用）
    def step_increase_city_time(self):
        # 城市时间步 +1
        self.city_time += 1
        # set city time of drivers
        # 遍历全部注册司机
        for driver_id, driver in list(self.drivers.items()):
            # 更新该司机记录的城市时间
            driver.set_city_time(self.city_time)

    # 推进一个时间步（10 分钟）的完整仿真流程——强化学习"环境"的核心接口，一轮八步：
    # ①step_dispatch_invalid 执行调度动作 (起点格,终点格,人数)，被调司机先离线、下一步才到达；
    # ②step_assign_order_broadcast_neighbor_reward_update 两轮订单匹配（本格匹配 + 邻居广播
    # 借司机），奖励 = 匹配订单价格之和，并更新接单率；③时间 +1；④step_driver_status_control
    # 送客到期司机在终点复位空闲、处理进城司机；⑤step_add_dispatched_drivers 调度司机到达；
    # ⑥生成新订单（generate_order==1 随机生成，否则回放 day_orders）；⑦step_driver_online_offline_nodewise
    # 按矩阵逐格采样上下线；⑧清理过期订单、get_observation 生成观测、step_pre_order_assigin
    # 算供需盈余；返回 (下一状态, 总奖励, info)
    # 参数 dispatch_actions：调度动作列表；generate_order：1=随机生成订单，其他=回放
    def step(self, dispatch_actions, generate_order=1):
        # info 信息列表（稍后填入逐格奖励与供需盈余上下文）
        info = []
        # —— 第①②步在时间 T 进行 ——
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 第①步：执行调度动作，被调司机暂时离线在途，返回 (目的地, 司机编号) 列表
        save_remove_id = self.step_dispatch_invalid(dispatch_actions)
        # When the drivers go to invalid grid, set them offline.

        # 第②步：两轮订单匹配（本格匹配 + 邻居广播借司机），得总奖励与逐格奖励
        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        # —— 以下第③~⑧步把时间推进到 T+1 ——
        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 第③步：城市时间 +1 并同步司机时间
        self.step_increase_city_time()
        # 第④步：送客到期司机在终点复位空闲，并按 out_grid_in_orders 放入进城司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 第⑤步：上一时间步调度的司机到达目的地格、恢复空闲
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # 第⑥步：生成 t+1 时刻的新订单
        if generate_order == 1:
            # 随机模式：按各节点订单数量分布随机生成
            self.step_generate_order_real()
        else:
            # 回放模式：取 day_orders 当时刻的订单装入各起点格
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])
        
        # offline online control;
        # 第⑦步：逐格采样上下线净数量，正=上线、负=下线
        self.step_driver_online_offline_nodewise()

        # 第⑧步：清理各格过期未接订单
        self.step_remove_unfinished_orders()
        # get states S_{t+1}  [driver_dist, order_dist]
        # 生成下一状态 S_{t+1}：[空闲司机分布, 订单分布]（2xMxN）
        next_state = self.get_observation()

        # 基于下一状态计算供需盈余上下文（模拟邻居借车抵扣后的剩司机/剩订单）
        context = self.step_pre_order_assigin(next_state)
        # info = [逐格奖励 [node_reward, neighbor_reward], 供需盈余上下文]
        info = [reward_node, context]
        # 返回 (下一状态, 总奖励, info)
        return next_state, reward, info
