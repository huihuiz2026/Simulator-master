# 导入 NumPy 数值计算库并简写为 np：用于随机采样（泊松/正态分布）和数组操作
import numpy as np
# 从 abc 模块导入抽象基类工具：ABCMeta（抽象类元类）、abstractmethod（抽象方法装饰器），
# 用于定义"分布"抽象基类，强制子类实现 sample 采样方法
from abc import ABCMeta, abstractmethod
# 从同包的 utilities.py 导入全部工具函数：ids_1dto2d / ids_2dto1d（一维编号与二维坐标互换）、
# get_layers_neighbors（计算分层邻居）等，Node 的邻居计算会用到
from .utilities import *

# 分布抽象基类：规定"每个网格每个时间步生成多少条订单"的随机采样接口
class Distribution(metaclass=ABCMeta):
    ''' Define the distribution from which sample the orders'''
    # 抽象采样方法：所有子类必须实现 sample，否则无法实例化
    @abstractmethod
    # 按分布抽取一个样本（订单数量）；参数与返回值由子类自行定义
    def sample(self):
        # 抽象方法体：仅占位，具体逻辑由子类实现
        pass

# 泊松分布：常用于建模随机到达的事件数量（如某网格每 10 分钟产生的订单数），是本仿真的默认选择
class PoissonDistribution(Distribution):

    # lam：泊松分布的均值 lambda，即单位时间内事件发生的平均次数
    def __init__(self, lam):
        # 保存 lambda 参数，供 sample 采样时使用
        self._lambda = lam

    # seed：随机种子；相同种子产生相同采样序列，保证仿真可复现
    def sample(self, seed=0):
        # 固定全局随机种子（环境用"网格编号+时刻"作种子，各格采样互不相同但可复现）
        np.random.seed(seed)
        # 从参数为 lambda 的泊松分布抽 1 个样本，作为本步生成的订单数
        return np.random.poisson(self._lambda, 1)[0]


# 高斯（正态）分布：也可用来建模每步订单数量，本仿真默认用泊松，此类作为备选
class GaussianDistribution(Distribution):

    # args：形如 (mu, sigma) 的元组，即分布的均值与标准差
    def __init__(self, args):
        # 解包元组，得到均值 mu 与标准差 sigma
        mu, sigma = args
        # 均值
        self.mu = mu        # mean
        # 标准差
        self.sigma = sigma  # standard deviation

    # seed：随机种子；相同种子产生相同采样序列
    def sample(self, seed=0):
        # 固定全局随机种子，保证可复现
        np.random.seed(seed)
        # 从正态分布 N(mu, sigma) 抽 1 个样本，作为本步生成的订单数
        return np.random.normal(self.mu, self.sigma, 1)[0]


# Node（网格/节点）类：城市地图上的一个网格，是仿真中最核心的实体容器；
# 持有本格的订单列表与司机字典，并实现"本格内订单-司机匹配""向邻居借司机"等核心调度逻辑
class Node(object):
    # __slots__：限定实例只能拥有下列属性（禁止动态添加其他属性），可节省内存、加快属性访问
    __slots__ = ('neighbors', '_index', 'orders', 'drivers',  # 邻居列表、本格编号、订单列表、司机字典
                 'order_num', 'idle_driver_num', 'offline_driver_num',  # 订单数、空闲司机数、离线司机数
                 'order_generator',  # 本格的订单数量随机生成器
                 'n_side', 'layers_neighbors', 'layers_neighbors_id')  # 邻域边数、分层邻居（坐标/编号两种形式）

    # index：本网格的唯一编号（0 ~ M*N-1，按"行优先"规则对应二维地图坐标）
    def __init__(self, index):
        # private
        # 本格的唯一编号（私有属性，通过 get_node_index() 读取）
        self._index = index   # unique node index.

        # public
        # 邻居网格对象列表（按 6 个方向排列，地图边界外的位置为 None）
        self.neighbors = []  # a list of nodes that neighboring the Nodes
        # 本格当前待匹配的订单列表
        self.orders = []     # a list of orders
        # 本格的司机字典 {司机ID: 司机对象}（存放驻留在本格的司机，含离线司机）
        self.drivers = {}    # a dictionary of driver objects contained in this node
        # 本格订单计数（与 len(self.orders) 保持一致）
        self.order_num = 0
        # 本格空闲司机数（在线且不在送客）
        self.idle_driver_num = 0  # number of idle drivers in this node
        # 本格离线司机数（暂不接单但仍记录在本格）
        self.offline_driver_num = 0
        # 订单数量生成器（PoissonDistribution 或 GaussianDistribution 实例）
        self.order_generator = None

        # 邻域边数：6 表示六边形邻域，4 表示四边形邻域
        self.n_side = 0      # the topology is a n-sided map
        # 分层邻居坐标列表：layers_neighbors[0] 是第 1 层（紧邻一圈）邻居的 (行,列) 坐标，
        # layers_neighbors[1] 是第 2 层，依此类推
        self.layers_neighbors = []  # layer 1 indices: layers_neighbors[0] = [[1,1], [0, 1], ...],
        # layer 2 indices layers_neighbors[1]
        # 分层邻居编号列表：layers_neighbors_id[0] 是第 1 层有效邻居的一维网格编号
        self.layers_neighbors_id = [] # layer 1: layers_neighbors_id[0] = [2, 1,.]

    # 清空本格的全部订单与司机数据（环境 reset 时逐格调用，用于重新开始一天的仿真）
    def clean_node(self):
        # 清空订单列表
        self.orders = []
        # 订单计数归零
        self.order_num = 0
        # 清空司机字典
        self.drivers = {}
        # 空闲司机计数归零
        self.idle_driver_num = 0
        # 离线司机计数归零
        self.offline_driver_num = 0

    # 计算本格的"分层邻居"：第 1 层是紧邻的一圈网格，第 2 层是再外一圈……最多到第 l_max 层；
    # 结果同时保存为二维坐标形式（layers_neighbors）和一维编号形式（layers_neighbors_id）
    # 参数 l_max：最大层数（也是订单最大时长/最远可达层数）；M、N：地图行数、列数；
    # env：城市环境对象（用它的 nodes 列表判断某编号网格是否有效）
    def get_layers_neighbors(self, l_max, M, N, env):

        # 把本格的一维编号转换成二维坐标 (行 x, 列 y)
        x, y = ids_1dto2d(self.get_node_index(), M, N)
        # 调用工具函数：以本格为中心逐层向外扩展，得到每层邻居的 (x, y) 坐标列表
        self.layers_neighbors = get_layers_neighbors(x, y, l_max, M, N)
        # 遍历每一层邻居：把坐标换算成一维编号，并剔除无效网格（None）
        for layer_neighbors in self.layers_neighbors:
            # 收集本层的有效邻居编号
            temp = []
            # 遍历本层的每个邻居坐标
            for item in layer_neighbors:
                # 解包出 (行, 列) 坐标
                x, y = item
                # 坐标换算成一维网格编号
                node_id = ids_2dto1d(x, y, M, N)
                # 该编号在地图上是有效网格（非 None）
                if env.nodes[node_id] is not None:
                    # 有效才加入本层邻居编号列表
                    temp.append(node_id)
            # 本层结果存入分层邻居编号列表
            self.layers_neighbors_id.append(temp)

    # 返回本格的唯一编号
    def get_node_index(self):
        # 返回创建时指定的编号（0 ~ M*N-1）
        return self._index

    # 返回本格空闲司机数（直接读计数器，不做遍历）
    def get_driver_numbers(self):
        # 返回空闲司机计数字段
        return self.idle_driver_num

    # 逐个遍历本格司机字典统计空闲司机数（在线且不在送客）；
    # 与计数器 idle_driver_num 互为校验的"逐个点数"版本
    def get_idle_driver_numbers_loop(self):
        # 空闲司机计数器
        temp_idle_driver = 0
        # 遍历本格司机字典（list() 先拷贝一份，防止遍历过程中字典被修改）
        for key, driver in list(self.drivers.items()):
            # 不在送客且在线 = 空闲可接单
            if driver.onservice is False and driver.online is True:
                # 空闲数 +1
                temp_idle_driver += 1
        # 返回统计到的空闲司机数
        return temp_idle_driver

    # 逐个遍历本格司机字典统计离线司机数（不在送客且已离线）
    def get_off_driver_numbers_loop(self):
        # 离线司机计数器（变量名沿用 idle，实际统计的是离线司机）
        temp_idle_driver = 0
        # 遍历本格司机字典的拷贝
        for key, driver in list(self.drivers.items()):
            # 不在送客且离线 = 已下线司机
            if driver.onservice is False and driver.online is False:
                # 离线数 +1
                temp_idle_driver += 1
        # 返回统计到的离线司机数
        return temp_idle_driver

    # 按分布名称为本格创建订单数量生成器
    # 参数 distribution：分布名称字符串（'Poisson' 或 'Gaussian'）；
    # dis_paras：分布参数（泊松为 lambda，高斯为 (mu, sigma) 元组）
    def order_distribution(self, distribution, dis_paras):

        # 名称是泊松分布
        if distribution == 'Poisson':
            # 创建泊松生成器并保存到本格
            self.order_generator = PoissonDistribution(dis_paras)
        # 名称是高斯分布
        elif distribution == 'Gaussian':
            # 创建高斯生成器并保存到本格
            self.order_generator = GaussianDistribution(dis_paras)
        # 其他名称暂不支持
        else:
            # 什么都不做（order_generator 保持为 None）
            pass

    # 简易随机模式：每个时间步为本格随机生成若干新订单（价格、时长、终点均为简单分布）
    # 参数 city_time：当前时刻；nodes：全城网格列表（按编号索引）；seed：随机种子
    def generate_order_random(self, city_time, nodes, seed):
        """Generate new orders at each time step
        """
        # 按分布采样本步要生成的订单数量
        num_order_t = self.order_generator.sample(seed)
        # 本格订单计数累加
        self.order_num += num_order_t

        # 逐条构造新订单
        for ii in np.arange(num_order_t):
            # 订单价格从正态分布 N(50, 5) 中采样
            price = np.random.normal(50, 5, 1)[0]
            # 价格若采到负数，兜底设为 10
            price = 10 if price < 0 else price

            # 起点：本格编号
            current_node_id = self.get_node_index()
            # 候选终点：全城除本格之外的所有网格编号（不区分是否有效格）
            destination_node = [kk for kk in np.arange(len(nodes)) if kk != current_node_id]
            # 构造订单并加入本格订单列表（对应五元组：起点网格, 终点网格, 起始时刻, 时长, 价格）
            self.orders.append(Order(nodes[current_node_id],  # 起点网格：本格
                                     nodes[np.random.choice(destination_node, 1)[0]],  # 终点网格：从候选中随机抽 1 个
                                     city_time,  # 起始时刻
                                     # city_time + np.random.choice(5, 1)[0]+1,
                                     # 订单时长：从 {0,1} 抽 1 个再加 1，即 1 或 2 个时间步
                                     np.random.choice(2, 1)[0]+1,  # duration
                                     price, 1))  # 订单价格；最后的 1 是等待时长（可等 1 步，超时作废）
        # 无返回值（订单已存入 self.orders）
        return

    # 真实数据模式：每个时间步为本格生成新订单——数量按泊松采样，时长按给定概率抽取，
    # 价格按时长档位的正态分布采样，终点限制在本格的分层邻居（可达范围）内
    # 参数 l_max：最大层数（也是订单最大时长）；order_time_dist：各时长的出现概率；
    # order_price_dist：各时长档的 [价格均值, 价格标准差]；city_time：当前时刻；
    # nodes：全城网格列表；seed：随机种子
    def generate_order_real(self, l_max, order_time_dist, order_price_dist, city_time, nodes, seed):
        """Generate new orders at each time step
        """
        # 采样本步生成的订单数量
        num_order_t = self.order_generator.sample(seed)
        # 本格订单计数累加
        self.order_num += num_order_t
        # 逐条构造新订单
        for ii in np.arange(num_order_t):

            # 最大层数为 1 时
            if l_max == 1:
                # 时长固定为 1 步
                duration = 1
            # 否则
            else:

                # 按概率 order_time_dist 从 1~l_max 步中抽取订单时长
                duration = np.random.choice(np.arange(1, l_max+1), p=order_time_dist)
            # 取该时长档位对应的价格均值与标准差（列表下标从 0 开始，故减 1）
            price_mean, price_std = order_price_dist[duration-1]
            # 从正态分布 N(均值, 标准差) 采样价格
            price = np.random.normal(price_mean, price_std, 1)[0]
            # 价格若非正数，兜底改用该档位的均值
            price = price if price > 0 else price_mean

            # 起点：本格编号
            current_node_id = self.get_node_index()
            # 候选终点编号列表
            destination_node = []
            # 时长为 duration 步的订单，终点可能落在第 1~duration 层邻居之内
            for jj in np.arange(duration):
                # 遍历第 jj 层（下标 jj 对应第 jj+1 层）邻居编号
                for kk in self.layers_neighbors_id[jj]:
                    # 该编号网格在地图上有效
                    if nodes[kk] is not None:
                        # 加入候选终点
                        destination_node.append(kk)
            # 构造订单并加入本格订单列表
            self.orders.append(Order(nodes[current_node_id],  # 起点网格：本格
                                     nodes[np.random.choice(destination_node, 1)[0]],  # 终点网格：候选中随机抽 1 个
                                     city_time,  # 起始时刻
                                     duration,  # 订单时长（时间步数）
                                     price, 1))  # 订单价格；最后的 1 是等待时长（可等 1 步）
        # 无返回值
        return

    # 添加一条预生成的"真实订单"（来自历史数据五元组 [起点, 终点, 时刻, 时长, 价格]）；
    # 终点可能不在目标区域内，此时传入的 destination_node 为 None
    # 参数 city_time：当前时刻；destination_node：终点网格对象（区域外为 None）；
    # duration：时长；price：价格
    def add_order_real(self, city_time, destination_node, duration, price):
        # 起点：本格编号
        current_node_id = self.get_node_index()
        # 构造订单并加入本格订单列表
        self.orders.append(Order(self,  # 起点网格：本格
                                 destination_node,  # 终点网格：目标区域外时为 None
                                 city_time,  # 起始时刻
                                 duration,  # 订单时长（时间步数）
                                 price, 0))  # 订单价格；最后的 0 是等待时长（只在当前这一步可被接）
        # 本格订单计数 +1
        self.order_num += 1

    # 设置本格的邻居网格列表
    # 参数 nodes_list：邻居 Node 对象列表（按方向排列，地图边界外的位置为 None）
    def set_neighbors(self, nodes_list):
        # 保存邻居列表
        self.neighbors = nodes_list
        # 记录邻域边数（6=六边形邻域，4=四边形邻域）
        self.n_side = len(nodes_list)

    # 从本格移除一个空闲司机（调度算法把司机派去别的网格时调用；
    # 名字带 random，实际是按字典顺序取第一个空闲司机）
    def remove_idle_driver_random(self):
        """Randomly remove one idle driver from current grid"""
        # 被移除司机的 ID，"NA" 表示尚未找到
        removed_driver_id = "NA"
        # 遍历本格司机字典的拷贝
        for key, item in list(self.drivers.items()):
            # 该司机空闲（不在送客且在线）
            if item.onservice is False and item.online is True:
                # 从本格移除该司机
                self.remove_driver(key)
                # 记录其 ID
                removed_driver_id = key
            # 已经移除到一个司机
            if removed_driver_id != "NA":
                # 结束遍历
                break
        # 断言：本格必须存在空闲司机可移除，否则报错
        assert removed_driver_id != "NA"
        # 返回被移除司机的 ID
        return removed_driver_id

    # 将本格一个空闲司机置为离线（人仍留在本格字典中，只是不接单；
    # 名字带 random，实际按字典顺序取第一个空闲司机）
    def set_idle_driver_offline_random(self):
        """Randomly set one idle driver offline"""
        # 被下线司机的 ID，"NA" 表示尚未找到
        removed_driver_id = "NA"
        # 遍历本格司机字典的拷贝
        for key, item in list(self.drivers.items()):
            # 该司机空闲（不在送客且在线）
            if item.onservice is False and item.online is True:
                # 将其置为离线（本格空闲数 -1、离线数 +1）
                item.set_offline()
                # 记录其 ID
                removed_driver_id = key
            # 已找到一个司机
            if removed_driver_id != "NA":
                # 结束遍历
                break
        # 断言：本格必须存在可下线的空闲司机
        assert removed_driver_id != "NA"
        # 返回被下线司机的 ID
        return removed_driver_id

    # 将本格一个离线司机重新置为在线（恢复接单），返回其 ID
    def set_offline_driver_online(self):

        # 被上线司机的 ID，"NA" 表示尚未找到
        online_driver_id = "NA"
        # 遍历本格司机字典的拷贝
        for key, item in list(self.drivers.items()):
            # 该司机不在送客且已离线
            if item.onservice is False and item.online is False:
                # 将其重新置为在线（本格空闲数 +1、离线数 -1）
                item.set_online()
                # 记录其 ID
                online_driver_id = key
            # 已找到一个司机
            if online_driver_id != "NA":
                # 结束遍历
                break
        # 断言：本格必须存在可上线的离线司机
        assert online_driver_id != "NA"
        # 返回被上线司机的 ID
        return online_driver_id

    # 取出本格的一个司机对象（名字带 random，实际返回字典中第一个司机；
    # 调用前断言本格至少有一个空闲司机，但返回的未必是空闲的那个）
    def get_driver_random(self):
        """Randomly get one driver"""
        # 断言：本格空闲司机数必须大于 0
        assert self.idle_driver_num > 0
        # 用于记录取到的司机 ID，先初始化为 0
        get_driver_id = 0
        # 遍历本格司机 ID 列表
        for key in list(self.drivers.keys()):
            # 记下第一个 ID
            get_driver_id = key
            # 立即退出循环（只取第一个）
            break
        # 返回该 ID 对应的司机对象
        return self.drivers[get_driver_id]

    # 按 ID 从本格移除司机并返回该司机对象；若该司机不在本格则抛出异常
    # 参数 driver_id：要移除的司机 ID
    def remove_driver(self, driver_id):

        # 从字典弹出该司机（不存在时得到 None）
        removed_driver = self.drivers.pop(driver_id, None)
        # 本格空闲司机计数 -1
        self.idle_driver_num -= 1
        # 弹出结果为 None，说明该司机不在本格
        if removed_driver is None:
            # 抛出异常：不能移除不属于本格的司机
            raise ValueError('Nodes.remove_driver: Remove a driver that is not in this node')

        # 返回被移除的司机对象
        return removed_driver

    # 把司机登记进本格（司机完成订单出现在终点网格、或被调度到达新网格时调用）
    # 参数 driver_id：司机 ID；driver：司机对象
    def add_driver(self, driver_id, driver):
        # 以 ID 为键存入本格司机字典
        self.drivers[driver_id] = driver
        # 本格空闲司机计数 +1
        self.idle_driver_num += 1

    # 清理本格中"已作废"和"已完成"的订单：
    # 超过等待时长仍未被接单的订单作废移除；已被接单且到送达时刻的订单完成移除
    # 参数 city_time：当前时刻
    def remove_unfinished_order(self, city_time):
        # 记录需要移除的订单下标
        un_finished_order_index = []
        # 带下标遍历本格订单列表
        for idx, o in enumerate(self.orders):
            # order un served
            # 订单未被接：起始时刻 + 等待时长 已过，超过容忍等待时间，订单作废
            if o.get_wait_time()+o.get_begin_time() < city_time:
                # 记录该作废订单的下标
                un_finished_order_index.append(idx)

            # order completed
            # 订单已完成：接单时刻 + 时长 == 当前时刻（司机恰好送达），
            # 且 assigned_time != -1 表示确实被接过单
            if o.get_assigned_time() + o.get_duration() == city_time and o.get_assigned_time() != -1:
                # 记录该已完成订单的下标
                un_finished_order_index.append(idx)

        # 存在需要移除的订单
        if len(un_finished_order_index) != 0:
            # remove unfinished orders
            # 用列表推导式保留不在移除名单中的订单
            self.orders = [i for j, i in enumerate(self.orders) if j not in un_finished_order_index]
            # 订单计数同步为列表长度
            self.order_num = len(self.orders)

    # 本格内的简易订单匹配：可成单数 = min(本格订单数, 本格空闲司机数)，
    # 每成一单把订单价格累加进 reward，接单司机转为送客状态并离开本格司机池
    # 参数 city_time：当前时刻（记为接单时刻）；city：城市环境（更新其全城空闲司机计数 n_drivers）
    # 返回：(本格奖励, 本格总订单数, 成功匹配订单数)
    def simple_order_assign(self, city_time, city):
        # 本格奖励累计器（= 成功匹配订单的价格之和）
        reward = 0
        # 本轮可匹配单数：订单数与空闲司机数的较小者
        num_assigned_order = min(self.order_num, self.idle_driver_num)
        # 记录已服务订单在列表中的下标
        served_order_index = []
        # 逐单匹配（订单按列表顺序优先）
        for idx in np.arange(num_assigned_order):
            # 取第 idx 条订单
            order_to_serve = self.orders[idx]
            # 记录接单时刻（用于推算订单完成时刻）
            order_to_serve.set_assigned_time(city_time)
            # 本格订单计数 -1
            self.order_num -= 1
            # 奖励累加该单价格
            reward += order_to_serve.get_price()
            # 记录已服务下标
            served_order_index.append(idx)
            # 遍历本格司机，找一个空闲司机接这单
            for key, assigned_driver in list(self.drivers.items()):
                # 该司机空闲（不在送客且在线）
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    # 司机接单：转为送客状态
                    assigned_driver.take_order(order_to_serve)
                    # 接单司机从本格司机池移除（送客期间不属于任何网格）
                    removed_driver = self.drivers.pop(assigned_driver.get_driver_id(), None)
                    # 断言：移除成功
                    assert removed_driver is not None
                    # 全城空闲司机计数 -1
                    city.n_drivers -= 1
                    # 已找到司机，结束内层循环去匹配下一单
                    break

        # 本格本轮总订单数（匹配前的订单列表长度）
        all_order_num = len(self.orders)
        # 成功匹配的订单数
        finished_order_num = len(served_order_index)

        # remove served orders
        # 从订单列表中删除已服务的订单
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        # 一致性校验：订单计数必须等于列表长度
        assert self.order_num == len(self.orders)

        # 返回 (奖励, 总单数, 成单数)
        return reward, all_order_num, finished_order_num

    # 真实模式的本格订单匹配：规则与 simple_order_assign 相同（成单数 = min(订单数, 空闲司机数)），
    # 区别在于处理"终点在目标区域外"的订单——接这种单的司机被直接置为离线
    # （语义：司机送客出了目标区域后就离开市场，不再回来）
    # 参数 city_time：当前时刻；city：城市环境
    # 返回：(本格奖励, 本格总订单数, 成功匹配订单数)
    def simple_order_assign_real(self, city_time, city):

        # 本格奖励累计器（= 成功匹配订单的价格之和）
        reward = 0
        # 本轮可匹配单数：订单数与空闲司机数的较小者
        num_assigned_order = min(self.order_num, self.idle_driver_num)
        # 记录已服务订单下标
        served_order_index = []
        # 逐单匹配（订单按列表顺序优先）
        for idx in np.arange(num_assigned_order):
            # 取第 idx 条订单
            order_to_serve = self.orders[idx]
            # 记录接单时刻（用于推算订单完成时刻）
            order_to_serve.set_assigned_time(city_time)
            # 本格订单计数 -1
            self.order_num -= 1
            # 奖励累加该单价格
            reward += order_to_serve.get_price()
            # 记录已服务下标
            served_order_index.append(idx)
            # 遍历本格司机，找一个空闲司机接这单
            for key, assigned_driver in list(self.drivers.items()):
                # 该司机空闲（不在送客且在线）
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    # 订单终点在目标区域内
                    if order_to_serve.get_end_position() is not None:
                        # 司机接单：转为送客状态
                        assigned_driver.take_order(order_to_serve)
                        # 接单司机从本格司机池移除（送客期间不属于任何网格）
                        removed_driver = self.drivers.pop(assigned_driver.get_driver_id(), None)
                        # 断言：移除成功
                        assert removed_driver is not None
                    # 订单终点在目标区域外
                    else:
                        # 司机置为离线（送出城后离开市场；仍留在本格字典但不再接单）
                        assigned_driver.set_offline()  # order destination is not in target region
                    # 全城空闲司机计数 -1（接单的转为送客、出城的转为离线）
                    city.n_drivers -= 1
                    # 已找到司机，结束内层循环去匹配下一单
                    break

        # 本格本轮总订单数（匹配前的订单列表长度）
        all_order_num = len(self.orders)
        # 成功匹配的订单数
        finished_order_num = len(served_order_index)

        # remove served orders
        # 从订单列表中删除已服务的订单
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        # 一致性校验：订单计数必须等于列表长度
        assert self.order_num == len(self.orders)

        # 返回 (奖励, 总单数, 成单数)
        return reward, all_order_num, finished_order_num


    # 第二轮"广播"匹配：本格空闲司机已用完但仍有剩单时，向各方向的邻居网格借空闲司机接单；
    # 借出司机的邻居同时把这部分奖励记到自己名下（neighbor_node_reward），
    # 供多智能体训练时给"出借司机的网格"分配回报
    # 参数 city：城市环境；neighbor_node_reward：按网格编号索引的奖励数组（借出方收益记账）
    # 返回：(本格借司机获得的奖励, 借司机完成的订单数)
    def simple_order_assign_broadcast_update(self, city, neighbor_node_reward):

        # 断言：进入此方法时本格已没有空闲司机（第一轮匹配已用完）
        assert self.idle_driver_num == 0
        # 本格通过邻居司机获得的奖励累计
        reward = 0
        # 借司机完成的总单数
        num_finished_orders = 0
        # 依次访问各方向的邻居网格
        for neighbor_node in self.neighbors:
            # 邻居存在且有空闲司机可借
            if neighbor_node is not None and neighbor_node.idle_driver_num > 0:
                # 该邻居可帮接的单数：本格剩单数与邻居空闲司机数的较小者
                num_assigned_order = min(self.order_num, neighbor_node.idle_driver_num)
                # 用该邻居的司机匹配本格订单，返回本次赚到的奖励
                rr = self.utility_assign_orders_neighbor(city, neighbor_node, num_assigned_order)
                # 累加到本格奖励
                reward += rr
                # 同时记入邻居的奖励账上（借出司机的贡献归邻居）
                neighbor_node_reward[neighbor_node.get_node_index()] += rr
                # 完成单数累加
                num_finished_orders += num_assigned_order
            # 本格订单已全部匹配完
            if self.order_num == 0:
                # 提前结束，不再遍历其余邻居
                break

        # 一致性校验：订单计数必须等于列表长度
        assert self.order_num == len(self.orders)
        # 返回 (奖励, 完成单数)
        return reward, num_finished_orders

    # 用邻居网格的空闲司机接本格的订单（匹配逻辑与 simple_order_assign_real 相同，
    # 只是司机来自邻居的司机字典，接单后从邻居字典移除）
    # 参数 city：城市环境；neighbor_node：出借司机的邻居网格；num_assigned_order：本轮要匹配的单数
    # 返回：本轮匹配的总奖励
    def utility_assign_orders_neighbor(self, city, neighbor_node, num_assigned_order):

        # 已服务订单下标
        served_order_index = []
        # 奖励累计器
        reward = 0
        # 取城市当前时刻作为接单时刻
        curr_city_time = city.city_time
        # 逐单匹配
        for idx in np.arange(num_assigned_order):
            # 取本格第 idx 条订单
            order_to_serve = self.orders[idx]
            # 记录接单时刻
            order_to_serve.set_assigned_time(curr_city_time)
            # 本格订单计数 -1
            self.order_num -= 1
            # 奖励累加该单价格
            reward += order_to_serve.get_price()
            # 记录已服务下标
            served_order_index.append(idx)
            # 遍历邻居的司机字典，找一个空闲司机接这单
            for key, assigned_driver in list(neighbor_node.drivers.items()):
                # 该司机空闲（不在送客且在线）
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    # 订单终点在目标区域内
                    if order_to_serve.get_end_position() is not None:
                        # 司机接单：转为送客状态
                        assigned_driver.take_order(order_to_serve)
                        # 接单司机从邻居的司机池移除
                        removed_driver = neighbor_node.drivers.pop(assigned_driver.get_driver_id(), None)
                        # 断言：移除成功
                        assert removed_driver is not None
                    # 订单终点在目标区域外
                    else:
                        # 司机置为离线（送出城后离开市场）
                        assigned_driver.set_offline()
                    # 全城空闲司机计数 -1
                    city.n_drivers -= 1
                    # 已找到司机，处理下一单
                    break

        # remove served orders
        # 从本格订单列表中删除已服务的订单
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        # 一致性校验：订单计数必须等于列表长度
        assert self.order_num == len(self.orders)

        # 返回本轮匹配的总奖励
        return reward


# Driver（司机）类：一名网约车司机。两个状态标志：
# online（在线/离线，离线=退出市场暂不接单）、onservice（是否正在送客）；
# 位置 node 指向其所在的 Node 网格（送客途中处于"不属于任何网格"的状态）
class Driver(object):
    # __slots__：限定实例属性集合（在线、送客中、当前订单、所在网格、本地时钟、司机 ID）
    __slots__ = ("online", "onservice", 'order', 'node', 'city_time', '_driver_id')

    # driver_id：司机的唯一编号（全城从 0 递增分配）
    def __init__(self, driver_id):
        # 默认在线（可接单）
        self.online = True
        # 默认不在送客
        self.onservice = False
        # 当前正在服务的订单（无单为 None）
        self.order = None     # the order this driver is serving
        # 司机当前所在的网格（尚未安置时为 None）
        self.node = None      # the node that contain this driver.
        # 司机本地时钟，与环境时间保持同步（判断订单何时完成要用）
        self.city_time = 0  # track the current system time

        # private
        # 唯一司机 ID（私有属性，通过 get_driver_id() 读取）
        self._driver_id = driver_id  # unique driver id.

    # 设置司机所在的网格
    # 参数 node：目标 Node 对象（调度在途时可传 None）
    def set_position(self, node):
        # 更新位置指向
        self.node = node

    # 记录司机开始服务的订单
    # 参数 order：要送的订单对象
    def set_order_start(self, order):
        # 保存当前订单
        self.order = order

    # 结束订单：清空订单并退出送客状态
    def set_order_finish(self):
        # 清空当前订单
        self.order = None
        # 退出送客状态（变回可接单）
        self.onservice = False

    # 返回司机的唯一 ID
    def get_driver_id(self):
        # 返回私有字段 _driver_id
        return self._driver_id

    # 司机本地时钟 +1（推进一个时间步）
    def update_city_time(self):
        # 本地时间自增
        self.city_time += 1

    # 直接设置司机本地时钟（环境每步统一同步所有司机的时间时调用）
    # 参数 city_time：要设置成的时刻
    def set_city_time(self, city_time):
        # 覆盖本地时间
        self.city_time = city_time

    # 空闲司机下线：退出市场暂不接单，但仍留在所在网格的司机字典中
    def set_offline(self):
        # 断言：只有"不在送客且在线"的司机才能下线
        assert self.onservice is False and self.online is True
        # 标记为离线
        self.online = False
        # 所在网格空闲司机数 -1
        self.node.idle_driver_num -= 1
        # 所在网格离线司机数 +1
        self.node.offline_driver_num += 1

    # 调度出发时下线：司机被派往其他网格、正在路上，期间不算空闲；
    # 不改动网格计数（调用前司机已从原网格移除、node 也已置 None）
    def set_offline_for_start_dispatch(self):

        # 断言：送客中的司机不能参与调度
        assert self.onservice is False
        # 标记为离线（在途状态）
        self.online = False

    # 离线司机重新上线：恢复接单资格，所在网格的计数随之调整
    def set_online(self):
        # 断言：送客中的司机不能切换在线状态
        assert self.onservice is False
        # 标记为在线
        self.online = True
        # 所在网格空闲司机数 +1
        self.node.idle_driver_num += 1
        # 所在网格离线司机数 -1
        self.node.offline_driver_num -= 1

    # 调度到达后重新上线：司机已由调用方加入目标网格，这里只恢复状态标志（不动网格计数）
    def set_online_for_finish_dispatch(self):

        # 标记为在线（回到可接单状态）
        self.online = True
        # 断言：必须不在送客
        assert self.onservice is False

    # 接单：记录订单并转为送客状态（仿真做了简化——不模拟沿途轨迹，
    # 司机在订单完成时直接"出现"在终点网格）
    # 参数 order：司机接下的订单
    def take_order(self, order):
        """ take order, driver show up at destination when order is finished
        """
        # 断言：必须在线才能接单
        assert self.online == True
        # 记录要送的订单
        self.set_order_start(order)
        # 转为送客状态
        self.onservice = True
        # 所在网格空闲司机数 -1（本格少了一个可接单的司机）
        self.node.idle_driver_num -= 1

    # 每个时间步检查司机状态：若正在送的订单恰好到完成时刻，
    # 司机移动到订单终点网格、退出送客状态并重新进入司机池
    # 参数 city：城市环境（读取当前时刻、更新全城空闲司机数）
    def status_control_eachtime(self, city):

        # 断言：司机本地时钟与城市时钟必须同步
        assert self.city_time == city.city_time
        # 正在送客
        if self.onservice is True:
            # 断言：送客中的司机必然在线
            assert self.online is True
            # 订单完成时刻 = 接单时刻 + 订单时长
            order_end_time = self.order.get_assigned_time() + self.order.get_duration()
            # 恰好到达完成时刻
            if self.city_time == order_end_time:
                # 司机位置移动到订单终点网格
                self.set_position(self.order.get_end_position())
                # 结束这单：清空订单、退出送客状态
                self.set_order_finish()
                # 把自己登记进终点网格的司机池
                self.node.add_driver(self._driver_id, self)
                # 全城空闲司机数 +1（司机重新可接单）
                city.n_drivers += 1
            # 还没到完成时刻
            elif self.city_time < order_end_time:
                # 送客途中，无需处理
                pass
            # 当前时间已超过完成时刻（不应出现的状态错误）
            else:
                # 抛出异常：订单完成时刻早于当前时间
                raise ValueError('Driver: status_control_eachtime(): order end time less than city time')


# Order（订单）类：一次网约车行程，对应数据五元组 [起点网格, 终点网格, 起始时刻, 时长(步), 价格]，
# 另带等待时长（超时未接作废）与接单时刻（初始 -1 表示尚未被接）
class Order(object):
    # __slots__：限定实例属性集合（起点、终点、起始时刻、时长、价格、等待时长、接单时刻）
    __slots__ = ('_begin_p', '_end_p', '_begin_t',
                 '_t', '_p', '_waiting_time', '_assigned_time')

    # 参数 begin_position：起点网格；end_position：终点网格（目标区域外为 None）；
    # begin_time：起始时刻；duration：时长（时间步数）；price：价格；
    # wait_time：等待时长（订单生成后最多可等这么久被接，超时作废）
    def __init__(self, begin_position, end_position, begin_time, duration, price, wait_time):
        # 起点网格对象
        self._begin_p = begin_position  # node
        # 终点网格对象（区域外订单为 None）
        self._end_p = end_position      # node
        # 起始时刻（订单产生的时间步）
        self._begin_t = begin_time
        # self._end_t = end_time
        # 订单时长（要经过几个时间步才送达）
        self._t = duration              # the duration of order.
        # 订单价格
        self._p = price
        # 等待时长：订单生成后可等待接单的步数
        self._waiting_time = wait_time  # a order can last for "wait_time" to be taken
        # 接单时刻，-1 表示尚未被任何司机接下
        self._assigned_time = -1

    # 返回起点网格对象
    def get_begin_position(self):
        # 起点网格
        return self._begin_p

    # 返回起点网格的编号
    def get_begin_position_id(self):
        # 起点网格对象 -> 其唯一编号
        return self._begin_p.get_node_index()

    # 返回终点网格对象（订单终点在目标区域外时为 None）
    def get_end_position(self):
        # 终点网格
        return self._end_p

    # 返回订单起始时刻
    def get_begin_time(self):
        # 起始时刻
        return self._begin_t

    # 记录订单被接下的时刻
    # 参数 city_time：接单时的城市时刻
    def set_assigned_time(self, city_time):
        # 保存接单时刻
        self._assigned_time = city_time

    # 返回接单时刻（尚未被接时为 -1）
    def get_assigned_time(self):
        # 接单时刻
        return self._assigned_time

    # def get_end_time(self):
    #     return self._end_t

    # 返回订单时长（时间步数）
    def get_duration(self):
        # 时长
        return self._t

    # 返回订单价格
    def get_price(self):
        # 价格
        return self._p

    # 返回订单等待时长
    def get_wait_time(self):
        # 等待时长
        return self._waiting_time
