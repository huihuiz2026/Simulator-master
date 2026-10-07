import numpy as np  # 导入 NumPy 数值计算库并简写为 np：本文件主要用它创建全零矩阵
import os  # 导入 os 模块：用它的 makedirs 创建目录、isdir 判断路径是否为目录
import errno  # 导入 errno 模块：用其中的错误码常量 EEXIST（表示"文件/目录已存在"）



from datetime import datetime, timedelta  # 导入日期时间类 datetime 和时间增量类 timedelta：供下面的 datetime_range 生成时间序列

# 生成器函数：从 start 到 end（不含 end）按步长 delta 逐个产出时间点，
# 例如 datetime_range(0点, 次日0点, 10分钟) 会依次给出一天 144 个时刻
# 参数: start 起始时间，end 结束时间（不包含在内），delta 步长（如 timedelta(minutes=10)）
def datetime_range(start, end, delta):
    current = start  # 当前时间点，从 start 开始
    while current < end:  # 只要还没越过结束时间就继续
        yield current  # 产出当前时间点（yield 使本函数成为生成器，被迭代时逐个给值）
        current += delta  # 当前时间点前进一个步长


# 递归创建目录，效果等同 shell 的 mkdir -p：目录已存在时不报错
# 参数: path 要创建的目录路径
def mkdir_p(path):
    try:  # 尝试创建目录（makedirs 会把缺失的各级父目录一并创建）
        os.makedirs(path)  # 递归创建目录
    # 捕获创建过程中的 OSError（下一行的英文注释是原作者说明：Python 2.5 以上适用）
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):  # 错误码是"已存在"且该路径确实是个目录
            pass  # 目录本来就有，视为成功，忽略此错误
        else:  # 不是"目录已存在"这类可忽略的错误
            raise  # 其他错误（如权限不足）原样重新抛出


# 二维坐标 (i,j) 转一维编号：把 M 行 N 列的地图按"行优先"（一行接一行）方式展平成 M*N 的列表
# 编号公式: index = 行号 * N + 列号。例: 3x3 地图中 (1,1) -> 1*3+1 = 4
# 参数: i 行号（0 ~ M-1），j 列号（0 ~ N-1），M 地图行数，N 地图列数
def ids_2dto1d(i, j, M, N):
    '''
    convert (i,j) in a M by N matrix to index in M*N list. (row wise)
    matrix: [[1,2,3], [4, 5, 6]]
    list: [0, 1, 2, 3, 4, 5, 6]
    index start from 0
    '''
    assert 0 <= i < M and 0 <= j < N  # 断言坐标在地图范围内，越界立即报错（防御性检查）
    index = i * N + j  # 行优先展平公式：前 i 行共占 i*N 个格子，再加本行的第 j 列
    return index  # 返回一维编号（0 ~ M*N-1）


# 一维编号转二维坐标 (i,j)：ids_2dto1d 的逆运算
# 例: 3x3 地图中编号 4 -> (4 // 3, 4 - 3*1) = (1, 1)，即第 1 行第 1 列
# 参数: ids 一维编号（0 ~ M*N-1），M 地图行数，N 地图列数
def ids_1dto2d(ids, M, N):
    ''' inverse of ids_2dto1d(i, j, M, N)
        index start from 0
    '''
    i = ids // N  # 行号 = 编号整除列数（前面已排了 i 行，每行 N 个格子）
    j = ids - N * i  # 列号 = 编号减去前 i 行占掉的格子数（等价于编号对 N 取余）
    return (i, j)  # 返回 (行号, 列号) 元组


# 获取格子 (i,j) 周围的邻居节点列表：n=6 为六边形邻域（本项目默认），n=4 为四邻域
# 六边形邻域的 6 个方向见下方英文示意图，按列表下标 0~5 依次是：上、右上、右下、下、左下、左上；
# 六边形网格按"列错开"排布，列号 j 的奇偶决定左右两侧斜向邻居偏上还是偏下
# 参数: i,j 中心格子的行列号；M,N 地图行数、列数；n 邻域边数（4 或 6）；nodes 按一维编号存放全部格子节点的列表
# 返回: 长度为 n 的列表，第 k 个元素是方向 k+1 的邻居节点对象；该方向越界（地图上无格子）时保持 None
def get_neighbor_list(i, j, M, N, n, nodes):
    ''' n: n-sided polygon, construct for a 2d map
                 1
             6       2
               center
             5       3
                 4
    return index of neighbor 1, 2, 3, 4, 5,6 in the matrix
    '''

    neighbor_list = [None] * n  # 先用 n 个 None 占位：越界方向没有邻居，对应位置保持 None
    if n == 6:  # 六边形邻域：每个格子最多 6 个邻居
        # hexagonal
        if j % 2 == 0:  # 偶数列：右上邻居与本行同排，右下邻居在下一行
            if i - 1 >= 0:  # 上方（行号-1）不越界
                neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]  # 方向1 上：(i-1, j)
            if j + 1 < N:  # 右侧一列不越界
                neighbor_list[1] = nodes[ids_2dto1d(i,   j+1, M, N)]  # 方向2 右上：(i, j+1)
            if i + 1 < M and j + 1 < N:  # 下一行、右一列都不越界
                neighbor_list[2] = nodes[ids_2dto1d(i+1, j+1, M, N)]  # 方向3 右下：(i+1, j+1)
            if i + 1 < M:  # 下方（行号+1）不越界
                neighbor_list[3] = nodes[ids_2dto1d(i+1, j,   M, N)]  # 方向4 下：(i+1, j)
            if i + 1 < M and j - 1 >= 0:  # 下一行、左一列都不越界
                neighbor_list[4] = nodes[ids_2dto1d(i+1, j-1, M, N)]  # 方向5 左下：(i+1, j-1)
            if j - 1 >= 0:  # 左侧一列不越界
                neighbor_list[5] = nodes[ids_2dto1d(i,   j-1, M, N)]  # 方向6 左上：(i, j-1)
        elif j % 2 == 1:  # 奇数列：右侧两个斜向邻居整体上移一行（右上到上一行、右下到本行）
            if i - 1 >= 0:  # 上方不越界
                neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]  # 方向1 上：(i-1, j)
            if i - 1 >= 0 and j + 1 < N:  # 上一行、右一列都不越界
                neighbor_list[1] = nodes[ids_2dto1d(i-1, j+1, M, N)]  # 方向2 右上：(i-1, j+1)
            if j + 1 < N:  # 右侧一列不越界
                neighbor_list[2] = nodes[ids_2dto1d(i,   j+1, M, N)]  # 方向3 右下：(i, j+1)
            if i + 1 < M:  # 下方不越界
                neighbor_list[3] = nodes[ids_2dto1d(i+1, j,   M, N)]  # 方向4 下：(i+1, j)
            if j - 1 >= 0:  # 左侧一列不越界
                neighbor_list[4] = nodes[ids_2dto1d(i,   j-1, M, N)]  # 方向5 左下：(i, j-1)
            if i - 1 >= 0 and j - 1 >= 0:  # 上一行、左一列都不越界
                neighbor_list[5] = nodes[ids_2dto1d(i-1, j-1, M, N)]  # 方向6 左上：(i-1, j-1)
    elif n == 4:  # 四邻域（正方形网格）：只有上、右、下、左 4 个方向
        # square
        if i - 1 >= 0:  # 上方不越界
            neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]  # 方向1 上：(i-1, j)
        if j + 1 < N:  # 右侧不越界
            neighbor_list[1] = nodes[ids_2dto1d(i,   j+1, M, N)]  # 方向2 右：(i, j+1)
        if i + 1 < M:  # 下方不越界
            neighbor_list[2] = nodes[ids_2dto1d(i+1, j,   M, N)]  # 方向3 下：(i+1, j)
        if j - 1 >= 0:  # 左侧不越界
            neighbor_list[3] = nodes[ids_2dto1d(i,   j-1, M, N)]  # 方向4 左：(i, j-1)

    return neighbor_list  # 返回邻居节点列表（越界方向为 None）


# 返回格子 (i,j) 的 6 个六边形邻居的二维坐标，方向与 get_neighbor_list 的 n=6 分支一致
# 注意：本函数不做越界检查，地图边缘格子的某些邻居坐标可能为负数或超出地图范围
# 参数: i,j 中心格子的行列号
# 返回: 6 个 [行, 列] 坐标组成的列表，顺序对应方向 1~6（上、右上、右下、下、左下、左上）
def get_neighbor_index(i, j):
    """
                 1
             6       2
                center
             5       3
                 4
    return index of neighbor 1, 2, 3, 4, 5,6 in the matrix
    """
    neighbor_matrix_ids = []  # 存放 6 个邻居坐标的列表（先置空）
    if j % 2 == 0:  # 偶数列
        neighbor_matrix_ids = [[i - 1, j    ],  # 方向1 上：(i-1, j)
                               [i,     j + 1],  # 方向2 右上：(i, j+1)
                               [i + 1, j + 1],  # 方向3 右下：(i+1, j+1)
                               [i + 1, j    ],  # 方向4 下：(i+1, j)
                               [i + 1, j - 1],  # 方向5 左下：(i+1, j-1)
                               [i    , j - 1]]  # 方向6 左上：(i, j-1)
    elif j % 2 == 1:  # 奇数列：斜向邻居整体上移一行
        neighbor_matrix_ids = [[i - 1, j    ],  # 方向1 上：(i-1, j)
                               [i - 1, j + 1],  # 方向2 右上：(i-1, j+1)
                               [i    , j + 1],  # 方向3 右下：(i, j+1)
                               [i + 1, j    ],  # 方向4 下：(i+1, j)
                               [i    , j - 1],  # 方向5 左下：(i, j-1)
                               [i - 1, j - 1]]  # 方向6 左上：(i-1, j-1)

    return neighbor_matrix_ids  # 返回 6 个邻居的 [行, 列] 坐标（可能包含越界坐标）


# 按层获取格子 (i,j) 周围的邻居（相当于逐层展开的广度优先搜索）：
# 第 1 层是 1 步可达的邻居（最多 6 个格子），第 2 层是恰好 2 步可达的新格子，依此类推；
# 中心格子和已在更早层出现过的格子不会重复收录
# 参数: i,j 中心格子的行列号；l_max 最大层数（>=1）；M,N 地图行数、列数
# 返回: layers_neighbors[k] 为第 k+1 层邻居的 [行, 列] 坐标列表（已过滤越界坐标），共 l_max 层
def get_layers_neighbors(i, j, l_max, M, N):
    """get neighbors of node layer by layer, todo BFS.
       i, j: center node location
       L_max: max number of layers
       layers_neighbors: layers_neighbors[0] first layer neighbor: 6 nodes: can arrived in 1 time step.
       layers_neighbors[1]: 2nd layer nodes id
       M, N: matrix rows and columns.
    """
    assert l_max >= 1  # 断言最大层数至少为 1：保证至少算出第一层邻居
    layers_neighbors = []  # 收集各层邻居坐标的结果列表
    # 调用 get_neighbor_index 得到中心 6 个方向的邻居坐标（注意：可能包含越界坐标）
    layer1_neighbor = get_neighbor_index(i, j)  #[[1,1], [0, 1], ...]
    temp = []  # 暂存第一层中未越界的邻居坐标
    for item in layer1_neighbor:  # 逐个检查 6 个方向的邻居
        x, y = item  # 取出该邻居的行列坐标
        if 0 <= x <= M-1 and 0 <= y <= N-1:  # 坐标在地图范围内（未越界）
            temp.append(item)  # 收入第一层邻居
    layers_neighbors.append(temp)  # 第一层邻居坐标入列

    node_id_neighbors = []  # 暂存第一层邻居的一维编号
    for item in layer1_neighbor:  # 再次遍历第一层邻居
        x, y = item  # 取出行列坐标
        if 0 <= x <= M-1 and 0 <= y <= N-1:  # 未越界
            node_id_neighbors.append(ids_2dto1d(x, y, M, N))  # 行列坐标转一维编号后收集

    layers_neighbors_set = set(node_id_neighbors)  # 用集合记录"已出现过"的格子编号（初始为第一层）
    curr_node_id = ids_2dto1d(i, j, M, N)  # 中心格子的一维编号
    layers_neighbors_set.add(curr_node_id)  # 中心格子也计入"已出现"，避免被后续层重复收录

    t = 1  # 当前已算好的层数（第一层已完成）
    while t < l_max:  # 还不足 l_max 层就继续向外扩张
        t += 1  # 开始计算第 t 层（t 从 2 递增到 l_max）
        layer_neighbor_temp = []  # 候选列表：上一层所有格子的全部邻居坐标（未去重、未过滤）
        for item in layers_neighbors[-1]:  # 遍历上一层的每个格子（[-1] 取列表最后一个元素）
            x, y = item  # 取出行列坐标
            if 0 <= x <= M-1 and 0 <= y <= N-1:  # 上一层格子本身有效（未越界）
                layer_neighbor_temp += get_neighbor_index(x, y)  # 把它的 6 个邻居坐标全部加入候选

        # 存放本层新格子（行尾英文注释意为：去掉已出现过的旧层邻居）
        layer_neighbor = []  # remove previous layer neighbors
        for item in layer_neighbor_temp:  # 逐个检查候选坐标
            x, y = item  # 取出行列坐标
            if 0 <= x <= M-1 and 0 <= y <= N-1:  # 未越界
                node_id = ids_2dto1d(x, y, M, N)  # 转成一维编号，用于查重
                if node_id not in layers_neighbors_set:  # 中心及更早层都没出现过（本层新格子）
                    layer_neighbor.append(item)  # 收入本层
                    layers_neighbors_set.add(node_id)  # 标记为已出现（同一格子不会进两层）
        layers_neighbors.append(layer_neighbor)  # 本层邻居坐标入列

    return layers_neighbors  # 返回各层邻居：第 k 个元素是恰好 k+1 步可达的格子坐标列表


# 统计环境中空闲司机的分布：遍历所有司机，把"在线且未在接单服务中"的司机
# 按其所在格子累加到一个 MxN 矩阵里（矩阵第 row 行 col 列 = 该格子当前的空闲司机数）
# 参数: env 城市仿真环境（CityReal 对象，含 M、N、drivers 等属性）
def get_driver_status(env):
    idle_driver_dist = np.zeros((env.M, env.N))  # 全零 MxN 矩阵，用于累加各格子的空闲司机数
    for driver_id, cur_drivers in list(env.drivers.items()):  # 遍历环境登记的全部司机（转 list 以免遍历时字典变动报错）
        if cur_drivers.node is not None:  # 司机当前有位置（正处在某个格子上，而非调度途中）
            node_id = cur_drivers.node.get_node_index()  # 司机所在格子的一维编号
            row, col = ids_1dto2d(node_id, env.M, env.N)  # 格子编号转成 (行, 列) 坐标
            if cur_drivers.onservice is False and cur_drivers.online is True:  # 在线且空闲（未在送客）
                idle_driver_dist[row, col] += 1  # 对应格子的空闲司机计数 +1

    return idle_driver_dist  # 返回 MxN 的空闲司机分布矩阵

# 调试辅助函数：以表格形式打印某个格子上所有司机的状态（编号、所在位置、是否在线、是否在服务）
# 参数: node 要查看的格子（Node 对象）
def debug_print_drivers(node):
    print("Status of all drivers in the node {}".format(node.get_node_index()))  # 打印标题：该格子的编号
    print("|{:12}|{:12}|{:12}|{:12}|".format("driver id", "driver location", "online", "onservice"))  # 打印表头（每列占 12 个字符宽）

    for driver_id, cur_drivers in list(node.drivers.items()):  # 遍历该格子上的所有司机
        if cur_drivers.node is not None:  # 司机有位置
            node_id = cur_drivers.node.get_node_index()  # 取其所在格子的编号
        else:  # 司机当前没有位置（如在调度途中）
            node_id = "none"  # 无位置时显示 none
        print("|{:12}|{:12}|{:12}|{:12}|".format(driver_id, node_id, cur_drivers.online, cur_drivers.onservice))  # 打印一行司机状态
