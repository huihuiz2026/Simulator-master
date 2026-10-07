# 本文件：Python 2 -> 3 与 TensorFlow 兼容性迁移后的行为校验单元测试
# （覆盖坐标换算返回整数、一整天仿真的司机数守恒、调度动作生效，以及三种 RL 算法的完整前向/更新链路）
"""Behavior checks for the Python 3 and TensorFlow compatibility migration."""

import importlib  # 导入 importlib：用于按模块名字符串动态导入模块
import importlib.util  # 导入 importlib.util：其中的 find_spec 用来探测 TensorFlow 是否已安装
import unittest  # 导入 unittest：Python 标准库单元测试框架（提供 TestCase、断言方法和测试运行器）

import numpy as np  # 导入 NumPy 并简写为 np：用于 isfinite 等数值检查

from run_example import build_example_env  # 从同目录的 run_example.py 导入示例环境构造函数（3x3 网格小城市）
from simulator.utilities import ids_1dto2d, ids_2dto1d  # 导入被测对象：一维编号 <-> 二维坐标 的换算函数


# 仿真器（simulator 包）的 Python 3 兼容性测试类：坐标换算、全天司机数守恒、调度动作生效
class SimulatorCompatibilityTests(unittest.TestCase):
    # 测试点 1：一维编号 -> 二维坐标必须返回内置整数 int，且与 二维 -> 一维 互为逆运算
    # （Python 2 -> 3 迁移检查点：整除若误写成 "/" 会得到浮点数，浮点数不能直接当列表下标用）
    def test_grid_coordinates_are_integer_indices(self):
        for index in range(9):  # 遍历 3x3 地图全部 9 个格子的一维编号 0~8
            row, column = ids_1dto2d(index, 3, 3)  # 编号转 (行, 列)：如编号 4 -> (1, 1)
            self.assertIsInstance(row, int)  # 行号必须是 int 类型（不能是 float 等其他类型）
            self.assertEqual(ids_2dto1d(row, column, 3, 3), index)  # 再转回编号应与原编号相同（往返一致）

    # 测试点 2：不做任何调度跑完一整天 144 步：司机总数守恒、每步奖励有限、接单率在 [0,1] 内
    def test_full_day_preserves_driver_counts(self):
        env = build_example_env()  # 构造 3x3 网格的示例城市环境
        state = env.reset_clean()  # 重置环境（清空司机/订单并重新初始化），返回初始状态
        self.assertIsInstance(env.n_intervals, int)  # 一天总步数必须是整数（由 1440 // 10 计算而来）
        self.assertEqual(env.n_intervals, 144)  # 一天 = 1440 分钟 / 每步 10 分钟 = 144 步
        for _ in range(144):  # 逐步推进，仿真完整一天
            state, reward, _ = env.step([], generate_order=2)  # 空调度列表（不移动任何司机），使用预生成的固定订单
            self.assertTrue(np.isfinite(reward))  # 每步奖励必须是有限数（不能是 NaN/inf）
            self.assertEqual(state[0].sum(), env.n_drivers)  # 状态中的空闲司机总数 == 环境计数器记录的空闲司机数
            self.assertEqual(env.get_observation_driver_state().sum(), env.n_drivers)  # 逐格子重新累加的空闲司机数也应一致
            self.assertTrue(0 <= env.order_response_rate <= 1)  # 接单率（完成订单数/当步总订单数）应在 0~1 之间
        self.assertEqual(env.city_time, 144)  # 一天结束，城市时钟应恰好走到 144

    # 测试点 3：执行一条调度指令（向相邻格子移动 1 名司机），验证司机在下一步到达目的地
    def test_dispatched_driver_arrives_next_step(self):
        env = build_example_env()  # 构造示例城市环境
        env.reset_clean()  # 重置环境
        env.onoff_driver_location_mat[:] = 0  # 司机上下线参数全清零（均值=标准差=0，采样结果恒为 0），排除上下线干扰
        env.out_grid_in_orders[:] = 0  # "城外进入订单"计数清零（这类订单完成时会向城市补充司机），排除其影响
        env.day_orders = [[] for _ in range(144)]  # 144 个时刻的预生成订单全部置为空列表：后续不产生任何新订单
        for node in env.nodes:  # 遍历地图上全部 M*N=9 个格子
            if node is not None:  # 跳过无效格子（无效格在 nodes 列表中是 None）
                node.orders = []  # 清空该格子当前的订单列表
                node.order_num = 0  # 订单计数归零（保证本步没有任何订单、奖励为 0）
        origin = env.nodes[env.target_grids[0]]  # 取第一个有效格子作为调度起点（示例地图左上角的网格）
        destination = next(node for node in origin.neighbors if node is not None)  # 取起点邻居中第一个真实存在的格子作为目的地（None 表示该方向越界或是无效格）
        env.utility_add_driver_real_nodewise(origin.get_node_index(), 1)  # 在起点格子放置 1 名空闲司机（无下线司机时新建司机）
        origin_count = origin.idle_driver_num  # 记录起点在调度前的空闲司机数
        destination_count = destination.idle_driver_num  # 记录目的地在调度前的空闲司机数
        total = env.n_drivers  # 记录全城司机总数
        _, reward, _ = env.step(  # 推进一个时间步并执行调度动作
            [(origin.get_node_index(), destination.get_node_index(), 1)],  # 调度列表：[(起点编号, 目的地编号, 司机数 1)]
            generate_order=2)  # 使用（已清空的）预生成订单，即本步不生成新订单
        self.assertEqual(origin.idle_driver_num, origin_count - 1)  # 起点空闲司机应减少 1（司机已出发）
        self.assertEqual(destination.idle_driver_num, destination_count + 1)  # 目的地空闲司机应增加 1（司机已到达）
        self.assertEqual(env.n_drivers, total)  # 全城司机总数不变（司机只是换了格子，没有消失）
        self.assertEqual(reward, 0)  # 已清空所有订单，本步奖励应为 0


# 算法（algorithm 包）的 TensorFlow 兼容性测试类：仅当本机安装了 TensorFlow 时才执行，
# 否则整个类被跳过（find_spec 找不到 tensorflow 时返回 None，skipUnless 的条件不成立）
@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
class AlgorithmCompatibilityTests(unittest.TestCase):
    # 三种算法共用的校验流程：按名字动态导入算法模块，完整走一遍
    # "构造状态 -> 网络前向预测 -> 选取动作 -> 计算 TD 目标 -> 更新网络" 的链路
    # 参数: name 算法模块名，取 "IDQN"、"cDQN" 或 "cA2C"
    def check_algorithm(self, name):
        module = importlib.import_module("algorithm." + name)  # 按字符串名字动态导入对应的算法模块
        tf = module.tf  # 取出该模块使用的 TensorFlow（经 tf_compat 以 TF1 兼容模式运行）
        env = build_example_env()  # 构造 3x3 示例城市环境
        raw_state = env.reset_clean()  # 重置环境得到原始全局状态（形状 2xMxN：空闲司机分布 + 订单分布）
        grid_count = env.n_valid_grids  # 有效格子数（示例地图 9 格中有 1 个无效格，故为 8）
        # 原始状态展平后前 M*N 个数是司机分布、后 M*N 个是订单分布，
        # 因此有效格子在两段中的下标分别是 target_grids 和 target_grids + M*N，两段拼接成 indices
        indices = env.target_grids + [i + env.M * env.N for i in env.target_grids]  # 有效格子状态在展平数组中的下标
        processor = module.stateProcessor(indices, env.target_grids, grid_count)  # 构造状态预处理器（负责提取/归一化/扩展状态）
        state = processor.utility_conver_states(raw_state)  # 展平原始状态并只保留 8 个有效格子的司机数和订单数（共 16 个值）
        states = processor.to_grid_states(processor.utility_normalize_states(state), 0)  # 先归一化，再扩展为 8 个格子各自的状态向量（含时间/格子 one-hot），时刻取 0
        with tf.Graph().as_default(), tf.Session() as session:  # 新建一张独立计算图并打开会话（TF1 风格用法）
            tf.set_random_seed(1)  # 固定随机种子，保证测试结果可复现
            estimator = module.Estimator(session, 7, grid_count * 3 + 144, env)  # 构造算法网络：7 个动作（6 个邻居方向 + 原地不动），状态维度 3*8+144
            session.run(tf.global_variables_initializer())  # 初始化网络全部权重
            prediction = estimator.predict(states)  # 前向计算：对每个格子的状态输出一个预测值（Q 值/状态价值）
            self.assertEqual(prediction.size, grid_count)  # 输出个数必须等于有效格子数（每格一个值）
            self.assertTrue(np.isfinite(prediction).all())  # 所有输出必须是有限数（无 NaN/inf，网络数值稳定）
            result = estimator.action(states, np.ones(grid_count), 0.1)  # 选取动作：每个格子给 1 名可调度司机（全 1 上下文），探索率 0.1
            actions = result[4] if name == "IDQN" else result[0]  # 三种算法 action 返回的元组结构不同：IDQN 的动作列表在第 5 项，cDQN/cA2C 在第 1 项
            for origin, destination, count in actions:  # 遍历每条调度动作 (起点格子, 目的地格子, 调度司机数)
                self.assertIn(env.nodes[destination], env.nodes[origin].neighbors)  # 目的地必须是起点的相邻格子（调度只允许派往邻居）
                self.assertGreater(count, 0)  # 每条动作调度的司机数必须大于 0
            if name == "IDQN":  # IDQN（独立 DQN）分支：对 Q 网络做一次监督式更新
                selected = np.eye(7, dtype=np.float32)[np.zeros(grid_count, dtype=int)]  # 用 7x7 单位矩阵按 8 个 0 取行，得到 8x7 的 one-hot 矩阵：每格都选第 0 号动作
                loss = estimator.update(states, selected, np.ones(grid_count), 1e-3, 0)  # 以全 1 为目标值、学习率 1e-3 更新一次，返回损失
            else:  # cDQN / cA2C 分支：两者都基于价值网络，先验证动作概率再计算 TD 目标
                probabilities = result[1]  # 取各格子的动作概率分布（cDQN、cA2C 均为 action 返回的第 2 项）
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)  # 每个格子的动作概率之和应约为 1（容差 1e-6）
                targets = estimator.compute_targets(  # 计算 TD 目标值：节点奖励 + gamma * 下一状态价值
                    probabilities, states, np.ones((grid_count, 1)), 0.9)  # 节点奖励取全 1（8x1 矩阵），折扣因子 gamma=0.9
                self.assertEqual(targets.shape, (grid_count, 1))  # 目标值形状必须为 (格子数, 1)
                if name == "cDQN":  # cDQN（集中式 DQN）分支
                    loss = estimator.update(states, targets, 1e-3, 0)  # 用 TD 目标对价值网络更新一次
                else:  # cA2C（集中式 A2C：价值网络 + 策略网络）分支
                    loss = estimator.update_value(states, targets, 1e-3, 0)  # 先用 TD 目标更新价值网络
                    advantages = estimator.compute_advantage(  # 计算优势值：奖励 + gamma*下一状态价值 - 当前状态价值
                        result[4], result[6], states, np.ones((grid_count, 1)), 0.9)  # 依次为当前状态价值(result[4])、下一状态格子编号(result[6])、状态、全 1 奖励、gamma=0.9
                    policy_loss = estimator.update_policy(  # 用优势值对策略网络做一次策略梯度更新
                        result[2], np.asarray(advantages).reshape(-1, 1),  # 策略输入状态(result[2])、优势值拉直成 8x1 列向量
                        result[3], result[5], 1e-3, 0)  # 所选动作 one-hot(result[3])、邻居掩码(result[5])、学习率 1e-3、全局步数 0
                    self.assertTrue(np.isfinite(policy_loss))  # 策略损失必须是有限数
            self.assertTrue(np.isfinite(loss))  # 最终损失必须是有限数（更新过程数值正常）

    def test_idqn(self):  # 测试 IDQN 算法（独立 DQN）
        self.check_algorithm("IDQN")  # 对 IDQN 模块运行公共校验流程

    def test_cdqn(self):  # 测试 cDQN 算法（集中式 DQN，代码中说明其本质是期望 SARSA）
        self.check_algorithm("cDQN")  # 对 cDQN 模块运行公共校验流程

    def test_ca2c(self):  # 测试 cA2C 算法（集中式 A2C：价值网络 + 策略网络）
        self.check_algorithm("cA2C")  # 对 cA2C 模块运行公共校验流程


# 作为脚本直接运行时才执行以下代码（被其他模块导入时不执行）
if __name__ == "__main__":
    unittest.main()  # 启动 unittest 测试运行器：自动发现并运行本文件中的全部测试方法
