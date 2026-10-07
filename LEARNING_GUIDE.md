# Simulator 代码库学习指南

> 为"会常规 Python、但没有机器学习/强化学习背景"的你定制。
> 代码库中所有文件都已加上逐行中文注释，本指南配合注释使用。
> 预计总学习时间：4~6 周（每天 1~2 小时的节奏）。

---

## 1. 这个项目是什么

这是 KDD 2018 论文《Efficient Large-Scale Fleet Management via Multi-Agent Deep
Reinforcement Learning》（Lin, Zhao, Xu, Zhou）的官方代码库。论文 PDF 就在仓库根目录。

**它解决的问题**：网约车平台（滴滴式）一天中不同时刻、不同城区的供需不匹配——
早高峰 A 区打不到车、B 区却堆着大量空车。论文用**强化学习**学习一个"空车调度策略"：
每个时间步决定把哪些空闲司机提前挪到哪些相邻区域，让总接单金额（GMV）最大化。

**三层结构**（自底向上）：

```
simulator/          仿真环境：城市网格、司机、订单、匹配规则（纯 Python+NumPy，无 ML）
    ↑ 被调用
algorithm/          学习算法：IDQN / cDQN / cA2C（TensorFlow 实现）
    ↑ 被调用
run/                训练入口脚本：读真实数据 → 建环境 → 建算法 → 训练循环 → 存模型
```

**核心交互范式**（所有强化学习代码都是这个循环）：

```
state = env.reset_clean()                 # 拿到初始状态（全城供需快照）
while 没结束:
    action = 算法根据 state 决策            # 调度哪些空闲司机去哪里
    next_state, reward, info = env.step(action)   # 环境推进 10 分钟
    算法用 (state, action, reward, next_state) 学习   # 更新网络参数
```

- **状态 (state)**：2×M×N 数组 = [每格空闲司机数, 每格订单数]
- **动作 (action)**：每格从 7 个方向中选一个（6 个六边形邻居 + 原地不动），`action_dim=7`
- **奖励 (reward)**：这一步所有成功匹配的订单价格之和（近似 GMV）

---

## 2. 代码库地图

| 文件 | 行数 | 内容 | 难度 | 依赖 |
|---|---|---|---|---|
| `tests/run_example.py` | 161 | 人工数据示例（无策略跑一天）| ★ | simulator |
| `simulator/utilities.py` | 198 | 工具函数：网格 id 换算、六边形邻居、时间序列 | ★ | 无 |
| `simulator/objects.py` | 469 | 实体类：Node（网格）/ Driver（司机）/ Order（订单）| ★★ | utilities |
| `simulator/envs.py` | 743 | CityReal 城市环境（最核心）：reset/step/订单匹配/奖励 | ★★★ | objects |
| `algorithm/tf_compat.py` | 5 | TF1→TF2 兼容开关 | ★ | TensorFlow |
| `algorithm/alg_utility.py` | 242 | 算法公共件：网络层、概率分布、动作投影、状态转换 | ★★★ | TF |
| `algorithm/IDQN.py` | 377 | 独立多智能体 DQN | ★★★★ | alg_utility |
| `algorithm/cDQN.py` | 377 | 上下文 DQN（邻居广播奖励）| ★★★★ | alg_utility |
| `algorithm/cA2C.py` | 528 | 上下文 A2C（论文最优方法）| ★★★★★ | alg_utility |
| `run/run_baseline_nopolicy.py` | 147 | 真实数据无策略基线 | ★★ | simulator |
| `run/run_IDQN.py` | 218 | IDQN 训练入口（结构最简，先读它）| ★★★★ | IDQN |
| `run/run_cDQN.py` | 220 | cDQN 训练入口 | ★★★★ | cDQN |
| `run/run_cA2C.py` | 240 | cA2C 训练入口 | ★★★★★ | cA2C |
| `tests/test_python3.py` | 114 | Python3 迁移兼容性测试 | ★★ | 全部 |

> 注：表中"行数"是加注释前的原始行数。现在所有文件都已完成逐行中文注释，
> 实际行数约为原来的 1.1~1.7 倍。

**读代码总原则**：先 `simulator` 后 `algorithm` 最后 `run`；先小文件后大文件；
同一个文件里先读类定义和 `__init__`，再读被别人调用的方法，最后读内部工具方法。

---

## 3. 学习路径总览

| 阶段 | 目标 | 时间 | 前置 |
|---|---|---|---|
| 0 | 环境准备、跑通示例 | 0.5 天 | 无 |
| 1 | 吃透仿真环境（simulator/）| 1 周 | 阶段 0 |
| 2 | RL 理论充电（概念级）| 1 周 | **可与阶段 1 并行** |
| 3 | 算法基础件 + IDQN | 1 周 | 阶段 1+2 |
| 4 | cDQN、cA2C（论文核心）| 1 周 | 阶段 3 |
| 5 | 训练脚本 + 论文精读 | 3~5 天 | 阶段 4 |

---

## 4. 分阶段详解

### 阶段 0：跑起来（0.5 天）

```bash
cd Simulator-master
python tests/run_example.py                    # 人工数据示例：无策略跑一天
python -m unittest discover -s tests -p 'test_*.py' -v   # 兼容性测试
```

- [ ] 跑通 `run_example.py`，看懂输出三行各是什么（接单率 / 空闲司机一致性 / 司机总数）
- [ ] 对照 `run_example.py` 的注释，理解"构造环境 → reset → 循环 step"的骨架

### 阶段 1：吃透仿真环境（1 周）

**阅读顺序**（每个文件先通读注释，再回头精读标 ★ 的方法）：

1. `simulator/utilities.py` —— 热身。理解 `ids_1dto2d/ids_2dto1d`（网格坐标换算）
   和六边形邻居是怎么算出来的
2. `simulator/objects.py` —— 三个实体类
   - ★ `Node.simple_order_assign_real`：单格订单匹配（本项目的"经济学核心"：
     匹配 `min(订单数, 空闲司机数)` 单，每单把价格累加进 reward）
   - `Node.simple_order_assign_broadcast_update`：本格司机不够时向邻居借
   - Driver 的状态机：online / onservice / offline
3. `simulator/envs.py` —— 主战场，分四块读：
   - 构造：`__init__` / `construct_node_real` / `construct_map_real`（地图→网格→邻居）
   - 重置：★ `reset_clean`（清场→装载一天订单→放初始司机→返回观测）
   - 推进：★ `step`（八步流水线，全项目最重要的一段代码）
   - 匹配与奖励：★ `step_assign_order_broadcast_neighbor_reward_update`
     （两轮匹配 + 逐格奖励归因，返回的 info 是后面所有算法的"上下文"来源）
   - 司机供给侧：`step_driver_online_offline_nodewise`（onoff 矩阵如何让司机上下线）

**动手实验**（强烈建议，加深理解）：
- [ ] 复制 `run_example.py` 为 `my_exp.py`，把 `order_real` 每步订单从 9 条改成 30 条，
      观察接单率如何变化；再把初始司机数 `idle_driver_dist_time` 从 10 改成 3，对比结果
- [ ] 写 5 行代码打印 `state[0]`（司机分布矩阵），验证它和打印的空闲司机数一致

**自测**（能回答就过关）：
1. `env.n_drivers` 和 `len(env.drivers)` 差在哪？
2. `generate_order=1` 和 `=2` 的区别？
3. reward 是怎么算出来的？为什么设计成"订单价格之和"？
4. 司机总数为什么从 20 涨到 335？（提示：进城订单 + onoff 矩阵）

### 阶段 2：RL 理论充电（1 周，可与阶段 1 并行）

不需要数学推导，**概念级理解**即可。重点概念清单（按顺序学）：

| 概念 | 一句话理解 | 对应代码 |
|---|---|---|
| MDP（马尔可夫决策过程）| 用"状态-动作-奖励-转移"描述序贯决策问题 | 整个 `envs.py` 就是一个 MDP |
| 策略 π | 从状态到动作的映射（决策函数）| `Estimator.action()` |
| 折扣因子 γ | 未来奖励打几折（0.9 = 明天一块钱值今天九毛）| `GAMMA = 0.9` |
| Q 函数 | "在状态 s 做动作 a"的长期总回报期望 | `q_estimator` 整个网络 |
| Bellman 方程 | Q(s,a) = r + γ·max Q(s',·)，递归分解 | `r_grid + gamma * target_estimator.predict(...)` |
| DQN | 用神经网络近似 Q 函数 | `IDQN.Estimator` |
| 经验回放 | 把交互样本存池子里随机抽样训练（打破相关性）| `ReplayMemory` |
| target network | 参数滞后复制的副网络，稳定训练目标 | `target_estimator` + `ModelParametersCopier` |
| ε-greedy | 小概率随机探索，大概率贪心 | `run_*.py` 里的 `epsilons` |
| 策略梯度 | 直接调参数让"好动作"概率变大 | `cA2C._build_mlp_policy` |
| actor-critic | actor 出动作 + critic 估价值，互相配合 | cA2C 的两个网络 |
| 优势函数 A | 动作比平均好多少（指导 actor 更新）| `compute_advantage` |

**推荐资源**（按顺序，够用就停）：
1. 莫烦 Python 强化学习系列（中文视频，搜索"莫烦 强化学习"）——最快建立直觉
2. OpenAI Spinning Up（spinningup.openai.com，有中文翻译）——"Key Algorithms"几篇短文
   把 DQN/Actor-Critic 讲得极清楚
3. Sutton & Barto《Reinforcement Learning: An Introduction》第 3~6 章
   （免费电子版：incompleteideas.net/book/the-book.html）——权威教材，选读
4. David Silver 的 RL 公开课（YouTube，有中文字幕版）——理论更系统，可选

### 阶段 3：算法基础件 + IDQN（1 周）

**阅读顺序**：

1. `algorithm/tf_compat.py`（5 行）——明白项目在 TF2 下用 TF1 老式 API
   （`placeholder`/`session`/`run`，"先搭计算图、再喂数据执行"的两段式编程）
2. `algorithm/alg_utility.py` ——分两类读：
   - 神经网络积木：`fc` / `dense` / `ortho_init`（全连接层怎么搭）
   - 状态与动作工具：`utility_conver_states`（环境状态 → 网络输入）、
     `collision_action`（统计冲突动作）、`continuous_quadratic_knapsack`（动作投影，
     把"要调几个司机"限制在合法范围内，较难，可最后攻克）
3. `algorithm/IDQN.py` ——第一个完整算法：
   - `Estimator._build_model`：3 层 MLP（128→64→32→7）输出每格 7 个动作的 Q 值；
     损失 = (TD 目标 − Q(s,选的动作))²，Adam 优化器
   - `Estimator.action`：ε-greedy 选动作 + 有效动作掩码（边界格邻居不足 6 个）
   - `stateProcessor`：把环境状态切成"每格一条样本"（关键设计：一个网络批量给
     所有网格出动作 → 多智能体共享一个模型）
   - `ReplayMemory`：经验回放池
   - `ModelParametersCopier`：target network 参数复制器

**关键设计要理解透**：`state_dim = n_valid_grids × 3 + 144`
——每个网格 3 个通道（空闲司机数、订单数、网格身份编码）+ 144 维时刻编码。
把全城状态一次性喂给一个网络，输出所有格子各自的动作。

### 阶段 4：cDQN 与 cA2C —— 论文的核心贡献（1 周）

**先想清楚动机**：IDQN 每格独立学习，但你的调度会影响邻居的收益
（我把车调走，邻居格子反而受益）——"信用分配"问题。

1. `algorithm/cDQN.py`：与 IDQN 结构几乎相同，**对比着读**，重点看差异：
   - `Estimator.action(s, context, epsilon)` 里 context（上下文）怎么拼进输入
   - `compute_targets`：用广播奖励修正 TD 目标
   - 环境侧来源：`envs.py` 的 `step_assign_order_broadcast_neighbor_reward_update`
     返回的逐格奖励 `info`，经 `stateProcessor.compute_context` 转换
2. `algorithm/cA2C.py`（论文最优方法，528 行，最难也最重要）：
   - `_build_value_model`：critic 网络（128→64→32→1，输出状态价值 V(s)）
   - `_build_mlp_policy`：actor 网络（输出 7 个动作的概率分布 softmax）
   - 总损失 = actor_loss + 0.5·value_loss − 熵正则（鼓励探索，防止过早收敛）
   - `compute_advantage`：优势函数 = 实际回报 − critic 估值
   - 两个回放池：`policyReplayMemory` + `ReplayMemory`

**自测**：
1. IDQN/cDQN/cA2C 三者的递进逻辑一句话各是什么？
2. cA2C 的 actor 和 critic 各自的输入输出是什么？训练时怎么配合？
3. 熵正则项为什么要加？

### 阶段 5：训练脚本 + 论文精读（3~5 天）

1. `run/run_baseline_nopolicy.py`：真实数据下的"不作为"基线
2. `run/run_IDQN.py`（结构最清晰的训练循环，逐行读）：
   - 数据加载（pickle，`encoding='latin1'` 读 Python2 时代的旧文件）
   - `q_estimator` + `target_estimator` + `ReplayMemory(1e6, batch=3000)`
   - 外层 25 个 episode（每个 = 模拟一天 144 步），ε 从 0.5 线性衰减到 0.1
   - 内层每步：出动作 → `env.step(action, 2)` → 算 TD 目标存回放池 →
     抽 20 个 minibatch 更新 → episode 结束时复制参数到 target network、存模型
3. `run/run_cDQN.py`、`run/run_cA2C.py`：同构，对比差异即可
4. 论文精读：方法节逐段与代码对照（下表）

| 论文概念 | 代码位置 |
|---|---|
| City simulator | `simulator/envs.py` CityReal |
| Order matching (两轮广播) | `step_assign_order_broadcast_neighbor_reward_update` |
| State 定义 | `get_observation` + `stateProcessor.to_grid_states` |
| Action（6 邻居+不动）| `action_dim=7`、`valid_action_mask` |
| Reward（GMV）| 匹配订单价格之和 |
| 邻居广播奖励 | `weights_layers_neighbors = [1, e⁻¹, e⁻²]`、info |
| IDQN | `algorithm/IDQN.py` |
| contextual DQN | `algorithm/cDQN.py` |
| contextual A2C | `algorithm/cA2C.py` |
| ε-greedy 调度 | run 脚本 `epsilons = np.linspace(0.5, 0.1, 15)` |

> 注：`run/` 脚本依赖的原始数据文件不在仓库里（`dir_prefix` 指向作者服务器路径），
> 所以本地无法真正训练——读懂流程即可。想动手可把 `run_IDQN.py` 的数据加载部分
> 替换成 `tests/run_example.py` 里的人工数据，自己搭一个 mini 训练。

---

## 5. 需要补充的知识清单

### 5.1 Python / NumPy（边读边补，1~2 天搞定）
- NumPy：`np.array` 形状/索引、`np.where`、`np.sum(axis=)`、广播、
  `np.random.normal/choice/seed`（随机种子决定可复现性——本代码大量依赖）
- 面向对象：实例属性、组合（Node 持有 drivers/orders）、继承（DiagGaussianPd → Pd）
- `pickle`（Python 对象存盘/读盘）、`datetime`、`collections`
- 列表推导式、元组解包（`a, b, c = f()` 在本代码里到处都是）

### 5.2 机器学习最小集（概念级，2~3 天）
- 神经网络 = 层层矩阵乘法 + 激活函数（项目用到 `relu`/`elu`/`softmax`）
- 前向传播（算输出）/ 损失函数（量化差距）/ 反向传播（算梯度）/ 梯度下降（调参数）
- Adam 优化器（梯度下降的改良版，不用深究，知道是"调参神器"即可）
- "训练" = 循环执行 前向→算损失→反向→更新

### 5.3 强化学习核心（见阶段 2 的表格，最重要）
### 5.4 TensorFlow 1.x 读码最小集（半天）
- `placeholder`：图的输入槽位；`session.run(op, {placeholder: 数据})`：喂数据执行
- `variable_scope`：变量命名空间（区分 q_estimator / target_q 两套参数）
- 本项目用 `tf.compat.v1` 兼容层在 TF2 下跑 TF1 代码，遇到陌生 API 查
  "tf1 placeholder session" 即可，不需要系统学 TF2

---

## 6. 术语表

| 术语 | 含义 |
|---|---|
| GMV | 成交总额（这里是匹配订单的价格总和）|
| 接单率 / response rate | 成功匹配订单数 ÷ 总订单数 |
| 调度 / dispatch | 把空闲司机移动到别的网格（区别于"接单"）|
| 空驶 | 空车行驶（调度本身有空驶成本，论文靠奖励设计间接体现）|
| 供需盈余 | 某格司机数−订单数（正=车多，负=单多）|
| episode | 一轮完整交互（本项目=模拟一天 144 步）|
| TD 目标 | r + γ·max Q(s')，Q 学习的回归目标 |
| 经验回放 | 存历史样本随机抽训，打破时序相关性 |
| target network | 滞后更新的副网络，防止"追着自己的尾巴学" |
| 信用分配 | 多智能体下"谁的功劳/锅"问题，cDQN/cA2C 的动机 |
| 上下文 / context | 拼进网络输入的邻居信息（广播奖励、供需盈余）|
| 参数共享 | 所有智能体共用一个网络（cA2C），省参数且泛化好 |

---

## 7. 常见卡点 FAQ

**Q: 为什么 reward 是订单价格之和？**
A: 论文目标是最大化平台 GMV；接单率是次要指标。环境内建的匹配机制处理"接单"，
RL 只学"空车往哪放"能让后续匹配赚到更多钱。

**Q: `step` 里为什么先执行调度再匹配订单？**
A: 模拟真实时序：司机先被派往目标格（在途 1 步、不算空闲），到站后才能接单；
所以本步调度影响的是下一步及以后的匹配——这正是 RL 需要学"长期回报"的原因。

**Q: IDQN 每格一个智能体，参数是各一套吗？**
A: 是"各学各的值"，但代码实现上是**一个共享网络批量处理所有格子**的样本
（stateProcessor 把全城状态切成每格一条），效率和泛化都更好。

**Q: 本机没装 TensorFlow 能学吗？**
A: 能。simulator 部分纯 NumPy；algorithm 部分读代码+注释即可，
想跑通可 `pip install -r requirements.txt`（Python 3.10~3.12）。

**Q: 为什么 `run_example.py` 里 `l_max=2` 而 `run_IDQN.py` 里 `l_max=9`？**
A: l_max 是订单最长时长（步数）。人工小数据只有 1 步订单，真实数据最长 90 分钟=9 步。

**Q: `run_IDQN.py` 打印的 "Starting training Deep SARSA" 是 SARSA 算法吗？**
A: 不是，是作者遗留的文案错误，实际训练的就是 IDQN。代码注释里已标明。

**Q: `run_IDQN.py` 每轮打印的"接单率"为什么波动越来越小？**
A: 阅读陷阱：`order_response_rates` 列表**跨所有 episode 累积**（不像基线脚本每轮
重建），循环末尾的"本轮接单率"实际是对全部历史求平均，所以越往后越平滑。
字面理解代码时注意这一点。

---

## 8. 验收清单（全部打勾 = 毕业）

- [ ] 能画出 `env.step()` 的八步流水线
- [ ] 能向别人解释 reward 怎么来、info 里有什么
- [ ] 能说清 IDQN → cDQN → cA2C 各自解决了前者的什么问题
- [ ] 能指出 target network、经验回放、ε-greedy 在代码里的位置和作用
- [ ] 能读懂 `run_IDQN.py` 的训练循环并画出流程图
- [ ] 能对照论文方法节找到每个公式的代码实现

---

*生成于 2026-10-06，2026-10-07 更新；配合仓库内逐行中文注释使用。*
