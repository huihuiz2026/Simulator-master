# TensorFlow 2 兼容层：算法包（IDQN/cDQN/cA2C）统一通过 `from .tf_compat import tf` 引用 TensorFlow，
# 借助该兼容层让按 TF1"先建计算图、再开 session 运行"风格编写的算法代码能直接跑在 TF2 上
"""TensorFlow 2 compatibility for the original session-based algorithms."""

# 导入 TF1 兼容 API（tensorflow.compat.v1）并简写为 tf：
# 它提供 placeholder（预分配的输入槽位，session 运行时喂数据）、Session（会话）、variable_scope 等旧式接口
import tensorflow.compat.v1 as tf

# 关闭 TF2 的默认行为（如 Eager Execution 即时执行模式），恢复 TF1 的
# 延迟建图 + session 运行模式；不调用此行，旧式代码在 TF2 下会直接报错
tf.disable_v2_behavior()
