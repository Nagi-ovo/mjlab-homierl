.. _walkthrough-task-homie-h1:

实例三：Homie —— 混合运动与干扰（Unitree G1 / H1）
============================================================

Homie 是一个综合性任务，结合了 **速度追踪 (Velocity)**、**蹲起 (Squat)**、
**站立 (Stand)** 以及 **上身随机干扰**。实现对齐 OpenHomie 参考实现
（``HomieRL/legged_gym``）的奖励集、命令采样与课程机制。

主要 task id：

- ``Mjlab-Homie-Unitree-G1``：标准版本（OpenHomie 原版机器人，29 dof，12 个下肢
  动作）。默认锁腰（waist_roll/pitch 由 PD 保持默认位姿、不参与上身扰动，对齐
  OpenHomie 焊死这两个关节的 27-dof URDF 与实机部署行为）。
- ``Mjlab-Homie-Unitree-H1``：H1 移植（19 dof，10 个下肢动作）。
- ``Mjlab-Homie-Unitree-H1-with_hands``：H1 额外挂载 Robotiq 2F85 夹爪
  （包含 policy-free 的夹爪随机动作与手部负载随机化）。

G1 变体（观测/动作接口与标准版完全一致，checkpoint 可互相加载）：

- ``Mjlab-Homie-Unitree-G1-free_waist``：三个腰关节全部参与上身扰动
  （原版训练分布的严格超集）。
- ``Mjlab-Homie-Unitree-G1-turn_mode``：可选扩展，walk 模式 1/3 的命令重采样
  转为原地转（vx = vy = 0、|wz| ≥ 0.3，约占全部环境的 1/6）。忠实版采样器对
  vx/vy/wz 联合采样，纯旋转命令测度为零，因此基线策略遇到纯 yaw 命令会站立
  不动；需要原地碎步转向时用此变体训练或续训。
- ``Mjlab-Homie-Unitree-G1-with_dex3`` / ``-with_inspire``：挂载 Unitree Dex3 /
  Inspire RH56 手模型（惯性附件 + 持物负载随机化）。基线任务的腕端负载随机化
  已覆盖这些手的质量，同一 checkpoint 裸腕/双手型通用；这两个变体主要用于
  带真实手几何的 play/评估。
- ``Mjlab-Homie-Unitree-G1-mjlab_gains``：mjlab 第一性原理增益的消融版本
  （仅仿真）。

核心设计思想：**缩小策略的动作空间，将其集中在下肢控制上，而将上身（以及可选
的夹爪）作为"随时间变化的平滑扰动"。**

任务注册
------------------------------------------------------------

路径：``src/mjlab_homierl/__init__.py``

通过 ``mjlab.tasks.registry.register_mjlab_task`` 注册上述 task id。
play env 是同一配置在 ``play=True`` 下的轻量化版本（剥离 critic 观测、
奖励与课程，上身扰动幅度直接拉满）。

三模式命令采样
------------------------------------------------------------

路径：``src/mjlab_homierl/mdp/velocity_command.py``

每 4 秒重采样一次命令，每个环境独立抽取三种互斥模式之一（OpenHomie 方案）：

- **squat**（p = 1/3）：twist 置零，随机采样相对高度目标；
- **walk**（p = 1/2）：随机采样 twist（x ∈ [-0.8, 1.2]、y ∈ [-0.5, 0.5]、
  yaw ∈ [-0.8, 0.8]），高度目标为站立高度；
- **stand**（p = 1/6）：twist 置零，高度目标为站立高度。

``twist`` 命令负责抽模式并通过 ``mode`` 属性暴露；``height`` 命令
（``RelativeHeightCommand``，相对最低足底站点的骨盆高度）与之耦合。
两个命令必须共享重采样周期，且 ``twist`` 在 commands 字典中必须位于
``height`` 之前。

高度范围按机器人身高设定：G1 站立 0.78 m、蹲至 0.28 m；H1 站立 0.98 m、
蹲至 0.4 m。一批"仅在站立高度激活"的奖励项（hip/ankle 偏差、feet_parallel、
stand_still 等）以命令高度是否接近站立高度做门控。

奖励集
------------------------------------------------------------

路径：``src/mjlab_homierl/homie_env_cfg.py``（权重）、
``src/mjlab_homierl/mdp/rewards.py``（实现）

奖励项与权重完整对齐 OpenHomie 的 G1 配置：x/y 速度追踪拆分（1.5 / 1.0）、
yaw 追踪（2.0，σ²=0.25）、高度追踪 ``exp(-4|err|)``（2.0）、hip/ankle 偏差
（-0.2 / -0.5）、膝驱动蹲起（-0.75）、全套关节正则（力矩、功率、速度、
加速度、软限位）、足部项（air time、no-fly、clearance、slip、stumble、
接触力、接触动量、足底水平、双足平行、横向间距）等。

与 OpenHomie 的有意差异：

- IsaacGym 的躯干接触终止改为 **接触惩罚 + 躯干接触终止（G1）**；
  另有 self-collision 与髋/膝触地惩罚。
- 倾倒终止阈值与原版一致（``asin(0.8) ≈ 53°``）。

上身扰动与课程
------------------------------------------------------------

路径：``src/mjlab_homierl/mdp/actions.py``、``mdp/curriculums.py``

``UpperBodyPoseAction`` 贡献 0 维策略动作。每 1 秒（全局事件）为所有环境
重采样上身目标位姿：目标幅度来自课程比率的截断指数变换（课程早期强烈偏向
小幅动作），方向在关节上下硬限位之间掷硬币，因此幅度与各关节行程成正比；
目标在一个采样周期内线性插值到达。

课程推进对齐 OpenHomie：仅当 ``common_step_counter`` 是最大回合步数的整数倍
时检查一次，若 x 方向速度追踪的回合平均原始奖励 ≥ 0.8，全局比率 +0.05。

域随机化
------------------------------------------------------------

全部使用 mjlab 原生 ``dr.*`` 事件：PD 增益 ×[0.9, 1.1]（每次 reset）、
连杆质量 ×[0.8, 1.2]、躯干负载 +[-2, 5] kg、质心偏移、编码器偏置、足底摩擦、
每 4 秒全局水平推撞（Δv ≤ 0.5 m/s）、重置时关节位姿与根速度随机。
OpenHomie 的逐步力矩注入在 mjlab 中没有对应机制，由 PD 增益随机化与
编码器偏置近似。

HIM-PPO
------------------------------------------------------------

路径：``src/mjlab_homierl/rl/himppo/``

超参数与网络结构（actor/critic 隐层 512-256-256、估计器 latent 32、
prototype 64、sinkhorn 对比学习）对齐 OpenHomie。对称性增强的左右镜像映射
从关节名自动推导（``left_*``/``right_*`` 配对，名字含 ``yaw``/``roll``
的关节变号），因此 G1 与 H1 共用同一实现。

已知差异：终止步的估计器 next critic 观测为重置后的观测
（mjlab 在 reset 之后才计算观测），OpenHomie 会替换为终止前的观测。
