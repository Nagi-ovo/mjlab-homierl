# HOMIE+ 训练计划（草案 v1，2026-07-03）

目标：给 mjlab 原生 G1 HOMIE 下肢策略增加**躯干俯仰命令**（+ 确认深蹲高度覆盖），
使其以约 20% 的成本覆盖 AMO backend 对 BiGym 2.0 任务集的主要价值（弯腰捡低处
物体），落地后按 BiGym 2.0 方案退役两个进口 ckpt（OpenHomie-G1 / AMO-G1）。

上游依据：`/home/jz5725/Projects/CQN-AS-G1/bigym2_architecture_plan.md`
（Roadmap 并行线、§7-4 AMO 定位、§8.4 "AMO retires the day our own HOMIE+
lands"）。

**范围：仅 G1。** H1 的 torso 关节是纯 yaw，机械上没有腰部俯仰——H1 若要俯身
只能走"全身髋部前倾"的另一种设计，不在本计划内。

---

## 1. 能力目标

| 能力 | 现状（HOMIE, deploy-gains run） | HOMIE+ 目标 |
|---|---|---|
| 速度跟踪 | vx,vy,wz 命令 | 不回退（vx 单步原始奖励 ≥ 基线） |
| 蹲起高度 | h ∈ (0.28, 0.78)，已覆盖方案要求的 0.35–0.75 | 维持 |
| 躯干俯仰 | 无（waist_pitch 是随机扰动的一部分） | pitch 命令可控，站立/蹲姿下可保持前倾 |
| 上肢扰动鲁棒 | waist(3)+arms(14) 随机扰动课程 | waist_pitch 移出扰动集，其余不变 |

## 2. 设计

### 2.1 命令空间

`(vx, vy, wz, h)` → `(vx, vy, wz, h, torso_pitch)`。

- pitch 命令语义：**waist_pitch 关节角目标**（rad，正=前倾）。选关节角而非躯干
  link 世界姿态：与 BiGym 插件的 command_spec 直接对应、可由上层 IK/遥操作直
  接给出、无需姿态估计回路。
- 范围建议：`(-0.2, +0.45)`（waist_pitch 硬限位 ±0.52，软限位 ~±0.47）。
- 采样耦合（三模式扩展，待实现时定稿的默认方案）：
  - walk 模式（1/2）：pitch = 0（70%）或小幅 U(-0.15, 0.25)（30%，覆盖"边走边
    低头够东西"）；
  - squat 模式（1/3）：h 与 pitch 独立采样，pitch = 0（50%）或 U(-0.2, 0.45)
    （50%）——深蹲+前倾是捡地面物体的主工况；
  - stand 模式（1/6）：pitch = 0。

### 2.2 waist_pitch 的归属（保持 HOMIE 哲学：策略不控上身）

策略动作空间**不变**（12 条腿）。waist_pitch 由命令**直驱**：新增一个
policy-free 动作项（参照 `UpperBodyPoseAction` 的结构），位置目标 =
default + pitch_cmd，带限速平滑插值；策略的任务是在腰被命令前倾时保住平衡
（重心前移由髋/踝代偿）。

**腰关节归属裁决（2026-07-05，基于原版 27-dof 事实）**：原版 URDF 把
waist_roll/pitch 直接 `type="fixed"` 焊死（只有 waist_yaw 可动，27=12+1+14），
即原版扰动永远不会让躯干前后/左右倾——我们移植成 29-dof 并把三个腰关节都放进
了扰动集，是比原版更狠的超集（当前 v3 即此配置；锁死=扰动分布内的子集，故
兼容）。HOMIE+ 轮起：waist_pitch → 命令直驱（本计划主体）；**waist_roll →
移出扰动集、锁默认**（原版本就不动它，BiGym 也锁腰）；waist_yaw → 留在扰动
集（原版如此）。

对应 BiGym 插件声明的变化：controlled_joints 仍为 12 腿 + waist_pitch（命令直
驱），command_spec 从 4 维变 5 维。

### 2.3 观测

一步观测的命令段 4 → 5 维（one-step dim 80 → 81，actor 486）。**观测布局改变
= 必然是新策略、新一轮训练**；布局经 ONNX metadata（`num_one_step_obs`、
命令语义字段）自举，BiGym 插件零硬编码。镜像映射无需改逻辑（pitch 是左右对称
量，sign=+1）。

### 2.4 奖励调整（俯仰与现有项的冲突点）

1. **orientation 惩罚必须改为"相对命令姿态"**：现在罚 projected_gravity_xy²
   （偏离直立），命令前倾 0.45 rad 会被自己的奖励打架。改为罚"实际躯干俯仰与
   命令俯仰的偏差"+ 保留 roll 直立惩罚。
2. **新增 pitch 跟踪奖励**：`exp(-|waist_pitch - cmd| * k)`，权重 ~1.0（具体
   k/权重训练时标定；跟踪对象是关节角，简单可靠）。
3. **deviation_hip_joint 门控加 pitch 条件**：前倾时髋必须偏离默认位姿来配平
   重心，门控从 `height ≥ 站立` 改为 `height ≥ 站立 AND |pitch_cmd| < ε`。
4. `fell_over`（53°）与 pitch 命令上限 0.45 rad（~26°）留有余量，暂不动；训练
   中若前倾工况误终止率异常再放宽。
5. 其余（feet 系列、stand_still、正则项）不动。

### 2.5 Dex3 手部质量（已落地，2026-07-03，commit 9548034）

已实现为独立变体 `Mjlab-Homie-Unitree-G1-with_dex3`：真实 Dex3 模型挂载
（从 AMO g1_29dof_with_dex3 提取，手指焊死开手位姿、惯量精确、仅视觉 geom，
~0.53 kg/手）+ 持物 payload DR（0–1 kg/手）。观测/动作接口与基础任务相同，
checkpoint 双向兼容（基础任务的 ckpt 可直接在 with_dex3 play 环境做分布外检
查）。**HOMIE+ 训练时直接用 dex3=True 的配置组合即可**（HOMIE+ 任务注册时叠加
该 flag，或提供 `-plus-with_dex3` 变体）。

### 2.5b 手型无关化（2026-07-04 定稿，导师要求支持 Inspire RH56DFX）

下肢策略不按手型分训。手对下肢策略只体现为腕部末端质量/惯量；OpenHomie 原版
就在裸腕模型上做 hand_payload DR（[-0.1, 0.3] kg）。方案：

- **训练（HOMIE+ 轮）**：基线任务加腕部负载 DR `hand_payload`，U(0, 1.5) kg/手
  加在 wrist_yaw_link（0=裸腕兼容；1.5 覆盖 Dex3 0.53 kg、Inspire RH56DFX
  ~0.54 kg、以及手持 0.5–1 kg 物体）。一个策略覆盖所有手型。
- **评测/演示**：真实挂载变体仅作 play/BiGym 保真（接口兼容，任意 ckpt 直载）。
  with_dex3 已有；**with_inspire 已落地（commit cd5e671）**：几何/安装变换取自
  unitree_ros 官方 G1+Inspire URDF（BSD-3），厂商惯量（仅 0.19 kg/手，占位值）
  重标定到 RH56DFX 规格 0.54 kg/手；`hands={'dex3','inspire'}` 统一参数。
- 不做固定单手型训练（分布收缩、弄脏基线对照）；不做额外 fake 挂载机制
  （dr.body_mass 点质量即是）。
- 验证实验（零成本，已排）：本轮跑完后裸腕 ckpt 在 with_dex3 play 环境做分布
  外检查——掉分明显 ⇒ 腕部 DR 必要性实锤；几乎不掉 ⇒ DR 是保险。

### 2.6 任务隔离

HOMIE+ 注册为独立任务 ID（如 `Mjlab-Homie-Unitree-G1-plus`）：命令 4→5 维改变
观测布局，checkpoint 与基础任务**不**兼容，是接口级分叉（比 with_dex3 的
"训练分布变体"隔离等级更高）。

## 3. 训练与验收

配置：与当前 deploy-gains run 相同（4096 envs、HIM-PPO、30k iter、deploy 增益
+ 0.25 action scale、Newton、~14h @ RTX 5090）。

验收标准：

**sim 侧（本 repo）**
- pitch 跟踪：站立与蹲姿下，命令 (0, 0.45) 范围内稳态误差 < 0.05 rad；
- 速度/高度跟踪不回退：vx 单步原始奖励 ≥ 0.75、高度误差 ≤ 5 cm（对齐当前基线）；
- 摔倒率与基线同量级（fell_over ~0.1/iter@4096）。

**BiGym 侧（验收即接入）**
- ONNX metadata 含 5 维命令语义 + pitch 范围，插件 command_spec 自举成功；
- 行走探针 ≥ OpenHomie 基线（1.17 m/6 s；mjlab-G1 上一版 1.86 m）；
- settle ~40 步量级（mjlab 原生特征）；
- 低位够物可行性 demo：站立前倾 + 深蹲前倾各一段遥操作，验证 AMO 替代成立；
- snapshot roundtrip 逐位重放（Newton）。

## 4. 实施步骤与工作量

| 步骤 | 内容 | 量 |
|---|---|---|
| A | 命令项扩展（pitch 采样+模式耦合）、waist_pitch 直驱动作项、奖励调整 §2.4、观测 5 维、测试+冒烟 | 0.5–1 天 |
| B | Dex3 腕部负载 DR | 几行 |
| C | 4096×30k 训练 + sim 验收 | ~14 h GPU |
| D | 导出 ONNX（全量 metadata）→ BiGym 插件 command_spec 升级 + 四件套重验（行走/settle/roundtrip/低位够物） | BiGym 侧半天 |

前置条件：当前 deploy-gains G1 run 跑完并通过验收（它是"原味 HOMIE 复现"里程
碑，也是 BiGym Phase 0.5 重采 G1 demo 的依赖项）；HOMIE+ 与之并行不冲突，但建
议先看完基线的步态目检结论（拖步问题若需调 feet_air_time/no_fly 权重，改动应
合入 HOMIE+ 这一轮，避免多训一次）。

## 5. 待定项

1. pitch 采样耦合比例（§2.1 给了默认方案，训练前定稿）；
2. pitch 跟踪奖励的 k 与权重（首轮训练标定）；
3. ~~拖步问题的奖励调整是否合入本轮（等基线 play 目检结论）~~ **已裁决
   （2026-07-04，见 §6）：no_fly/contact_momentum 符号 bug 已修（812ff15），
   必须合入本轮重训；self_collision 处理待用户拍板**；
4. waist_yaw/roll 是否也从扰动集移除、锁默认（BiGym 里 HOMIE 赛道腰本就锁死；
   保留扰动=训练分布超集，默认保留）；
5. H1 版 HOMIE+（机械上无腰 pitch，需另行设计，暂不排期）。

## 5b. HIM-PPO 对账遗留项（2026-07-05 审计）

- **终止转移的 estimator 目标**：原版用重置前的终止观测（runner:144），我们
  之前喂的是重置后观测。**已修复对齐（commit 9cf01bf）**：RecorderTerm 在
  pre-reset 钩子重建单步 critic 观测 → extras → process_env_step 替换。当前
  v3 run 不含此修复（影响 ~0.1–0.5% estimator 样本，量级为标签噪声）；下轮
  起生效。
- **max_iterations**：原版 G1 配置 100k，我们 30k。非 bug，预算选择。裁决：
  若 v3 30k 验收时深蹲/转向未收敛，从 checkpoint 续训延长属于"配方内"操作
  （rsl-rl resume），优先于任何奖励调参。

## 6. 基线（deploy-gains run）验收结论（2026-07-04）

30k iter 跑完，model_29999 定量 play 探针（64 envs，钉死命令）：

| 项 | 结果 |
|---|---|
| 站立/行走高度跟踪 | ✅ 误差 3 mm、时序 std 2–6 mm，非常稳 |
| 行走步态 | ⚠️ 真实交替步但偏拖：0.8 m/s 双支撑 38%、摆动仅 0.13 s |
| vx 跟踪 | ⚠️ 0.8 m/s 误差 0.14；低速 0.4 与高速 1.2 误差 ~0.27 |
| 原地转向 | ❌ 完全不转（wz 误差≈命令值，98% 双支撑站着不动） |
| 下蹲 | ❌ 墙在 rel-h≈0.67（膝 1.2 rad 封顶），命令更深反而站更高 |
| 腾空 | 双脚同时离地 ≤0.4%，maxfly 0.06 s——无需双腾空惩罚 |
| 突变切换 | walk 1.2→squat→walk 0 摔倒 |

根因（已定位）：
- **no_fly / contact_momentum 全程死项**：mjlab 足-地接触力 z 为负，代码用
  `>阈值` 判接触永远 False。no_fly 只剩零命令满分条款 ⇒ 行走/转向期间无单支撑
  激励 ⇒ 拖步 + 拒绝原地转向。已修（commit 812ff15，改用幅值判定）。
- **self_collisions(-1.0) 是移植新增项且严重误触发**：默认站姿下 25–30% 步都有
  腕↔髋接触（腕 kp 5–10 太软，手搭在髋上就不走了），下蹲时加剧（出现腕↔hip_yaw
  对）⇒ 反下蹲梯度。OpenHomie G1 原版 IsaacGym `self_collision=1`（=禁用自碰撞）
  且奖励表无此项。待定：忠实复现（禁手臂↔下肢碰撞对+删项）vs 保留但加力阈值/降权。
- hip_knee_contact(-1.0) 同为新增项但全程 0（髋膝从不触地），无害。

⇒ 基线未过验收（下蹲、转向）；HOMIE+ 训练轮必须携带上述修复，验收增加
squat 深度（0.35 可达）与原地 wz 跟踪两条硬性门槛。

**2026-07-04 裁决与重训**：self_collisions 惩罚项已从 G1 任务移除（commit
9e3d680，物理自碰撞保留，仅删奖励项；H1 不动）。"先站立再下蹲"顺序约束
**不做**：突变切换是原版行为且有部署价值（探针 0 摔倒），下蹲失败根因是
奖励 bug 不是切换设计；顺序约束反而会让策略对直接切换变脆。这一轮通过
验收后才进入 HOMIE+（本计划前置条件恢复成立）。

**重训 v2（2026-07-04 晚）**：首次重启（`g1-homie-deploy-fixed-rewards-4096`）
在 ~1.3k iter 时被主动放弃——发现移植遗漏了 OpenHomie 原版的腕端负载 DR
（hand_payload [-0.1,0.3] kg），且用户确认要手型无关基线。补上
`hand_payload` U(0, 1.5) kg/手（腕端独立采样，commit 219a90c，§2.5b 的训练侧
就此生效）后重启 `g1-homie-deploy-fix-handdr-4096`。

**v3 → v4（当前，2026-07-05）**：v3 在 14k/30k 被用户裁决停止，换全量对齐版
重训。v4 新增（相对 v3）：终止转移 estimator 目标对齐（9cf01bf）+ **锁腰默认**
（615d4c3，waist_roll/pitch 移出扰动集、PD 锁默认位，= OpenHomie 27-dof 有效
分布；waist_yaw 仍扰动；`-free_waist` 变体保留超集模式，ckpt 双向兼容）。
run `g1-homie-deploy-v4-aligned-4096`，tmux `g1-homie-fix`。v4 = 迄今与原版
对齐度最高的配方（奖励修复 + DR 补齐 + 终止对齐 + 27-dof 有效分布）。

**v2 中途验证成功 → v3（2026-07-04 17:23）**：v2 的 model_2600（仅 8.7%
训练量）探针已全面超越坏基线跑满 30k 的终值：步行单支撑 84%（旧 38%）、摆动
0.227s（旧 0.13）、vx 误差 0.037@0.8（旧 0.14）、原地转向出现（wz 误差 0.60，
旧 0.78=完全不转）、下蹲深度 0.63 越过旧终值墙 0.70 且持续下探。奖励修复
判定生效。随即杀掉 v2，改用全量最新配方重启：**run
`2026-07-04_17-23-18_g1-homie-deploy-v3-fulldr-4096`**（commit 038fd45 =
奖励修复 + 腕端 DR + 动作延迟 DR + payload/摩擦/编码器/CoM 范围补齐），
tmux `g1-homie-fix`，4096 envs、30k iter。这是基线验收的正式候选。
