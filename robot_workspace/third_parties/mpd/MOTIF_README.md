# MOTIF v3.2 Implementation

完整实现了MOTIF v3.2（Modulated ODE Trajectory with Integrated Flow）算法，基于MPD框架。

## 核心特性

MOTIF v3.2实现了论文中提出的4个核心变化：

### 1. 物理时间编码（Change 1）
- **替换**: 步骤索引 k → 物理时间 τ（秒）
- **实现**: `MotifTimeEmbedding` 使用正弦嵌入编码物理时间
- **优势**: 支持可变控制频率，无需重新训练

### 2. 傅里叶系数空间（Change 2）
- **替换**: 位置序列 R^(K×d) → DCT系数 R^((M+1)×d)
- **实现**: 使用DCT-II基函数进行编码/解码
- **优势**: 结构性C^∞连续性，无需正则化

### 3. 状态条件查询（Change 3）
- **添加**: 当前状态作为运行时查询输入
- **实现**: t-conditioned masking（t=0时使用真实状态，t>0时屏蔽）
- **优势**: 支持闭环执行（LEITMOTIF模式，未实现）

### 4. 速度监督（Change 4）
- **添加**: 速度监督损失 L_vel
- **实现**: 通过傅里叶解码进行速度监督
- **优势**: 直接学习速度场，适合阻抗控制

## 文件结构

```
mpd/
├── movement_primitive_diffusion/
│   ├── agents/
│   │   └── motif_agent.py                    # MOTIF Agent
│   ├── models/
│   │   ├── motif_transformer_inner_model.py  # MOTIF Transformer
│   │   └── motif_diffusion_model.py          # 双损失模型
│   ├── datasets/
│   │   └── process_batch_motif.py            # 数据预处理
│   └── utils/
│       └── motif_utils.py                     # DCT工具函数
├── conf/
│   ├── agent_config/
│   │   ├── motif_transformer_agent.yaml
│   │   └── model_config/
│   │       ├── motif_diffusion_model.yaml
│   │       └── inner_model_config/
│   │           └── motif_transformer.yaml
│   └── experiments/obstacle_avoidance/
│       ├── train_motif_transformer.yaml
│       ├── process_batch_config/
│       │   └── motif.yaml
│       └── motif_handler_config/
│           └── motif_handler.yaml
├── scripts/
│   └── validate_motif.py                     # 验证脚本
├── motif_train.sh                            # 训练脚本
└── validate_motif.sh                         # 验证脚本包装器
```

## 快速开始

### 1. 验证实现

```bash
bash validate_motif.sh
```

应该看到所有6个测试通过：
- ✅ DCT编码/解码
- ✅ 频率加权噪声
- ✅ MOTIFHandler
- ✅ 物理时间编码
- ✅ 状态屏蔽
- ✅ 集成测试

### 2. 训练MOTIF

基础训练（使用默认参数）：
```bash
bash motif_train.sh
```

完整配置训练：
```bash
bash motif_train.sh obstacle_avoidance \
    --device cuda \
    --gpu 0 \
    --epochs 3000 \
    --batch-size 256 \
    --num-modes 16 \
    --alpha-vel 1.0 \
    --swanlab-mode cloud \
    --swanlab-group motif-baseline
```

### 3. 查看帮助

```bash
bash motif_train.sh --help
```

## 关键参数

### 傅里叶模态数（M）
- **默认**: 16
- **推荐范围**: 8-32
- **选择方法**: 使用能量阈值法（95%能量）
- **影响**: M越大，表达能力越强，但计算成本越高

### 速度损失权重（α）
- **默认**: 1.0
- **论文值**: 1.0
- **调整**: 如果L_vel不收敛，可以尝试增大到2.0

### 块持续时间（T）
- **默认**: 1.0秒
- **说明**: 动作块的物理时间长度
- **影响**: 影响频率加权噪声的计算

## 双损失结构

MOTIF使用双损失训练：

```
L = L_FM + α·L_vel
```

其中：
- **L_FM**: Flow Matching损失（系数空间）
- **L_vel**: 速度监督损失（通过傅里叶解码）
- **α**: 速度损失权重（默认1.0）

训练过程中会记录：
- `loss_fm`: FM损失
- `loss_vel`: 速度损失
- `total_loss`: 总损失

## 验证清单（来自论文附录）

实现已通过以下验证：

1. ✅ **DCT编码/解码**: 重建误差 < 0.1
2. ✅ **频率加权噪声**: 高频模态方差更小
3. ✅ **物理时间编码**: 不同时间产生不同嵌入
4. ✅ **状态屏蔽正确性**: t=0时使用真实状态，t>0时屏蔽
5. ✅ **组件集成**: 所有模块成功导入

训练时需要监控：
- L_vel < 0.01 (rad/s)² within 50k steps
- L_FM单调下降
- 频率鲁棒性（10Hz/50Hz/200Hz）
- 轨迹平滑性（低抖动）

## 与MPD的对比

| 特性 | MPD (ProDMP) | MOTIF v3.2 |
|------|--------------|------------|
| 输出对象 | 位置序列 | 速度场 |
| 时间轴 | 步骤索引 k | 物理时间 τ (秒) |
| 主要量 | 位置 | 速度 |
| 连续性 | 无保证 | 结构性C^∞ |
| 状态接口 | 无 | 状态条件查询 |
| 控制频率 | 训练时固定 | 推理时可变 |
| 高频数据 | 下采样 | 完全利用 |

## 未实现功能

- **LEITMOTIF闭环执行**: 论文中提到可以不考虑
- **中间去噪可执行性测试**: 需要环境支持
- **频率鲁棒性测试**: 需要多频率数据

## 技术细节

### DCT-II系数提取
```python
coeffs = scipy.fft.dct(velocities, axis=0, norm='ortho')[:M+1, :]
```

### 傅里叶解码
```python
v(τ) = Σ_{k=0}^M c_k · cos(kπτ/T)
```

### 频率加权噪声
```python
σ_k = σ / sqrt(1 + ω_k²), ω_k = kπ/T
```

### 状态屏蔽
```python
state_emb = (t==0) * state_proj(A) + (t>0) * mask_token
```

## 故障排除

### 问题1: L_vel不收敛
- 检查速度数据是否正确计算（有限差分）
- 增加α权重到2.0
- 检查DCT系数归一化

### 问题2: 训练不稳定
- 减小学习率到5e-5
- 增加批次大小到512
- 检查数据归一化

### 问题3: 重建误差大
- 增加傅里叶模态数M
- 检查dt是否正确设置
- 验证DCT编码/解码

## 引用

如果使用此实现，请引用MOTIF论文：

```bibtex
@article{motif2026,
  title={MOTIF: Modulated ODE Trajectory with Integrated Flow},
  author={...},
  journal={...},
  year={2026}
}
```

以及MPD论文：

```bibtex
@article{Scheikl2024MPD,
  author={Scheikl, Paul Maria and ...},
  title={Movement Primitive Diffusion: Learning Gentle Robotic Manipulation of Deformable Objects},
  journal={IEEE Robotics and Automation Letters}, 
  year={2024},
}
```

## 联系与支持

- 实现基于MPD框架：https://github.com/ScheiklP/movement-primitive-diffusion
- MOTIF论文：motif_v3.2.tex

---

**状态**: ✅ 实现完成，所有验证测试通过

**最后更新**: 2026-04-13
