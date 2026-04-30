# Models Directory Structure

## 目录说明

```
models/
├── encoders/                    # 编码器模块（新）
│   └── base_encoder.py
│
├── inner_models/                # Inner Models实际代码（新）
│   ├── transformers/           # Transformer模型
│   ├── unets/                  # U-Net模型  
│   ├── mlps/                   # MLP模型
│   ├── parameter_space/        # 参数空间模型
│   └── decoders/               # 解码器模型
│
├── base_model.py               # 扩散模型基类
├── base_inner_model.py         # Inner model基类
├── diffusion_model.py          # 标准扩散模型
├── motif_diffusion_model.py    # MOTIF扩散模型
├── nll_model.py                # NLL模型
├── gmm_model.py                # GMM模型
├── scaling.py                  # 噪声缩放
├── robomimic_lstm_gmm.py       # Robomimic LSTM GMM
│
└── *_inner_model.py (13个)     # 向后兼容层（勿删）
    ├── causal_transformer_inner_model.py
    ├── conditional_unet1d_inner_model.py
    ├── lstm_gmm_inner_model.py
    ├── mlp_inner_model.py
    ├── motif_transformer_inner_model.py
    ├── nll_decoder_inner_model.py
    ├── parameter_space_mlp_inner_model.py
    ├── parameter_space_residual_mlp_inner_model.py
    ├── prodmp_causal_transformer_inner_model.py
    ├── prodmp_mlp_inner_model.py
    ├── prodmp_residual_mlp_inner_model.py
    └── prodmp_resnet1d_inner_model.py
```

## 文件说明

### 核心模型文件（保留）
这些是实际的模型实现，包含业务逻辑：
- `base_model.py` - 所有扩散模型的基类
- `diffusion_model.py` - 标准扩散模型实现
- `motif_diffusion_model.py` - MOTIF扩散模型（带速度监督）
- `nll_model.py` - 负对数似然模型
- `gmm_model.py` - 高斯混合模型
- `scaling.py` - Karras等噪声缩放方法
- `robomimic_lstm_gmm.py` - Robomimic基线模型

### 向后兼容层（13个 *_inner_model.py）
⚠️ **重要：请勿删除这些文件！**

这些文件是向后兼容层，每个只有7行代码：
```python
# Backward compatibility: Import from new location
from movement_primitive_diffusion.models.inner_models.xxx import YYY
__all__ = ['YYY']
```

**为什么需要保留？**
1. **Hydra配置依赖**：多个YAML配置文件引用这些路径
2. **向后兼容**：旧代码可以无缝工作
3. **零成本**：每个文件只有7行，不影响性能

**实际代码位置**：
- 真正的实现在 `inner_models/` 子目录中
- 这些文件只是重定向导入

### 新增目录结构

#### `encoders/` - 编码器模块
重构后将编码器独立为模块，更清晰的架构。

#### `inner_models/` - Inner Models分类组织
按模型类型组织，便于管理和扩展：
- `transformers/` - Transformer架构
- `unets/` - U-Net架构
- `mlps/` - MLP架构
- `parameter_space/` - 参数空间模型
- `decoders/` - 解码器

## 使用建议

### 推荐导入方式（新代码）
```python
# 使用新的组织化路径
from movement_primitive_diffusion.models.inner_models.transformers import MOTIFTransformerInnerModel
from movement_primitive_diffusion.models.encoders import Encoder
```

### 兼容导入方式（旧代码）
```python
# 仍然工作，但不推荐
from movement_primitive_diffusion.models.motif_transformer_inner_model import MOTIFTransformerInnerModel
```

## 维护说明

- ✅ 可以修改 `inner_models/` 中的实际实现
- ✅ 可以修改核心模型文件
- ⚠️ 不要删除 `*_inner_model.py` 兼容层文件
- ⚠️ 不要修改兼容层文件内容（除非要废弃向后兼容）

## 清理策略

如果未来确定不需要向后兼容，可以：
1. 更新所有Hydra配置中的 `_target_` 路径
2. 删除这13个兼容层文件
3. 目前**不推荐**这样做，保持兼容性更安全
