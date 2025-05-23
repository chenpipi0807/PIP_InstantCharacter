"""PIP-InstantCharacter 节点初始化文件
确保节点正确注册到ComfyUI框架"""

# 从原有节点文件导入NODE映射
from .pip_instantcharacter import NODE_CLASS_MAPPINGS as PIP_NODE_CLASS_MAPPINGS
from .pip_instantcharacter import NODE_DISPLAY_NAME_MAPPINGS as PIP_NODE_DISPLAY_NAME_MAPPINGS

# 从新的原版节点导入NODE映射
try:
    from .original_extractor_node import NODE_CLASS_MAPPINGS as ORIGINAL_NODE_CLASS_MAPPINGS
    from .original_extractor_node import NODE_DISPLAY_NAME_MAPPINGS as ORIGINAL_NODE_DISPLAY_NAME_MAPPINGS
    has_original_nodes = True
    print("[PIP-InstantCharacter] 原版特征提取器节点已加载")
except ImportError:
    has_original_nodes = False
    print("[PIP-InstantCharacter] 警告: 原版特征提取器节点加载失败")
    ORIGINAL_NODE_CLASS_MAPPINGS = {}
    ORIGINAL_NODE_DISPLAY_NAME_MAPPINGS = {}
    
# 特征适配器节点已移除，现在PIP InstantCharacter应用节点可以直接接收特征输入

# 合并NODE映射
NODE_CLASS_MAPPINGS = {**PIP_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**PIP_NODE_DISPLAY_NAME_MAPPINGS}

# 如果原版节点加载成功，添加到映射中
if has_original_nodes:
    NODE_CLASS_MAPPINGS.update(ORIGINAL_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(ORIGINAL_NODE_DISPLAY_NAME_MAPPINGS)

# 导出NODE映射以供主__init__.py使用
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
