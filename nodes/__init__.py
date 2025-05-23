"""PIP-InstantCharacter 节点初始化文件
确保节点正确注册到ComfyUI框架"""

# 从节点文件导入所有NODE相关映射
from .pip_instantcharacter import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 导出NODE映射以供主__init__.py使用
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
