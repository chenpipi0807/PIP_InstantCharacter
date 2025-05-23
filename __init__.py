"""
PIP-InstantCharacter for ComfyUI
基于腾讯InstantCharacter项目 (https://github.com/Tencent/InstantCharacter)
集成到FLUX工作流的角色保留生成节点
"""

import os
import sys
from pathlib import Path

# 定义导出的模块
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 添加模型目录
import folder_paths

# 添加当前路径到sys.path，确保模块能被正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 注册模型路径
models_dir = os.path.join(folder_paths.models_dir, "ipadapter")
if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)
folder_paths.folder_names_and_paths["ipadapter"] = ([models_dir], folder_paths.supported_pt_extensions)

# 从nodes子模块导入所有节点类和映射
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# NODE_DISPLAY_NAME_MAPPINGS已从.nodes导入

# 打印节点映射信息，帮助调试
print(f"[PIP-InstantCharacter] 注册节点映射: {list(NODE_CLASS_MAPPINGS.keys())}")
print(f"[PIP-InstantCharacter] 成功加载InstantCharacter主节点: PIPInstantCharacter")


