"""
用于调试节点加载问题的脚本
将打印导入路径和加载状态
"""
import os
import sys
import importlib

# 打印当前Python路径
print("\n=== Python Path ===")
for p in sys.path:
    print(f"路径: {p}")

# 尝试导入节点
print("\n=== 尝试导入节点 ===")
try:
    # 设置当前路径为脚本所在目录的父目录
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    print(f"添加路径: {parent_dir}")
    
    # 尝试导入
    print("导入主模块...")
    import PIP_InstantCharacter
    print("主模块导入成功!")
except Exception as e:
    print(f"主模块导入失败: {e}")

try:
    print("\n导入nodes子模块...")
    from PIP_InstantCharacter.nodes import NODE_CLASS_MAPPINGS
    print(f"节点映射导入成功! 找到 {len(NODE_CLASS_MAPPINGS)} 个节点:")
    for name, cls in NODE_CLASS_MAPPINGS.items():
        print(f" - {name}: {cls.__name__}")
except Exception as e:
    print(f"节点映射导入失败: {e}")

try:
    print("\n直接导入节点模块...")
    from PIP_InstantCharacter.nodes.pip_instantcharacter import PIPApplyInstantCharacter
    print(f"节点类导入成功: {PIPApplyInstantCharacter.__name__}")
except Exception as e:
    print(f"节点类导入失败: {e}")

print("\n=== 模块检查 ===")
for module_name in list(sys.modules.keys()):
    if "pip" in module_name.lower() or "instantcharacter" in module_name.lower():
        print(f"已加载模块: {module_name}")
