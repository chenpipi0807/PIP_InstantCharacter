"""
InstantCharacter IP适配器
处理InstantCharacter IP适配器的加载和应用
"""

import os
import torch
import torch.nn as nn

class InstantCharacterIPAdapter:
    """处理InstantCharacter IP适配器的加载和应用"""
    
    def __init__(self, adapter_path, device="cuda"):
        """
        初始化IP适配器
        
        Args:
            adapter_path: IP适配器模型路径
            device: 使用设备
        """
        self.device = device
        self.adapter_path = adapter_path
        
        # 检查IP适配器文件是否存在
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"InstantCharacter IP适配器文件不存在: {adapter_path}")
        
        print(f"[PIP-InstantCharacter] 加载IP适配器: {adapter_path}")
        self.load_adapter(adapter_path)
    
    def load_adapter(self, adapter_path):
        """加载IP适配器权重"""
        try:
            self.state_dict = torch.load(adapter_path, map_location="cpu")
            
            # 提取关键权重并加载到相应组件
            self.image_proj = self._extract_component(self.state_dict, "subject_image_proj")
            self.image_proj_norm = self._extract_component(self.state_dict, "subject_image_proj_norm")
            self.image_proj_2 = self._extract_component(self.state_dict, "subject_image_proj_2")
            self.image_proj_norm_2 = self._extract_component(self.state_dict, "subject_image_proj_norm_2")
            self.hidden_states_proj = self._extract_component(self.state_dict, "subject_hidden_states_proj")
            
            print(f"[PIP-InstantCharacter] IP适配器加载成功, 包含 {len(self.state_dict.keys())} 个权重")
        except Exception as e:
            print(f"[PIP-InstantCharacter] IP适配器加载失败: {str(e)}")
            raise e
    
    def _extract_component(self, state_dict, prefix):
        """从状态字典中提取特定前缀的参数并构建组件"""
        params = {k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
        if not params:
            return None
            
        # 简单情况下，我们假设这是一个线性层
        if "weight" in params and "bias" in params:
            layer = torch.nn.Linear(
                in_features=params["weight"].shape[1], 
                out_features=params["weight"].shape[0]
            )
            layer.weight.data.copy_(params["weight"])
            layer.bias.data.copy_(params["bias"])
            return layer
        
        return None
