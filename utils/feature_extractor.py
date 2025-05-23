"""
InstantCharacter 特征提取器 - 完全按照原版实现
直接复制原版InstantCharacter的实现方式，确保效果一致
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Optional, Tuple, Dict, Any, Union, List
from PIL import Image
import numpy as np
from einops import rearrange
import traceback
import folder_paths

# 导入多尺度处理器
from .multi_scale_processor import MultiScaleProcessor

# 常量定义
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

class InstantCharacterFeatureExtractor:
    """
    InstantCharacter特征提取器 - 严格按照原版实现
    完全复制原版InstantCharacter的实现方式，确保效果一致
    """
    def __init__(self, 
                device: str = "cuda",
                low_memory: bool = True):
        """
        初始化特征提取器
        Args:
            device: 计算设备
            low_memory: 是否启用低内存模式
        """
        self.device = device
        self.low_memory = low_memory
        
        # 初始化模型引用
        self.dinov2_model = None
        self.siglip_model = None
        self.dino_processor = None
        self.siglip_processor = None
        
        # 创建多尺度处理器实例 - 这是原版InstantCharacter的关键组件
        self.processor = MultiScaleProcessor(device=device)
        
        print(f"[PIP-InstantCharacter] 特征提取器初始化完成 (设备: {device}, 低内存模式: {low_memory})")
    
    def load_models(self, dinov2_model, siglip_model, dino_processor=None, siglip_processor=None):
        """
        加载DINOv2和SigLIP模型及其处理器
        
        Args:
            dinov2_model: DinoV2视觉模型
            siglip_model: SigLIP视觉模型
            dino_processor: DinoV2图像处理器
            siglip_processor: SigLIP图像处理器
        """
        self.dinov2_model = dinov2_model
        self.siglip_model = siglip_model
        self.dino_processor = dino_processor
        self.siglip_processor = siglip_processor
        
        print(f"[PIP-InstantCharacter] 特征提取器已加载DinoV2模型")
        if siglip_model is not None:
            print(f"[PIP-InstantCharacter] 特征提取器已加载SigLIP模型")
    
    def extract_features(self, dinov2_model, siglip_model, image, dino_processor=None, siglip_processor=None, low_memory_mode=None):
        """
        从图像中提取多尺度DinoV2和SigLIP特征，完全按照原版InstantCharacter实现
        
        Args:
            dinov2_model: DinoV2视觉模型
            siglip_model: SigLIP视觉模型
            image: 输入图像，PIL图像格式
            dino_processor: DinoV2图像处理器
            siglip_processor: SigLIP图像处理器
            low_memory_mode: 是否启用低内存模式
            
        Returns:
            特征字典，包含低分辨率和高分辨率特征
        """
        # 优先使用传入的低内存模式设置
        if low_memory_mode is not None:
            self.low_memory = low_memory_mode
            
        # 如果模型未加载，自动加载
        if self.dinov2_model is None or self.siglip_model is None:
            self.load_models(dinov2_model, siglip_model, dino_processor, siglip_processor)
        
        # 使用传入的处理器，如果未传入则使用已加载的
        dino_processor_to_use = dino_processor if dino_processor is not None else self.dino_processor
        siglip_processor_to_use = siglip_processor if siglip_processor is not None else self.siglip_processor
        
        # 如果处理器仍然为None，给出警告但继续尝试
        if dino_processor_to_use is None:
            print("[PIP-InstantCharacter] 警告: 未提供DinoV2处理器，可能会影响特征提取")
        if siglip_processor_to_use is None:
            print("[PIP-InstantCharacter] 警告: 未提供SigLIP处理器，可能会影响特征提取")
        
        try:
            print("[PIP-InstantCharacter] 提取多尺度特征...")
            
            # 使用多尺度处理器提取特征，完全按照原版实现
            with torch.no_grad():
                # 使用处理器的encode_multi_scale_features方法提取特征
                image_embeds_dict = self.processor.encode_multi_scale_features(
                    siglip_processor=siglip_processor_to_use,
                    siglip_model=siglip_model,
                    dino_processor=dino_processor_to_use,
                    dino_model=dinov2_model,
                    image=image,
                    device=self.device,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                if image_embeds_dict is None:
                    print("[PIP-InstantCharacter] 特征提取失败，返回空字典")
                    return {}
                
                # 清理显存
                torch.cuda.empty_cache()
                
                # 构建特征字典 - 保持与ComfyUI实现的兼容性
                deep_features = {
                    "low_res": image_embeds_dict['image_embeds_low_res_deep'],
                    "high_res": image_embeds_dict['image_embeds_high_res_deep']
                }
                
                # 返回完整特征字典，包括原版特征和兼容特征
                return {
                    # 原版InstantCharacter特征格式
                    "image_embeds_low_res_shallow": image_embeds_dict['image_embeds_low_res_shallow'],
                    "image_embeds_low_res_deep": image_embeds_dict['image_embeds_low_res_deep'],
                    "image_embeds_high_res_deep": image_embeds_dict['image_embeds_high_res_deep'],
                    
                    # 兼容已有代码的特征格式
                    "deep_features": deep_features,
                    
                    # 通用的deep_features访问方式
                    "combined": deep_features["low_res"]
                }
                
        except Exception as e:
            print(f"[PIP-InstantCharacter] 特征提取失败: {e}")
            traceback.print_exc()
            
            # 出错时返回空字典
            return {}
    
    def to(self, device):
        """移动特征提取器到指定设备"""
        self.device = device
        if hasattr(self, 'processor') and self.processor is not None:
            self.processor.device = device
        
        # 移动各个可能的模型
        for name, module in self.__dict__.items():
            if isinstance(module, (nn.Module, torch.Tensor)) and module is not None:
                setattr(self, name, module.to(device))
        
        return self
