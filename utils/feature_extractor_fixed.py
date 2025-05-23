"""
InstantCharacter 特征提取器 - 按照原版实现
直接复制原版InstantCharacter的实现方式，确保效果一致
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
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
    InstantCharacter特征提取器 - 原版实现
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
        self.dinov2_model = None
        self.siglip_model = None
        self.low_memory = low_memory
        self.feature_projector = None
        
        # 创建多尺度处理器实例 - 这是原版InstantCharacter的关键组件
        self.processor = MultiScaleProcessor(device=device)
        
        print(f"[PIP-InstantCharacter] 特征提取器初始化完成 (设备: {device}, 低内存模式: {low_memory})")
        print(f"[PIP-InstantCharacter] 多尺度处理器已初始化: {self.processor.__class__.__name__}")
    
    def load_models(self, dinov2_model, siglip_model):
        """
        加载DINOv2和SigLIP模型
        
        Args:
            dinov2_model: DinoV2视觉模型
            siglip_model: SigLIP视觉模型(可选)
        """
        self.dinov2_model = dinov2_model
        self.siglip_model = siglip_model
        print(f"[PIP-InstantCharacter] 特征提取器已加载DinoV2模型")
        if siglip_model is not None:
            print(f"[PIP-InstantCharacter] 特征提取器已加载SigLIP模型")
        else:
            print(f"[PIP-InstantCharacter] 特征提取器未加载SigLIP模型(可选)")
    
    def extract_features(self, dinov2_model, siglip_model, image, low_memory_mode=None):
        """
        从图像中提取多尺度DinoV2和SigLIP特征，完全按照原版InstantCharacter实现
        
        Args:
            dinov2_model: DinoV2视觉模型
            siglip_model: SigLIP视觉模型(可选)
            image: 输入图像，可以是PIL或Tensor(BCHW格式)，将进行相应处理
            low_memory_mode: 是否启用低内存模式
            
        Returns:
            特征字典，包含低分辨率和高分辨率特征
        """
        # 优先使用传入的低内存模式设置
        if low_memory_mode is not None:
            self.low_memory = low_memory_mode
            
        # 如果模型未加载，自动加载
        if self.dinov2_model is None or self.siglip_model is None:
            self.load_models(dinov2_model, siglip_model)
        
        # 记录设备信息
        print(f"[PIP-InstantCharacter] DINOv2 设备: {getattr(dinov2_model, 'device', 'unknown')}")
        print(f"[PIP-InstantCharacter] SigLIP 设备: {getattr(siglip_model, 'device', 'unknown')}")
        
        try:
            # 处理多尺度图像输入
            print("[PIP-InstantCharacter] 处理多尺度图像输入...")
            processed_images = self.processor.process_image(image)
            
            # 单个图像转换为批次
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # [C,H,W] -> [1,C,H,W]
                    image = image.unsqueeze(0)
            
            print("[PIP-InstantCharacter] 提取多尺度特征...")
            
            # 使用低内存模式提取特征
            with torch.no_grad():
                # 提取低分辨率DinoV2特征
                dino_features_low = self.encode_image(
                    self.dinov2_model, 
                    processed_images["dino_low_res"], 
                    model_type="DinoV2-Low",
                    low_memory_mode=self.low_memory,
                    use_processed=True
                )
                
                # 提取高分辨率DinoV2特征
                dino_features_high = self.encode_image(
                    self.dinov2_model, 
                    processed_images["dino_high_res"], 
                    model_type="DinoV2-High",
                    low_memory_mode=self.low_memory,
                    use_processed=True
                )
                
                # 提取SigLIP特征(如果可用)
                siglip_features_low = None
                siglip_features_high = None
                if self.siglip_model is not None:
                    print("[PIP-InstantCharacter] 提取多尺度SigLIP特征...")
                    siglip_features_low = self.encode_image(
                        self.siglip_model, 
                        processed_images["siglip_low_res"], 
                        model_type="SigLIP-Low",
                        low_memory_mode=self.low_memory,
                        use_processed=True
                    )
                    
                    siglip_features_high = self.encode_image(
                        self.siglip_model, 
                        processed_images["siglip_high_res"], 
                        model_type="SigLIP-High",
                        low_memory_mode=self.low_memory,
                        use_processed=True
                    )
                
                # 清理显存
                torch.cuda.empty_cache()
                
                # 合并特征并返回特征字典
                combined_features = self._combine_features(
                    dino_features_low, 
                    siglip_features_low if siglip_features_low is not None else None
                )
                
                return {
                    "dino_features_low": dino_features_low,
                    "dino_features_high": dino_features_high,
                    "siglip_features_low": siglip_features_low,
                    "siglip_features_high": siglip_features_high,
                    "combined": combined_features,
                    # 添加原版InstantCharacter需要的特征格式
                    "deep_features": {
                        "low_res": torch.cat([dino_features_low, siglip_features_low], dim=-1) if siglip_features_low is not None else dino_features_low,
                        "high_res": torch.cat([dino_features_high, siglip_features_high], dim=-1) if siglip_features_high is not None else dino_features_high
                    }
                }
                
        except Exception as e:
            print(f"[PIP-InstantCharacter] 特征提取失败: {e}")
            traceback.print_exc()
            
            # 出错时返回空字典
            return {}
    
    def _extract_siglip_features(self, model, image):
        """提取SigLIP特征 - 已弃用，使用extract_features代替"""
        print("[PIP-InstantCharacter] 警告: 使用了弃用的_extract_siglip_features方法")
        return torch.randn(1, 1, 1152, device=self.device)
    
    def encode_image(self, vision_model, image, model_type="unknown", low_memory_mode=False, use_processed=False):
        """
        将图像编码为特征向量
        
        Args:
            vision_model: 视觉模型(DinoV2或SigLIP)
            image: 输入图像，必须是BCHW格式(内部处理)
            model_type: 模型类型，用于日志
            low_memory_mode: 低内存模式开关
            use_processed: 图像是否已经通过processor预处理
            
        Returns:
            编码后的特征
        """
        try:
            # 安全检查
            if vision_model is None:
                print(f"[PIP-InstantCharacter] 警告: {model_type}模型未加载，返回空特征")
                return self._create_special_features(model_type)
            
            # 确保输入张量在正确的设备上
            original_device = self.device if hasattr(self, 'device') else "cuda"
            model_device = next(vision_model.parameters()).device
            
            # 如果图像已经预处理，则直接使用
            if use_processed:
                # 图像已经处理过，确保在正确的设备上
                if isinstance(image, torch.Tensor):
                    image = image.to(model_device)
            else:
                # 检测图像格式并进行必要转换
                if isinstance(image, Image.Image):
                    # 对PIL图像进行预处理 - 具体操作取决于模型类型
                    if 'dino' in model_type.lower():
                        # DinoV2预处理
                        transform = T.Compose([
                            T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=DINO_MEAN, std=DINO_STD)
                        ])
                    else:
                        # SigLIP预处理
                        transform = T.Compose([
                            T.Resize(224),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD)
                        ])
                    image = transform(image).unsqueeze(0).to(model_device)
                elif isinstance(image, torch.Tensor):
                    # 如果是张量，确保格式和设备正确
                    if image.dim() == 3:
                        image = image.unsqueeze(0)
                    image = image.to(model_device)
                else:
                    raise ValueError(f"不支持的图像类型: {type(image)}")
            
            # 记录特征提取开始
            print(f"[PIP-InstantCharacter] 开始提取{model_type}特征...")
            
            # 根据模型类型提取特征
            if 'dino' in model_type.lower():
                # DinoV2特征提取
                with torch.no_grad():
                    if low_memory_mode:
                        # 低内存模式 - 逐批处理
                        features = []
                        batch_size = image.shape[0]
                        for i in range(batch_size):
                            with torch.cuda.amp.autocast():
                                output = vision_model(image[i:i+1])
                                # 提取CLS token特征
                                if hasattr(output, 'last_hidden_state'):
                                    feature = output.last_hidden_state[:, 0]
                                else:
                                    feature = output[:, 0]
                            features.append(feature)
                        features = torch.cat(features, dim=0)
                    else:
                        # 标准模式 - 一次性处理
                        with torch.cuda.amp.autocast():
                            output = vision_model(image)
                            # 提取CLS token特征
                            if hasattr(output, 'last_hidden_state'):
                                features = output.last_hidden_state[:, 0]
                            else:
                                features = output[:, 0]
            else:
                # SigLIP特征提取
                with torch.no_grad():
                    if low_memory_mode:
                        # 低内存模式 - 逐批处理
                        features = []
                        batch_size = image.shape[0]
                        for i in range(batch_size):
                            with torch.cuda.amp.autocast():
                                outputs = vision_model(image[i:i+1], output_hidden_states=True)
                                # 提取hidden_state和image_embeds
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                    # 融合不同层的特征以获得更丰富的表示
                                    text_feature = outputs.hidden_states[-1][:, 0]  # CLS token
                                    image_feature = outputs.image_embeds
                                    feature = torch.cat([text_feature, image_feature], dim=-1)
                                else:
                                    # 如果没有隐藏状态，使用直接输出
                                    feature = outputs.image_embeds
                            features.append(feature)
                        features = torch.cat(features, dim=0)
                    else:
                        # 标准模式 - 一次性处理
                        with torch.cuda.amp.autocast():
                            outputs = vision_model(image, output_hidden_states=True)
                            # 提取hidden_state和image_embeds
                            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                # 融合不同层的特征以获得更丰富的表示
                                text_feature = outputs.hidden_states[-1][:, 0]  # CLS token
                                image_feature = outputs.image_embeds
                                features = torch.cat([text_feature, image_feature], dim=-1)
                            else:
                                # 如果没有隐藏状态，使用直接输出
                                features = outputs.image_embeds
            
            # 后处理特征（归一化等）
            features = F.normalize(features, p=2, dim=-1)
            
            # 将特征转换为(B, N, D)格式，适合后续处理
            if features.dim() == 2:
                features = features.unsqueeze(1)  # [B, D] -> [B, 1, D]
            
            # 将特征转回原始设备
            features = features.to(original_device)
            
            print(f"[PIP-InstantCharacter] {model_type}特征提取完成: {features.shape}")
            
            return features
        
        except Exception as e:
            print(f"[PIP-InstantCharacter] {model_type}特征提取失败: {e}")
            traceback.print_exc()
            
            # 出错时返回空特征
            return self._create_special_features(model_type)
    
    def _create_special_features(self, model_type):
        """创建特殊特征向量(用于错误情况)"""
        # 根据模型类型返回不同维度的空特征
        if "dino" in model_type.lower():
            # DinoV2特征维度
            return torch.zeros(1, 1, 1024, device=self.device)
        else:
            # SigLIP特征维度
            return torch.zeros(1, 1, 1152, device=self.device)
    
    def _combine_features(self, dino_features, siglip_features=None):
        """合并DinoV2和SigLIP特征"""
        if siglip_features is None:
            # 只有DinoV2特征
            return dino_features
        else:
            # 合并两种特征
            return torch.cat([dino_features, siglip_features], dim=-1)
    
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
