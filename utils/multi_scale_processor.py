"""
多尺度处理器 - 用于处理多分辨率图像
实现完全按照原版InstantCharacter的处理逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any, Union
from torchvision import transforms
from einops import rearrange
import traceback
import time
import gc
from PIL import Image
import numpy as np

# 常量定义 - 与原版InstantCharacter保持一致
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

class MultiScaleProcessor:
    """
    多尺度图像处理器 - 完全按照原版InstantCharacter实现
    同时处理低分辨率和高分辨率图像
    """
    def __init__(self, device="cuda"):
        """
        初始化多尺度处理器
        Args:
            device: 计算设备
        """
        self.device = device
        print(f"[PIP-InstantCharacter] 多尺度处理器已初始化: {self.__class__.__name__}")
        
    def prepare_multi_scale_images(self, image, low_res_size=384, high_res_size=768):
        """
        准备多尺度图像 - 完全按照原版InstantCharacter实现
        分别处理低分辨率图像和高分辨率图像
        
        Args:
            image: 输入图像 (PIL格式)
            low_res_size: 低分辨率图像大小
            high_res_size: 高分辨率图像大小
            
        Returns:
            包含低分辨率和高分辨率区域的字典
        """
        try:
            # 先准备低分辨率图像 (384x384)
            image_low_res = image.resize((low_res_size, low_res_size))
            
            # 然后准备高分辨率图像 (768x768) 并分割为4个区域
            image_high_res = image.resize((high_res_size, high_res_size))
            
            # 分割为4个区域 - 完全按照原版实现
            high_res_regions = [
                image_high_res.crop((0, 0, low_res_size, low_res_size)),                # 左上
                image_high_res.crop((high_res_size-low_res_size, 0, high_res_size, low_res_size)),    # 右上
                image_high_res.crop((0, high_res_size-low_res_size, low_res_size, high_res_size)),    # 左下
                image_high_res.crop((high_res_size-low_res_size, high_res_size-low_res_size, high_res_size, high_res_size))  # 右下
            ]
            
            # 返回多尺度图像字典
            return {
                'low_res': image_low_res,  # 单个低分辨率图像
                'high_res_regions': high_res_regions  # 4个高分辨率区域
            }
            
        except Exception as e:
            print(f"[PIP-InstantCharacter] 多尺度图像准备失败: {e}")
            traceback.print_exc()
            return None
    
    def encode_siglip_image(self, siglip_processor, siglip_model, images=None, device="cuda", dtype=torch.float16, pixel_values=None):
        """
        使用SigLIP模型编码图像 - 复制原版encode_siglip_image_emb
        支持ComfyUI的CLIP视觉模型直接使用
        """
        try:
            # 检测是否为ComfyUI的CLIP视觉模型
            is_comfy_clip = hasattr(siglip_model, 'encode_image')
            
            if is_comfy_clip:
                # 对于ComfyUI的CLIP视觉模型，直接使用encode_image接口
                print("[PIP-InstantCharacter] 检测到ComfyUI CLIP视觉模型，直接使用encode_image接口")
                
                # 准备图像输入
                pil_images = []
                if images is not None:
                    if isinstance(images, list):
                        pil_images = images
                    else:
                        pil_images = [images]
                
                # 手动将PIL图像转换为张量
                tensor_images = []
                for img in pil_images:
                    # 使用标准SigLIP预处理 - 调整大小，转为张量，归一化
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD)
                    ])
                    tensor_img = transform(img).unsqueeze(0).to(device, dtype=dtype)  # 添加批次维度
                    tensor_images.append(tensor_img)
                
                # 逐个处理图像
                all_features = []
                for tensor_img in tensor_images:
                    try:
                        # 尝试使用模型的encode_image方法
                        features = siglip_model.encode_image(tensor_img)
                        all_features.append(features)
                    except Exception as e:
                        print(f"[PIP-InstantCharacter] 尝试使用encode_image失败: {e}")
                        # 尝试直接调用模型
                        try:
                            features = siglip_model(tensor_img)
                            if hasattr(features, 'last_hidden_state'):
                                all_features.append(features.last_hidden_state)
                            elif hasattr(features, 'image_embeds'):
                                all_features.append(features.image_embeds.unsqueeze(1))
                            else:
                                # 假设返回的是特征
                                all_features.append(features.unsqueeze(1) if features.dim() == 2 else features)
                        except Exception as e:
                            print(f"[PIP-InstantCharacter] 模型直接调用也失败: {e}")
                            continue
                
                # 合并特征
                if len(all_features) > 0:
                    try:
                        # 先检查是否是ComfyUI特有的Output对象
                        if hasattr(all_features[0], 'image_embeds') and not isinstance(all_features[0], torch.Tensor):
                            # ComfyUI返回了特殊的Output对象
                            print("[PIP-InstantCharacter] 检测到ComfyUI特殊输出格式，正在适配...")
                            if len(all_features) > 1:
                                # 如果有多个特征，尝试提取image_embeds并合并
                                extracted_features = []
                                for feat in all_features:
                                    if hasattr(feat, 'image_embeds'):
                                        extracted_features.append(feat.image_embeds)
                                    else:
                                        # 尝试其他属性
                                        found = False
                                        for attr_name in ['last_hidden_state', 'pooler_output', 'features']:
                                            if hasattr(feat, attr_name):
                                                extracted_features.append(getattr(feat, attr_name))
                                                found = True
                                                break
                                        if not found and hasattr(feat, '__dict__'):
                                            # 如果没有常见属性，显示可用属性帮助调试
                                            print(f"[PIP-InstantCharacter] 未知输出格式，可用属性: {feat.__dict__.keys()}")
                                if extracted_features:
                                    combined_features = torch.cat(extracted_features, dim=0)
                                else:
                                    print("[PIP-InstantCharacter] 无法从特殊格式提取特征")
                                    return None, None
                            else:
                                # 单个特征对象
                                feat = all_features[0]
                                if hasattr(feat, 'image_embeds'):
                                    combined_features = feat.image_embeds
                                elif hasattr(feat, 'last_hidden_state'):
                                    combined_features = feat.last_hidden_state
                                elif hasattr(feat, 'pooler_output'):
                                    combined_features = feat.pooler_output
                                else:
                                    # 如果没有常见属性，显示可用属性帮助调试
                                    if hasattr(feat, '__dict__'):
                                        print(f"[PIP-InstantCharacter] 未知输出格式，可用属性: {feat.__dict__.keys()}")
                                    return None, None
                        else:
                            # 正常张量处理
                            combined_features = torch.cat(all_features, dim=0) if len(all_features) > 1 else all_features[0]
                            
                        # 确保是张量格式并检查维度
                        if isinstance(combined_features, torch.Tensor):
                            if combined_features.dim() == 2:
                                combined_features = combined_features.unsqueeze(1)  # [B, 1, D]
                            
                            # 返回深层和浅层特征（简化处理）
                            return combined_features, combined_features
                        else:
                            print(f"[PIP-InstantCharacter] 错误: 提取的特征不是张量格式: {type(combined_features)}")
                            return None, None
                    except Exception as e:
                        print(f"[PIP-InstantCharacter] 特征处理错误: {e}")
                        import traceback
                        traceback.print_exc()
                        return None, None
                else:
                    print("[PIP-InstantCharacter] 错误: 没有有效的图像特征从SigLIP获取")
                    return None, None
            
            # 非ComfyUI模型，使用原始处理方式
            if pixel_values is None and images is not None and siglip_processor is not None:
                # 使用SigLIP处理器处理图像
                pixel_values = siglip_processor(images=images, return_tensors="pt").pixel_values
            elif pixel_values is None and images is not None:
                # 如果没有处理器，使用默认变换
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD)
                ])
                
                if isinstance(images, list):
                    pixel_values = torch.stack([transform(img) for img in images])
                else:
                    pixel_values = transform(images).unsqueeze(0)
            
            if pixel_values is None:
                print("[PIP-InstantCharacter] 错误: 没有有效的图像或像素值提供给SigLIP编码器")
                return None, None
                
            pixel_values = pixel_values.to(device, dtype=dtype)
            
            # 用SigLIP模型提取特征
            with torch.no_grad():
                try:
                    res = siglip_model(pixel_values, output_hidden_states=True)
                    
                    # 提取最后一层隐藏状态作为深层特征
                    siglip_image_embeds = res.last_hidden_state
                    
                    # 连接特定层的隐藏状态作为浅层特征 (与原版一致使用7,13,26层)
                    siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
                    
                    return siglip_image_embeds, siglip_image_shallow_embeds
                except Exception as e:
                    print(f"[PIP-InstantCharacter] 标准SigLIP接口调用失败: {e}")
                    # 尝试最简单的调用方式
                    features = siglip_model(pixel_values)
                    if hasattr(features, 'image_embeds'):
                        return features.image_embeds.unsqueeze(1), features.image_embeds.unsqueeze(1)
                    elif hasattr(features, 'last_hidden_state'):
                        return features.last_hidden_state, features.last_hidden_state
                    else:
                        # 假设返回的就是特征
                        return features.unsqueeze(1) if features.dim() == 2 else features, \
                               features.unsqueeze(1) if features.dim() == 2 else features
        except Exception as e:
            print(f"[PIP-InstantCharacter] SigLIP编码失败: {e}")
            traceback.print_exc()
            return None, None
    
    def encode_dinov2_image(self, dino_processor, dino_model, images=None, device="cuda", dtype=torch.float16, pixel_values=None):
        """
        使用DinoV2模型编码图像 - 复制原版encode_dinov2_image_emb
        支持ComfyUI的CLIP视觉模型直接使用
        """
        try:
            # 检测是否为ComfyUI的CLIP视觉模型
            is_comfy_clip = hasattr(dino_model, 'encode_image')
            
            if is_comfy_clip:
                # 对于ComfyUI的CLIP视觉模型，直接使用encode_image接口
                print("[PIP-InstantCharacter] 检测到ComfyUI CLIP视觉模型，直接使用encode_image接口")
                
                # 准备图像输入
                pil_images = []
                if images is not None:
                    if isinstance(images, list):
                        pil_images = images
                    else:
                        pil_images = [images]
                
                # 手动将PIL图像转换为张量
                tensor_images = []
                for img in pil_images:
                    # 使用标准DinoV2预处理 - 调整大小，转为张量，归一化
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=DINO_MEAN, std=DINO_STD)
                    ])
                    tensor_img = transform(img).unsqueeze(0).to(device, dtype=dtype)  # 添加批次维度
                    tensor_images.append(tensor_img)
                
                # 逐个处理图像
                all_features = []
                for tensor_img in tensor_images:
                    try:
                        # 尝试使用模型的encode_image方法
                        features = dino_model.encode_image(tensor_img)
                        all_features.append(features)
                    except Exception as e:
                        print(f"[PIP-InstantCharacter] 尝试使用encode_image失败: {e}")
                        # 尝试直接调用模型
                        try:
                            features = dino_model(tensor_img)
                            if hasattr(features, 'last_hidden_state'):
                                # DinoV2特有: 删除CLS标记（第一个令牌）
                                hidden_state = features.last_hidden_state
                                if hidden_state.size(1) > 1:  # 确保有足够的令牌
                                    hidden_state = hidden_state[:, 1:]
                                all_features.append(hidden_state)
                            elif hasattr(features, 'image_embeds'):
                                all_features.append(features.image_embeds.unsqueeze(1))
                            else:
                                # 假设返回的是特征
                                all_features.append(features.unsqueeze(1) if features.dim() == 2 else features)
                        except Exception as e:
                            print(f"[PIP-InstantCharacter] 模型直接调用也失败: {e}")
                            continue
                
                # 合并特征
                if len(all_features) > 0:
                    try:
                        # 先检查是否是ComfyUI特有的Output对象
                        if hasattr(all_features[0], 'image_embeds') and not isinstance(all_features[0], torch.Tensor):
                            # ComfyUI返回了特殊的Output对象
                            print("[PIP-InstantCharacter] 检测到ComfyUI特殊输出格式，正在适配...")
                            if len(all_features) > 1:
                                # 如果有多个特征，尝试提取image_embeds并合并
                                extracted_features = []
                                for feat in all_features:
                                    if hasattr(feat, 'image_embeds'):
                                        extracted_features.append(feat.image_embeds)
                                    else:
                                        # 尝试其他属性
                                        found = False
                                        for attr_name in ['last_hidden_state', 'pooler_output', 'features']:
                                            if hasattr(feat, attr_name):
                                                # DinoV2特有: 删除CLS标记（第一个令牌）
                                                hidden_state = getattr(feat, attr_name)
                                                if isinstance(hidden_state, torch.Tensor) and hidden_state.size(1) > 1:
                                                    hidden_state = hidden_state[:, 1:]
                                                extracted_features.append(hidden_state)
                                                found = True
                                                break
                                        if not found and hasattr(feat, '__dict__'):
                                            # 如果没有常见属性，显示可用属性帮助调试
                                            print(f"[PIP-InstantCharacter] 未知输出格式，可用属性: {feat.__dict__.keys()}")
                                if extracted_features:
                                    combined_features = torch.cat(extracted_features, dim=0)
                                else:
                                    print("[PIP-InstantCharacter] 无法从特殊格式提取特征")
                                    return None, None
                            else:
                                # 单个特征对象
                                feat = all_features[0]
                                if hasattr(feat, 'image_embeds'):
                                    combined_features = feat.image_embeds
                                elif hasattr(feat, 'last_hidden_state'):
                                    # DinoV2特有: 删除CLS标记（第一个令牌）
                                    hidden_state = feat.last_hidden_state
                                    if hidden_state.size(1) > 1:  # 确保有足够的令牌
                                        combined_features = hidden_state[:, 1:]
                                    else:
                                        combined_features = hidden_state
                                elif hasattr(feat, 'pooler_output'):
                                    combined_features = feat.pooler_output
                                else:
                                    # 如果没有常见属性，显示可用属性帮助调试
                                    if hasattr(feat, '__dict__'):
                                        print(f"[PIP-InstantCharacter] 未知输出格式，可用属性: {feat.__dict__.keys()}")
                                    return None, None
                        else:
                            # 正常张量处理
                            combined_features = torch.cat(all_features, dim=0) if len(all_features) > 1 else all_features[0]
                            
                        # 确保是张量格式并检查维度
                        if isinstance(combined_features, torch.Tensor):
                            if combined_features.dim() == 2:
                                combined_features = combined_features.unsqueeze(1)  # [B, 1, D]
                            
                            # 返回深层和浅层特征（简化处理）
                            return combined_features, combined_features
                        else:
                            print(f"[PIP-InstantCharacter] 错误: 提取的特征不是张量格式: {type(combined_features)}")
                            return None, None
                    except Exception as e:
                        print(f"[PIP-InstantCharacter] 特征处理错误: {e}")
                        import traceback
                        traceback.print_exc()
                        return None, None
                else:
                    print("[PIP-InstantCharacter] 错误: 没有有效的图像特征从DinoV2获取")
                    return None, None
            
            # 非ComfyUI模型，使用原始处理方式
            if pixel_values is None and images is not None and dino_processor is not None:
                # 使用DinoV2处理器处理图像
                pixel_values = dino_processor(images=images, return_tensors="pt").pixel_values
            elif pixel_values is None and images is not None:
                # 如果没有处理器，使用默认变换
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DINO_MEAN, std=DINO_STD)
                ])
                
                if isinstance(images, list):
                    pixel_values = torch.stack([transform(img) for img in images])
                else:
                    pixel_values = transform(images).unsqueeze(0)
            
            if pixel_values is None:
                print("[PIP-InstantCharacter] 错误: 没有有效的图像或像素值提供给DinoV2编码器")
                return None, None
                
            pixel_values = pixel_values.to(device, dtype=dtype)
            
            # 用DinoV2模型提取特征
            with torch.no_grad():
                try:
                    res = dino_model(pixel_values, output_hidden_states=True)
                    
                    # 提取最后一层隐藏状态作为深层特征，不包括[CLS]令牌
                    dinov2_image_embeds = res.last_hidden_state[:, 1:]
                    
                    # 连接特定层的隐藏状态作为浅层特征 (与原版一致使用9,19,29层)
                    dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
                    
                    return dinov2_image_embeds, dinov2_image_shallow_embeds
                except Exception as e:
                    print(f"[PIP-InstantCharacter] 标准DinoV2接口调用失败: {e}")
                    # 尝试最简单的调用方式
                    features = dino_model(pixel_values)
                    if hasattr(features, 'image_embeds'):
                        return features.image_embeds.unsqueeze(1), features.image_embeds.unsqueeze(1)
                    elif hasattr(features, 'last_hidden_state'):
                        last_hidden = features.last_hidden_state
                        # 尝试删除CLS标记，如果存在
                        if last_hidden.size(1) > 1:
                            return last_hidden[:, 1:], last_hidden[:, 1:]
                        return last_hidden, last_hidden
                    else:
                        # 假设返回的就是特征
                        return features.unsqueeze(1) if features.dim() == 2 else features, \
                               features.unsqueeze(1) if features.dim() == 2 else features
        except Exception as e:
            print(f"[PIP-InstantCharacter] DinoV2编码失败: {e}")
            traceback.print_exc()
            return None, None
    
    def encode_multi_scale_features(self, siglip_processor, siglip_model, dino_processor, dino_model, image, device="cuda", dtype=torch.float16):
        """
        提取多尺度特征 - 完全符合原版InstantCharacter实现
        1. 处理低分辨率图像 - 384x384
        2. 处理高分辨率图像 - 将768x768的图像分为4个384x384的区域处理
        
        参数:
            siglip_processor: SigLIP图像处理器
            siglip_model: SigLIP视觉模型
            dino_processor: DinoV2图像处理器
            dino_model: DinoV2视觉模型
            image: 输入图像
            device: 计算设备
            dtype: 数据类型
        
        返回:
            多尺度特征字典
        """
        try:
            # 检测是否为ComfyUI CLIP视觉模型
            is_comfy_siglip = hasattr(siglip_model, 'encode_image') if siglip_model is not None else False
            is_comfy_dino = hasattr(dino_model, 'encode_image') if dino_model is not None else False
            
            if is_comfy_siglip:
                print(f"[PIP-InstantCharacter] 检测到ComfyUI SigLIP视觉模型，使用直接接口")
            if is_comfy_dino:
                print(f"[PIP-InstantCharacter] 检测到ComfyUI DinoV2视觉模型，使用直接接口")
            
            # 记录原始图像尺寸
            print(f"[PIP-InstantCharacter] 原始图像尺寸: {image.size}")
            
            # 1. 低分辨率图像预处理 (384x384)
            images_low_res = self.prepare_multi_scale_images(
                image=image, 
                low_res_size=384,
                high_res_size=768
            )
            
            # 检查图像是否正确预处理
            if images_low_res is None or 'low_res' not in images_low_res:
                print("[PIP-InstantCharacter] 图像预处理失败，无法提取特征")
                return None
            
            print(f"[PIP-InstantCharacter] 图像处理完成: 低分辨率(384x384)和高分辨率区域(768x768切分为4个384x384区域)")
            
            # 2. 低分辨率特征提取 (SigLIP + DinoV2)
            # 提取SigLIP低分辨率特征
            siglip_low_res_embeds = (None, None)
            if siglip_model is not None:
                siglip_low_res_embeds = self.encode_siglip_image(
                    siglip_processor=siglip_processor,
                    siglip_model=siglip_model,
                    images=images_low_res['low_res'],
                    device=device,
                    dtype=dtype
                )
            else:
                print("[PIP-InstantCharacter] 警告: SigLIP模型未提供，将使用全DinoV2模式")
            
            # 提取DinoV2低分辨率特征
            dinov2_low_res_embeds = (None, None)
            if dino_model is not None:
                dinov2_low_res_embeds = self.encode_dinov2_image(
                    dino_processor=dino_processor,
                    dino_model=dino_model,
                    images=images_low_res['low_res'],
                    device=device,
                    dtype=dtype
                )
            else:
                print("[PIP-InstantCharacter] 警告: DinoV2模型未提供，将使用全SigLIP模式")
            
            # 检查是否至少有一个特征提取成功
            if (siglip_low_res_embeds[0] is None and dinov2_low_res_embeds[0] is None):
                print("[PIP-InstantCharacter] 错误: SigLIP和DinoV2特征提取均失败")
                return None
            
            # 合并低分辨率特征 - 深层特征和浅层特征
            try:
                if siglip_low_res_embeds[0] is not None and dinov2_low_res_embeds[0] is not None:
                    # 两个模型特征都存在 - 需要检查并处理尺寸不匹配
                    siglip_deep = siglip_low_res_embeds[0]
                    dinov2_deep = dinov2_low_res_embeds[0]
                    siglip_shallow = siglip_low_res_embeds[1]
                    dinov2_shallow = dinov2_low_res_embeds[1]
                    
                    # 打印输入张量尺寸以便调试
                    print(f"[PIP-InstantCharacter] SigLIP深层特征尺寸: {siglip_deep.shape}")
                    print(f"[PIP-InstantCharacter] DinoV2深层特征尺寸: {dinov2_deep.shape}")
                    print(f"[PIP-InstantCharacter] SigLIP浅层特征尺寸: {siglip_shallow.shape}")
                    print(f"[PIP-InstantCharacter] DinoV2浅层特征尺寸: {dinov2_shallow.shape}")
                    
                    # 根据实际情况判断是否需要尺寸调整
                    if siglip_deep.shape[1] != dinov2_deep.shape[1]:
                        # 如果在序列维度上不匹配，我们需要调整
                        print(f"[PIP-InstantCharacter] 注意: 特征序列长度不匹配，进行自适应处理")
                        
                        # 安全的做法是使用其中一个模型的特征，而不是尝试合并
                        if siglip_deep.shape[1] > dinov2_deep.shape[1]:
                            # 使用SigLIP特征
                            print("[PIP-InstantCharacter] 使用SigLIP特征作为全部特征")
                            image_embeds_low_res_deep = siglip_deep
                            image_embeds_low_res_shallow = siglip_shallow
                        else:
                            # 使用DinoV2特征
                            print("[PIP-InstantCharacter] 使用DinoV2特征作为全部特征")
                            image_embeds_low_res_deep = dinov2_deep
                            image_embeds_low_res_shallow = dinov2_shallow
                    else:
                        # 检查特征维度是否兼容
                        try:
                            # 尝试按原来方式合并
                            image_embeds_low_res_deep = torch.cat([siglip_deep, dinov2_deep], dim=2)
                            image_embeds_low_res_shallow = torch.cat([siglip_shallow, dinov2_shallow], dim=2)
                            print("[PIP-InstantCharacter] 成功合并SigLIP和DinoV2特征")
                        except Exception as e:
                            print(f"[PIP-InstantCharacter] 特征合并失败: {e}, 选择一个模型的特征作为回退")
                            # 选择其中一个特征作为回退
                            if siglip_deep.shape[-1] >= dinov2_deep.shape[-1]:
                                image_embeds_low_res_deep = siglip_deep
                                image_embeds_low_res_shallow = siglip_shallow
                                print("[PIP-InstantCharacter] 使用SigLIP特征作为回退")
                            else:
                                image_embeds_low_res_deep = dinov2_deep
                                image_embeds_low_res_shallow = dinov2_shallow
                                print("[PIP-InstantCharacter] 使用DinoV2特征作为回退")
                elif siglip_low_res_embeds[0] is not None:
                    # 只有SigLIP特征
                    image_embeds_low_res_deep = siglip_low_res_embeds[0]
                    image_embeds_low_res_shallow = siglip_low_res_embeds[1]
                    print("[PIP-InstantCharacter] 使用纯SigLIP特征作为低分辨率特征")
                else:
                    # 只有DinoV2特征
                    image_embeds_low_res_deep = dinov2_low_res_embeds[0]
                    image_embeds_low_res_shallow = dinov2_low_res_embeds[1]
                    print("[PIP-InstantCharacter] 使用纯DinoV2特征作为低分辨率特征")
            except Exception as e:
                print(f"[PIP-InstantCharacter] 特征合并阶段出错: {e}")
                import traceback
                traceback.print_exc()
                # 选择一个可用的特征作为回退
                if siglip_low_res_embeds[0] is not None:
                    image_embeds_low_res_deep = siglip_low_res_embeds[0]
                    image_embeds_low_res_shallow = siglip_low_res_embeds[1]
                    print("[PIP-InstantCharacter] 错误恢复: 使用SigLIP特征作为回退")
                elif dinov2_low_res_embeds[0] is not None:
                    image_embeds_low_res_deep = dinov2_low_res_embeds[0]
                    image_embeds_low_res_shallow = dinov2_low_res_embeds[1]
                    print("[PIP-InstantCharacter] 错误恢复: 使用DinoV2特征作为回退")
                else:
                    print("[PIP-InstantCharacter] 严重错误: 无法获取任何特征")
                    return None
            
            # 3. 高分辨率区域特征提取 (只提取DinoV2特征)
            # 获取高分辨率区域图像 - 4个分区
            if 'high_res_regions' not in images_low_res or len(images_low_res['high_res_regions']) != 4:
                print("[PIP-InstantCharacter] 警告: 高分辨率区域未正确生成，跳过高分辨率特征提取")
                # 使用低分辨率特征作为回退
                return {
                    'image_embeds_low_res_shallow': image_embeds_low_res_shallow,
                    'image_embeds_low_res_deep': image_embeds_low_res_deep,
                    'image_embeds_high_res_deep': image_embeds_low_res_deep  # 使用低分辨率作为回退
                }
            
            # 初始化高分辨率区域特征列表
            high_res_embeds_list = []
            
            # 判断使用哪个模型提取高分辨率特征
            high_res_model = dino_model if dino_model is not None else siglip_model
            high_res_processor = dino_processor if dino_model is not None else siglip_processor
            model_name = "DinoV2" if dino_model is not None else "SigLIP"
            
            print(f"[PIP-InstantCharacter] 使用{model_name}提取高分辨率区域特征")
            
            for i, region in enumerate(images_low_res['high_res_regions']):
                # 提取高分辨率区域特征
                if model_name == "DinoV2":
                    # 如果使用DinoV2模型，只传递DinoV2相关参数
                    region_embeds = self.encode_dinov2_image(
                        dino_processor=high_res_processor,
                        dino_model=high_res_model,
                        images=region,
                        device=device,
                        dtype=dtype
                    )
                else:
                    # 如果使用SigLIP模型，只传递SigLIP相关参数
                    region_embeds = self.encode_siglip_image(
                        siglip_processor=high_res_processor,
                        siglip_model=high_res_model,
                        images=region,
                        device=device,
                        dtype=dtype
                    )
                
                if region_embeds[0] is not None:
                    high_res_embeds_list.append(region_embeds[0])
            
            # 检查是否有足够的区域特征
            if len(high_res_embeds_list) > 0:
                try:
                    # 打印区域特征信息
                    print(f"[PIP-InstantCharacter] 区域特征数量: {len(high_res_embeds_list)}")
                    print(f"[PIP-InstantCharacter] 第一个区域特征尺寸: {high_res_embeds_list[0].shape}")
                    
                    # 尝试合并所有区域特征
                    image_embeds_high_res_deep = torch.cat(high_res_embeds_list, dim=1)
                    print(f"[PIP-InstantCharacter] 成功合并高分辨率区域特征，结果尺寸: {image_embeds_high_res_deep.shape}")
                except Exception as e:
                    print(f"[PIP-InstantCharacter] 高分辨率区域特征合并失败: {e}，使用低分辨率特征作为回退")
                    import traceback
                    traceback.print_exc()
                    image_embeds_high_res_deep = image_embeds_low_res_deep
            else:
                # 如果没有区域特征，使用低分辨率特征
                print("[PIP-InstantCharacter] 无区域特征可用，使用低分辨率特征")
                image_embeds_high_res_deep = image_embeds_low_res_deep
            
            # 返回完整特征字典
            return {
                'image_embeds_low_res_shallow': image_embeds_low_res_shallow,
                'image_embeds_low_res_deep': image_embeds_low_res_deep,
                'image_embeds_high_res_deep': image_embeds_high_res_deep
            }
            
        except Exception as e:
            print(f"[PIP-InstantCharacter] 多尺度特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None