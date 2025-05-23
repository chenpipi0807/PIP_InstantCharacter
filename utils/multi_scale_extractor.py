import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from einops import rearrange
from typing import Dict, Tuple, Optional

class MultiScaleImageProcessor:
    """多尺度图像处理器，与原版实现完全一致"""
    def __init__(self, low_res=384, high_res=768):
        self.low_res = low_res
        self.high_res = high_res
        
        # DINOv2预处理
        self.dino_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # SigLIP预处理
        self.siglip_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
    def process_image(self, pil_image: Image.Image) -> Dict[str, torch.Tensor]:
        """处理图像并返回多尺度特征"""
        # 低分辨率处理
        low_res_pil = pil_image.resize((self.low_res, self.low_res), Image.BICUBIC)
        
        # 高分辨率处理 - 分成4个patch
        high_res_pil = pil_image.resize((self.high_res, self.high_res), Image.BICUBIC)
        high_res_patches = [
            high_res_pil.crop((0, 0, self.low_res, self.low_res)),
            high_res_pil.crop((self.high_res-self.low_res, 0, self.high_res, self.low_res)),
            high_res_pil.crop((0, self.high_res-self.low_res, self.low_res, self.high_res)),
            high_res_pil.crop((self.high_res-self.low_res, self.high_res-self.low_res, self.high_res, self.high_res))
        ]
        
        # 转换为tensor并标准化
        to_tensor = transforms.ToTensor()
        
        # 处理低分辨率图像
        dino_low_res = self.dino_normalize(to_tensor(low_res_pil)).unsqueeze(0)
        siglip_low_res = self.siglip_normalize(to_tensor(low_res_pil)).unsqueeze(0)
        
        # 处理高分辨率patch
        dino_high_res = torch.stack([self.dino_normalize(to_tensor(patch)) for patch in high_res_patches])
        siglip_high_res = torch.stack([self.siglip_normalize(to_tensor(patch)) for patch in high_res_patches])
        
        return {
            'dino_low_res': dino_low_res,
            'siglip_low_res': siglip_low_res,
            'dino_high_res': dino_high_res,
            'siglip_high_res': siglip_high_res,
        }

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器，与原版实现一致"""
    def __init__(self, dinov2_model, siglip_model, device='cuda'):
        super().__init__()
        self.dinov2_model = dinov2_model.to(device)
        self.siglip_model = siglip_model.to(device)
        self.device = device
        self.processor = MultiScaleImageProcessor()
        
        # 冻结模型参数
        for param in self.dinov2_model.parameters():
            param.requires_grad = False
        for param in self.siglip_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode_siglip_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取SigLIP特征，返回深层和浅层特征"""
        outputs = self.siglip_model(image, output_hidden_states=True)
        deep_features = outputs.last_hidden_state
        # 从不同层提取浅层特征
        shallow_features = torch.cat([outputs.hidden_states[i] for i in [7, 13, 26]], dim=1)
        return deep_features, shallow_features
    
    @torch.no_grad()
    def encode_dinov2_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取DINOv2特征，返回深层和浅层特征"""
        outputs = self.dinov2_model(image, output_hidden_states=True)
        # 移除CLS token
        deep_features = outputs.last_hidden_state[:, 1:]
        # 从不同层提取浅层特征
        shallow_features = torch.cat([outputs.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        return deep_features, shallow_features
    
    def forward(self, pil_image: Image.Image) -> Dict[str, torch.Tensor]:
        """处理图像并提取多尺度特征"""
        # 处理图像
        processed = self.processor.process_image(pil_image)
        
        # 移动数据到设备
        dino_low_res = processed['dino_low_res'].to(self.device)
        siglip_low_res = processed['siglip_low_res'].to(self.device)
        dino_high_res = processed['dino_high_res'].to(self.device)
        siglip_high_res = processed['siglip_high_res'].to(self.device)
        
        # 提取低分辨率特征
        with torch.amp.autocast('cuda'):
            # 低分辨率特征
            siglip_low_deep, siglip_low_shallow = self.encode_siglip_features(siglip_low_res)
            dino_low_deep, dino_low_shallow = self.encode_dinov2_features(dino_low_res)
            
            # 高分辨率特征
            b, n, c, h, w = dino_high_res.shape
            dino_high_flat = dino_high_res.view(-1, c, h, w)
            siglip_high_flat = siglip_high_res.view(-1, c, h, w)
            
            dino_high_deep, _ = self.encode_dinov2_features(dino_high_flat)
            siglip_high_deep, _ = self.encode_siglip_features(siglip_high_flat)
            
            # 重组高分辨率特征
            dino_high_deep = dino_high_deep.view(b, -1, dino_high_deep.size(-1))
            siglip_high_deep = siglip_high_deep.view(b, -1, siglip_high_deep.size(-1))
            
            # 特征融合 - 与原版完全一致
            image_embeds_low_res_shallow = torch.cat([siglip_low_shallow, dino_low_shallow], dim=2)
            image_embeds_low_res_deep = torch.cat([siglip_low_deep, dino_low_deep], dim=2)
            image_embeds_high_res_deep = torch.cat([siglip_high_deep, dino_high_deep], dim=2)
            
            return {
                'image_embeds_low_res_shallow': image_embeds_low_res_shallow,
                'image_embeds_low_res_deep': image_embeds_low_res_deep,
                'image_embeds_high_res_deep': image_embeds_high_res_deep,
                'low_res': {
                    'dino': dino_low_deep,
                    'siglip': siglip_low_deep
                },
                'high_res': {
                    'dino': dino_high_deep,
                    'siglip': siglip_high_deep
                }
            }
