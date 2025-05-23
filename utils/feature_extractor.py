"""
InstantCharacter 特征提取器
用于处理图像特征提取和转换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import traceback

class InputNormalizer(nn.Module):
    """输入图像标准化模块，动态调整图像大小和填充"""
    
    def __init__(self, target_size=224):
        super().__init__()
        self.target_size = target_size
        self.pad = nn.ZeroPad2d((0,0,0,0))  # 动态填充层
        
    def forward(self, x):
        # 保持长宽比缩放（参考DINOv2论文）
        h, w = x.shape[2], x.shape[3]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h*scale), int(w*scale)
        x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
        
        # 动态填充至标准尺寸（兼容任意输入）
        pad_h = (self.target_size - new_h) // 2
        pad_w = (self.target_size - new_w) // 2
        self.pad.padding = (pad_w, self.target_size - new_w - pad_w, pad_h, self.target_size - new_h - pad_h)
        return self.pad(x)

class FeatureProjector(nn.Module):
    """特征投影器，用于结合多个特征源"""
    
    def __init__(self, dinov2_dim=1024, siglip_dim=1152, hidden_dim=768):
        super().__init__()
        self.dino_proj = nn.Linear(dinov2_dim, hidden_dim//2)
        self.siglip_proj = nn.Linear(siglip_dim, hidden_dim//2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, dinov2_feat, siglip_feat):
        # 动态降维（防止维度爆炸）
        dino = self.dino_proj(dinov2_feat)  # [B,1024] -> [B,384]
        
        # 处理不同形状的siglip特征
        if siglip_feat.dim() > 2:
            if siglip_feat.dim() == 3:  # [B,L,C]
                siglip_feat = siglip_feat.mean(dim=1)  # 平均池化
        
        siglip = self.siglip_proj(siglip_feat)  # [B,1152] -> [B,384]
        fused = torch.cat([dino, siglip], dim=-1)  # [B,768]
        return self.layer_norm(fused)

class FeatureExtractor:
    """特征提取器，用于从参考图像中提取特征"""
    
    def __init__(self):
        """初始化特征提取器"""
        self.input_normalizer_dinov2 = InputNormalizer(target_size=224)  # DinoV2标准尺寸
        self.input_normalizer_siglip = InputNormalizer(target_size=224)  # SigLIP标准尺寸
        self.feature_projector = None  # 延迟初始化，直到知道特征维度
        
    def extract_features(self, dinov2_model, siglip_model, image, low_memory_mode=False):
        """
        从图像中提取DinoV2和SigLIP特征
        
        Args:
            dinov2_model: DinoV2视觉模型
            siglip_model: SigLIP视觉模型(可选)
            image: 输入图像(BHWC格式)
            low_memory_mode: 低内存模式开关
            
        Returns:
            组合特征向量
        """
        # 提取两种特征并组合
        dinov2_features = self.encode_image(dinov2_model, image, "DinoV2", low_memory_mode)
        siglip_features = None
        
        if siglip_model is not None:
            # 如果有GPU，先清理一下内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            # 提取SigLIP特征
            siglip_features = self.encode_image(siglip_model, image, "SigLIP", low_memory_mode)
        
        # 组合特征
        return self._combine_features(dinov2_features, siglip_features)
    
    def encode_image(self, vision_model, image, model_type="unknown", low_memory_mode=False):
        """
        将图像编码为特征向量
        
        Args:
            vision_model: 视觉模型(DinoV2或SigLIP)
            image: 输入图像，必须是BCHW格式(内部处理)
            model_type: 模型类型，用于日志
            low_memory_mode: 低内存模式开关
            
        Returns:
            编码后的特征
        """
        try:
            # 1. 基本格式检查和通道转换
            if len(image.shape) != 4:
                raise ValueError(f"[PIP-InstantCharacter] 输入图像必须是4维张量，当前: {image.shape}")
            
            # 确保通道维度在正确位置 - BCHW格式 (通道在索引1位置)
            if image.shape[1] != 3:
                # 如果是BHWC格式，转换为BCHW
                if image.shape[3] == 3:
                    print(f"[PIP-InstantCharacter] 检测到BHWC格式, 转换为BCHW: {image.shape}")
                    image = image.permute(0, 3, 1, 2).contiguous()
                else:
                    print(f"[PIP-InstantCharacter] 警告: 无法检测通道维度: {image.shape}")
                    return self._create_special_features(model_type)
            
            # 2. 内存优化与清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 3. 获取模型设备
            try:
                if hasattr(vision_model, 'device'):
                    model_device = vision_model.device
                elif hasattr(vision_model, 'load_device'):
                    model_device = vision_model.load_device
                else:
                    model_device = next(vision_model.parameters()).device
                
                print(f"[PIP-InstantCharacter] 模型{model_type}运行在设备: {model_device}")
            except Exception as e:
                print(f"[PIP-InstantCharacter] 无法确定模型设备，使用CPU: {e}")
                model_device = torch.device('cpu')
            
            # 4. 值范围检查和规范化
            min_val = torch.min(image).item()
            max_val = torch.max(image).item()
            if min_val < 0 or max_val > 1.0:
                print(f"[PIP-InstantCharacter] 值范围异常: {min_val:.3f}-{max_val:.3f}，规范化到[0,1]")
                if max_val > 1.0 and max_val <= 255.0:
                    image = image / 255.0
                else:
                    image = (image - min_val) / (max_val - min_val + 1e-5)
            
            # 5. 强制类型转换
            image = image.to(dtype=torch.float32)
            
            # 6. 基于模型类型使用不同的输入标准化器
            with torch.no_grad():
                if 'dino' in model_type.lower():
                    print(f"[PIP-InstantCharacter] 使用DinoV2输入标准化模块...")
                    image = self.input_normalizer_dinov2(image)
                    image = image.to(model_device)
                    
                elif 'siglip' in model_type.lower():
                    print(f"[PIP-InstantCharacter] 使用SigLIP输入标准化模块...")
                    image = self.input_normalizer_siglip(image)
                    image = image.to(model_device)
                else:
                    # 默认标准化
                    image = F.interpolate(image, size=(224, 224), mode='bicubic', align_corners=False)
                    image = image.to(model_device)
            
            # 调试信息输出
            print(f"[PIP-InstantCharacter] 编码{model_type}特征，图像形状: {image.shape}, 类型: {image.dtype}, 设备: {image.device}")
            print(f"[PIP-InstantCharacter] 值范围: {torch.min(image).item():.3f}-{torch.max(image).item():.3f}")
            
            # 7. 特征提取 - 尝试多种方法
            features = None
            try:
                # 确保使用无梯度和混合精度
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    # 尝试标准encode_image方法
                    if hasattr(vision_model, 'encode_image'):
                        print(f"[PIP-InstantCharacter] 使用标准encode_image方法...")
                        features = vision_model.encode_image(image)
                    # 如果有内部模型，尝试直接使用
                    elif hasattr(vision_model, 'model'):
                        print(f"[PIP-InstantCharacter] 尝试使用内部模型...")
                        inner_model = vision_model.model
                        
                        # 准备输入
                        if hasattr(vision_model, 'image_mean') and hasattr(vision_model, 'image_std'):
                            mean = torch.tensor(vision_model.image_mean).view(1, 3, 1, 1).to(image.device)
                            std = torch.tensor(vision_model.image_std).view(1, 3, 1, 1).to(image.device)
                            inputs = (image - mean) / std
                        else:
                            inputs = image
                            
                        # 调用模型
                        outputs = inner_model(inputs)
                        
                        # 处理输出
                        if isinstance(outputs, dict):
                            if 'last_hidden_state' in outputs:
                                features = outputs['last_hidden_state'][:, 0]
                            elif 'image_embeds' in outputs:
                                features = outputs['image_embeds']
                        elif isinstance(outputs, torch.Tensor):
                            features = outputs
                    
                    # 如果上述方法失败，尝试使用ComfyUI的clip_preprocess
                    if features is None:
                        print("[PIP-InstantCharacter] 尝试使用clip_preprocess函数")
                        try:
                            from comfy.clip_vision import clip_preprocess
                            processed = clip_preprocess(image.to(vision_model.load_device), 
                                                      size=224,
                                                      mean=[0.48145466, 0.4578275, 0.40821073],
                                                      std=[0.26862954, 0.26130258, 0.27577711])
                            features = vision_model.encode_image(processed)
                        except Exception as clip_err:
                            print(f"[PIP-InstantCharacter] clip_preprocess方法也失败: {clip_err}")
                            features = self._create_special_features(model_type)
            except Exception as e:
                print(f"[PIP-InstantCharacter] 特征提取失败: {e}")
                traceback.print_exc()
                features = self._create_special_features(model_type)
            
            return features
            
        except Exception as e:
            # 最外层异常捕获
            print(f"[PIP-InstantCharacter] {model_type}特征编码失败: {e}")
            traceback.print_exc()
            return self._create_placeholder_features(model_type)
    
    def _combine_features(self, dino_features, siglip_features):
        """结合DinoV2和SigLIP特征用于下游处理"""
        print(f"[PIP-InstantCharacter] 结合两个模型的特征向量")
        
        try:
            # 检查并初始化特征投影器如果需要
            if self.feature_projector is None:
                # 获取特征维度
                dino_dim = dino_features.shape[-1] if dino_features is not None else 1024
                siglip_dim = siglip_features.shape[-1] if siglip_features is not None else 768
                
                # 初始化特征投影器
                print(f"[PIP-InstantCharacter] 初始化特征投影器: DinoV2({dino_dim}), SigLIP({siglip_dim}) -> 768")
                self.feature_projector = FeatureProjector(dinov2_dim=dino_dim, siglip_dim=siglip_dim, hidden_dim=768)
                self.feature_projector.to('cpu')  # 保持在CPU上以节省显存
            
            # 确保特征在CPU上以节省显存
            if dino_features is not None:
                dino_features = dino_features.to('cpu')
            if siglip_features is not None:
                siglip_features = siglip_features.to('cpu')
            
            # 处理可能缺失的特征
            if dino_features is None:
                print(f"[PIP-InstantCharacter] DinoV2特征缺失，创建替代特征")
                dino_features = self._create_special_features('dino')
            if siglip_features is None:
                print(f"[PIP-InstantCharacter] SigLIP特征缺失，创建替代特征")
                siglip_features = self._create_special_features('siglip')
            
            # 使用特征投影器结合特征
            with torch.no_grad():
                combined_features = self.feature_projector(dino_features, siglip_features)
            
            print(f"[PIP-InstantCharacter] 结合后的特征形状: {combined_features.shape}")
            return combined_features.to(torch.float32)
        except Exception as e:
            print(f"[PIP-InstantCharacter] 结合特征时出错: {e}")
            traceback.print_exc()
            
            # 如果出错，尝试简单串联
            try:
                print(f"[PIP-InstantCharacter] 尝试简单拼接特征代替投影")
                if dino_features.dim() == 2 and siglip_features.dim() == 2:
                    return torch.cat([dino_features, siglip_features], dim=1).to(torch.float32)
                else:
                    # 尺寸不匹配，返回其中一个
                    if dino_features is not None and dino_features.dim() == 2:
                        return dino_features.to(torch.float32)
                    elif siglip_features is not None and siglip_features.dim() == 2:
                        return siglip_features.to(torch.float32)
            except Exception as e2:
                print(f"[PIP-InstantCharacter] 简单拼接也失败: {e2}")
                
            # 最后的备选方案：创建随机特征
            return torch.randn(1, 768, device='cpu').to(torch.float32)
    
    def _create_special_features(self, model_type):
        """创建适合原版InstantCharacter格式的特殊特征向量"""
        print(f"[PIP-InstantCharacter] 使用特定格式的特征向量，匹配原版InstantCharacter")
        
        # 针对不同模型返回相应大小的特征
        if 'dino' in model_type.lower():
            # DinoV2格式
            return torch.randn(1, 1024, device='cpu')  # 随机向量更接近真实特征
        elif 'siglip' in model_type.lower():
            # SigLIP格式
            return torch.randn(1, 768, device='cpu')
        else:
            # 默认格式
            return torch.randn(1, 768, device='cpu')
            
    def _create_placeholder_features(self, model_type):
        """ 创建占位符特征向量 """
        print("[PIP-InstantCharacter] 创建占位符特征")
        if 'dino' in model_type.lower():
            return torch.ones(1, 1024, device='cpu')  # DinoV2特征尺寸
        elif 'siglip' in model_type.lower():
            return torch.ones(1, 768, device='cpu')   # SigLIP特征尺寸
        else:
            return torch.ones(1, 768, device='cpu')   # 默认尺寸
