import os
import sys
import torch
import traceback
from PIL import Image
import numpy as np

# 确保能找到自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入模型加载器和特征提取器
from models.model_loader import ModelLoader
from models.feature_extractor_original import OriginalFeatureExtractor

class OriginalInstantCharacterExtractor:
    """
    原版InstantCharacter特征提取器节点
    完全按照原版InstantCharacter实现特征提取逻辑
    """
    
    def __init__(self):
        self.model_loader = None
        self.feature_extractor = None
        self.initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["cuda", "cpu"],),
            },
            "optional": {
                "siglip_model_id": ("STRING", {"default": "google/siglip-base-patch16-384"}),
                "dinov2_model_id": ("STRING", {"default": "facebook/dinov2-base"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FEATURES")
    RETURN_NAMES = ("image", "features")
    FUNCTION = "extract_features"
    CATEGORY = "IP-Adapter/InstantCharacter"
    
    def initialize_models(self, siglip_model_id, dinov2_model_id, device="cuda"):
        """初始化模型和特征提取器"""
        try:
            # 检查是否已初始化
            if self.initialized:
                return True
                
            # 创建模型加载器
            if self.model_loader is None:
                self.model_loader = ModelLoader()
            
            # 选择设备
            self.device = device
            self.dtype = torch.float16 if device == "cuda" else torch.float32
            
            # 加载模型
            success = self.model_loader.load_models(
                siglip_model_id=siglip_model_id,
                dinov2_model_id=dinov2_model_id,
                device=device,
                dtype=self.dtype
            )
            
            if not success:
                print("[PIP-InstantCharacter] 错误: 模型加载失败")
                return False
            
            # 创建特征提取器
            if self.feature_extractor is None:
                self.feature_extractor = OriginalFeatureExtractor()
            
            # 初始化特征提取器
            success = self.feature_extractor.initialize(self.model_loader)
            if not success:
                print("[PIP-InstantCharacter] 错误: 特征提取器初始化失败")
                return False
            
            self.initialized = True
            print("[PIP-InstantCharacter] 原版特征提取器节点初始化完成")
            return True
            
        except Exception as e:
            print(f"[PIP-InstantCharacter] 初始化失败: {e}")
            traceback.print_exc()
            return False
    
    def extract_features(self, image, device="cuda", siglip_model_id="google/siglip-base-patch16-384", dinov2_model_id="facebook/dinov2-base"):
        """提取图像特征"""
        try:
            # 初始化模型
            if not self.initialized:
                success = self.initialize_models(siglip_model_id, dinov2_model_id, device)
                if not success:
                    return (image, None)
            
            # 如果图像是张量，转换为PIL图像
            if isinstance(image, torch.Tensor):
                # 选择第一张图像如果是批次
                if len(image.shape) == 4:
                    image_tensor = image[0]
                else:
                    image_tensor = image
                
                # 将张量转换为PIL图像 - 改进的处理方式
                try:
                    # 格式转换：[C, H, W] -> [H, W, C]
                    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                    
                    # 缩放到 0-255 范围
                    image_np = (image_np * 255).astype(np.uint8)
                    
                    # 确保图像是 3 通道的 RGB
                    if image_np.shape[2] == 1:  # 如果是单通道图像
                        image_np = np.repeat(image_np, 3, axis=2)
                    elif image_np.shape[2] > 3:  # 如果通道超过 3
                        image_np = image_np[:, :, :3]  # 只保留前 3 个通道
                    
                    pil_image = Image.fromarray(image_np)
                except Exception as e:
                    print(f"[PIP-InstantCharacter] 图像转换错误，尝试备用方法: {e}")
                    
                    # 备用方法：创建一个新的空白 RGB 图像
                    # 获取原始尺寸
                    if len(image_tensor.shape) >= 2:
                        h, w = image_tensor.shape[1], image_tensor.shape[2]
                    else:
                        h, w = 512, 512  # 默认尺寸
                    
                    pil_image = Image.new('RGB', (w, h), (255, 255, 255))
            else:
                pil_image = image
            
            # 提取特征
            features = self.feature_extractor.extract_features(
                image=pil_image,
                device=self.device,
                dtype=self.dtype
            )
            
            if features is None:
                print("[PIP-InstantCharacter] 特征提取失败")
                return (image, None)
            
            print("[PIP-InstantCharacter] 特征提取成功")
            return (image, features)
            
        except Exception as e:
            print(f"[PIP-InstantCharacter] 特征提取过程中出错: {e}")
            traceback.print_exc()
            return (image, None)

# 节点映射 - 用于ComfyUI注册
NODE_CLASS_MAPPINGS = {
    "PIPOriginalInstantCharacterExtractor": OriginalInstantCharacterExtractor
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIPOriginalInstantCharacterExtractor": "PIP InstantCharacter Extractor (Original)"
}
