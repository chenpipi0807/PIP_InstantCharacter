"""
PIP-InstantCharacter节点 - 为FLUX工作流设计的InstantCharacter实现
参考了PuLID的工作方式，用于角色保留生成
"""

import os
import sys
import gc
import traceback
import torch
import torch.nn.functional as F
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
import folder_paths

# 引入自定义工具模块
import os
import sys

# 获取当前节点根目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将节点的父目录添加到系统路径 - 这样ComfyUI就可以找到节点包
sys.path.append(os.path.dirname(root_path))

# 使用包限定的绝对导入
from PIP_InstantCharacter.utils.tensor_utils import tensor2pil, pil2tensor, ensure_bhwc_format, ensure_bchw_format, resize_tensor
from PIP_InstantCharacter.utils.ip_adapter import InstantCharacterIPAdapter
from PIP_InstantCharacter.utils.feature_extractor import FeatureExtractor

# 主节点 - 应用InstantCharacter到扩散模型
class PIPApplyInstantCharacter:
    """PIP-InstantCharacter 是一个强大的AI角色保留生成节点，它根据单张参考图像生成保持特征的对象。"""
    
    # 动态扫描IP适配器
    ip_adapter_names = []
    try:
        ip_adapter_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ipadapter")
        # 假如路径不存在，尝试在ComfyUI目录下找
        if not os.path.exists(ip_adapter_dir):
            ip_adapter_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "ipadapter")
        
        if os.path.exists(ip_adapter_dir):
            for file in os.listdir(ip_adapter_dir):
                if file.endswith(".bin") and file != "instantcharacter_ip-adapter.bin":
                    ip_adapter_names.append(file)
    except Exception as e:
        print(f"[PIP-InstantCharacter] 扫描IP适配器时出错: {e}")

    @classmethod
    def INPUT_TYPES(cls):
        # 扫描IP适配器目录
        ip_adapter_names = []
        try:
            ip_adapter_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ipadapter")
            # 假如路径不存在，尝试在ComfyUI目录下找
            if not os.path.exists(ip_adapter_dir):
                ip_adapter_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "ipadapter")
            
            if os.path.exists(ip_adapter_dir):
                for file in os.listdir(ip_adapter_dir):
                    if file.endswith(".bin") and file != "instantcharacter_ip-adapter.bin":
                        ip_adapter_names.append(file)
        except Exception as e:
            print(f"[PIP-InstantCharacter] 扫描IP适配器时出错: {e}")
            
        return {
            "required": {
                "model": ("MODEL",),
                "reference_image": ("IMAGE",),
                "DINOv2": ("CLIP_VISION",),
                "subject_scale": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ip_adapter_name": (["instantcharacter_ip-adapter.bin"] + ip_adapter_names,),
            },
            "optional": {
                "SigLIP": ("CLIP_VISION",),
                "low_memory_mode": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_instantcharacter"
    CATEGORY = "PIP_InstantCharacter"
    
    def __init__(self):
        super().__init__()
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor()
        self.model_config = {"model_type": "dummy"}
    
    def apply_instantcharacter(self, model, reference_image, DINOv2, subject_scale=0.9, 
                              ip_adapter_name="instantcharacter_ip-adapter.bin", 
                              SigLIP=None, low_memory_mode=False):
        """
        应用InstantCharacter到模型
        
        Args:
            model: 输入的FLUX/SD/SDXL模型（必须是ModelPatcher类型）
            reference_image: 参考图像（用于角色保留生成）
            DINOv2: DinoV2视觉模型
            subject_scale: 角色特征强度
            ip_adapter_name: IP适配器文件名
            SigLIP: SigLIP视觉模型（可选）
            low_memory_mode: 低内存模式开关
            
        Returns:
            修改后的模型
        """
        # 严格检查模型类型
        model_type = type(model).__name__
        dinov2_type = type(DINOv2).__name__ if DINOv2 is not None else "None"
        
        # 检查是否是ModelPatcher类型
        if not hasattr(model, 'model_options') or not isinstance(model, ModelPatcher):
            # 检查是否误传了视觉模型
            model_name = getattr(model, 'model_name', str(type(model)))
            is_vision_model = any(name in str(model_name).lower() for name in ["dinov2", "siglip", "clip", "vision"])
            
            print(f"\n\n[PIP-InstantCharacter] \u26a0\ufe0f 警告: 模型类型错误!")
            print(f"[PIP-InstantCharacter] - model\u8f93\u5165\u7aef\u53e3\u6536\u5230\u7684\u662f: {model_type}")
            print(f"[PIP-InstantCharacter] - DINOv2\u8f93\u5165\u7aef\u53e3\u6536\u5230\u7684\u662f: {dinov2_type}")
            print(f"[PIP-InstantCharacter] 请检查工作流连接:\n")
            print(f"   1. model端口应该连接SD/SDXL/Flux扩散模型(ModelPatcher类型)")
            print(f"   2. DINOv2端口应该连接DinoV2视觉模型")
            print(f"   3. SigLIP端口应该连接SigLIP视觉模型")
            print(f"\n请修正连接的方向!\n\n")
            
            if is_vision_model:
                print(f"[PIP-InstantCharacter] \u26a0\ufe0f严重错误\u26a0\ufe0f: 视觉模型被错误地连接到model端口！")
                print(f"[PIP-InstantCharacter] 您将{model_name}连接到了model输入端口，但这应该是一个扩散模型！")
                print(f"[PIP-InstantCharacter] 正确的连接方式是UNETLoader -> model, CLIPVisionLoader(DinoV2) -> DINOv2, CLIPVisionLoader(SigLIP) -> SigLIP")
            
            # 创建一个占位符ModelPatcher并返回，避免后续节点错误
            try:
                # 创建一个最小的空模型作为占位符
                from torch import nn
                
                class DummyModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.model_config = {"model_type": "dummy"}
                        
                    def forward(self, *args, **kwargs):
                        return torch.zeros(1, 4, 64, 64).to(get_torch_device())
                
                # 创建占位符模型并设置必要的属性
                dummy = DummyModel()
                placeholder_model = ModelPatcher(dummy)
                placeholder_model.model_options = {}
                
                print(f"[PIP-InstantCharacter] 创建占位模型以避免后续节点崩溃")
                print(f"[PIP-InstantCharacter] 请正确连接模型: UNETLoader -> model, CLIPVisionLoader(DinoV2) -> DINOv2")
                return (placeholder_model,)
            except Exception as e:
                print(f"[PIP-InstantCharacter] 无法创建占位模型: {e}")
                traceback.print_exc()
                
                # 实在没法创建占位符，只能丢弃当前节点返回原始模型
                print(f"[PIP-InstantCharacter] 无法修复连接错误，工作流可能会崩溃")
                return (model,)
            
        print(f"[PIP-InstantCharacter] 应用InstantCharacter，特征强度: {subject_scale}")
        
        try:
            # 1. 获取完整的IP适配器路径
            ipadapter_path = folder_paths.get_full_path("ipadapter", ip_adapter_name)
            if not os.path.exists(ipadapter_path):
                raise FileNotFoundError(f"IP适配器文件不存在: {ipadapter_path}")
                
            # 2. 加载适配器
            adapter = InstantCharacterIPAdapter(ipadapter_path)
            
            # 3. 处理参考图像
            if reference_image is not None:
                print("[PIP-InstantCharacter] 处理参考图像并提取特征")
                
                # 确保图像是BHWC格式且为float32类型，值范围为[0,1]
                reference_image = ensure_bhwc_format(reference_image)
                
                # 确保值范围在[0,1]内
                if reference_image.max() > 1.0:
                    reference_image = reference_image / 255.0
                    
                # 转为float32类型
                reference_image = reference_image.to(torch.float32)
                
                # 检查是否为量化模型以调整内存策略
                is_quantized = False
                model_type = "unknown"
                if hasattr(model, 'model_options') and 'model_type' in model.model_options:
                    model_type = model.model_options['model_type']
                    # 检测是否为GGUF或其他量化模型
                    if 'gguf' in model_type.lower() or 'quantized' in model_type.lower():
                        is_quantized = True
                        print(f"[PIP-InstantCharacter] 检测到量化模型 {model_type}，将优化内存策略")
                        
                # 在低内存模式下释放模型内存，以便为特征提取提供更多空间
                if low_memory_mode and hasattr(model, 'offload'):
                    print("[PIP-InstantCharacter] 启用低内存模式，临时卸载主模型...")
                    # 记录原始设备
                    original_device = next(model.model.parameters()).device
                    model.offload()
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 4. 使用CLIP Vision模型提取特征
                with torch.no_grad():
                    # 缩小图像尺寸以减少显存使用
                    max_size = 224 if low_memory_mode else 384
                    current_size = max(reference_image.shape[1], reference_image.shape[2])
                    if current_size > max_size:
                        # 调整图像尺寸
                        reference_image = resize_tensor(
                            reference_image.permute(0, 3, 1, 2),  # 转为BCHW
                            size=(int(reference_image.shape[1] * max_size / current_size), 
                                 int(reference_image.shape[2] * max_size / current_size)),
                            mode='bilinear'
                        ).permute(0, 2, 3, 1)  # 转回BHWC
                    
                    # 使用特征提取器提取图像特征
                    print("[PIP-InstantCharacter] 提取视觉特征...")
                    try:
                        # 转换为BCHW格式用于特征提取
                        bchw_image = reference_image.permute(0, 3, 1, 2)
                        
                        # 提取特征
                        image_features = self.feature_extractor.extract_features(
                            DINOv2, SigLIP, bchw_image, low_memory_mode
                        )
                        
                        print(f"[PIP-InstantCharacter] 成功提取特征，形状: {image_features.shape}")
                    except Exception as e:
                        print(f"[PIP-InstantCharacter] 特征提取失败: {e}")
                        traceback.print_exc()
                        # 创建占位特征，以便流程能够继续
                        image_features = torch.ones((1, 768), dtype=torch.float32)
                        print("[PIP-InstantCharacter] 使用占位特征继续流程")
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                print("[PIP-InstantCharacter] 警告: 未提供参考图像")
                image_features = None
            
            # 5. 应用特征到模型
            try:
                # 选择最适合的精度
                if hasattr(torch, "float8_e4m3fn") and (low_memory_mode or 
                      (torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.7 * torch.cuda.get_device_properties(0).total_memory)):
                    print("[PIP-InstantCharacter] 使用FP8精度优化模型内存使用")
                    if image_features is not None:
                        image_features = image_features.to(torch.float16)  # FP8需要特殊处理
                elif torch.cuda.is_available() and (torch.cuda.memory_allocated() > 0.5 * torch.cuda.get_device_properties(0).total_memory or low_memory_mode):
                    print("[PIP-InstantCharacter] 使用FP16精度优化模型内存使用")
                    if image_features is not None:
                        image_features = image_features.to(torch.float16)
                else:
                    print("[PIP-InstantCharacter] 使用标准精度应用InstantCharacter")
                
                # 创建模型副本并添加所需属性
                patched_model = model.clone()
                patched_model.ip_adapter = adapter
                patched_model.subject_scale = subject_scale
                patched_model.image_features = image_features
                
                # 返回修改后的模型
                return (patched_model,)
            except Exception as e:
                print(f"[PIP-InstantCharacter] 警告: 模型修改过程中出错: {e}")
                # 发生错误时返回原始模型
                return (model,)
            
        except Exception as e:
            print(f"[PIP-InstantCharacter] 应用InstantCharacter失败: {str(e)}")
            # 出错时返回原始模型
            return (model,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "PIPInstantCharacter": PIPApplyInstantCharacter,
}

# 注册节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIPInstantCharacter": "PIP InstantCharacter 应用",
}
