"""
PIP-InstantCharacter节点 - 为ComfyUI实现的InstantCharacter功能
完全按照原版InstantCharacter实现，包括多尺度特征提取和注意力处理机制
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

# 获取当前节点根目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将节点的父目录添加到系统路径 - 这样ComfyUI就可以找到节点包
sys.path.append(os.path.dirname(root_path))

# 使用包限定的绝对导入
from PIP_InstantCharacter.utils.tensor_utils import tensor2pil, pil2tensor, ensure_bhwc_format, ensure_bchw_format, resize_tensor
from PIP_InstantCharacter.utils.ip_adapter import InstantCharacterIPAdapter
from PIP_InstantCharacter.utils.feature_extractor import InstantCharacterFeatureExtractor

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
                "subject_scale": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ip_adapter_name": (["instantcharacter_ip-adapter.bin"] + ip_adapter_names,),
            },
            "optional": {
                "features": ("FEATURES",),  # 新增: 直接接收特征字典
                "DINOv2": ("CLIP_VISION",),  # 移到可选输入
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
        self.feature_extractor = InstantCharacterFeatureExtractor()
        self.model_config = {"model_type": "dummy"}
    
    def apply_instantcharacter(self, model, reference_image, subject_scale=0.9, 
                              ip_adapter_name="instantcharacter_ip-adapter.bin", 
                              features=None, DINOv2=None, SigLIP=None, low_memory_mode=False):
        """
        应用InstantCharacter到模型 - 完全按照原版InstantCharacter实现
        
        Args:
            model: 输入的FLUX/SD/SDXL模型（必须是ModelPatcher类型）
            reference_image: 参考图像（用于角色保留生成）
            DINOv2: DinoV2视觉模型
            subject_scale: 角色特征强度
            ip_adapter_name: IP适配器模型文件名
            SigLIP: SigLIP视觉模型（可选）
            low_memory_mode: 是否启用低内存模式
        """
        try:
            # 检查模型类型
            if not isinstance(model, ModelPatcher):
                model_type = type(model).__name__
                dinov2_type = type(DINOv2).__name__ if DINOv2 is not None else "None"
                
                # 检查是否误传了视觉模型
                model_name = getattr(model, 'model_name', str(type(model)))
                is_vision_model = any(name in str(model_name).lower() for name in ["dinov2", "siglip", "clip", "vision"])
                
                print(f"\n\n[PIP-InstantCharacter] ⚠️ 警告: 模型类型错误!")
                print(f"[PIP-InstantCharacter] - model输入端口收到的是: {model_type}")
                print(f"[PIP-InstantCharacter] - DINOv2输入端口收到的是: {dinov2_type}")
                print(f"[PIP-InstantCharacter] 请检查工作流连接:\n")
                print(f"   1. model端口应该连接SD/SDXL/Flux扩散模型(ModelPatcher类型)")
                print(f"   2. DINOv2端口应该连接DinoV2视觉模型")
                print(f"   3. SigLIP端口应该连接SigLIP视觉模型")
                
                return (model,)
            device = get_torch_device()
            print(f"[PIP-InstantCharacter] 正在处理推理请求，设备: {device}")
            
            # 检查模型是否有效
            if model is None or not isinstance(model, ModelPatcher):
                print("[PIP-InstantCharacter] 错误: 输入的模型不是ModelPatcher类型")
                return (model,)
            
            # 检查参考图像
            if reference_image is None:
                print("[PIP-InstantCharacter] 错误: 没有提供参考图像")
                return (model,)
            
            # 确保参考图像格式正确
            if isinstance(reference_image, list):
                reference_image = torch.cat(reference_image, dim=0)
            
            # 处理特征提取
            if features is not None:
                # 1. 如果已经提供了特征，直接使用
                print("[PIP-InstantCharacter] 检测到预提取的特征，直接使用")
            elif DINOv2 is None and SigLIP is None:
                # 2. 如果没有提供特征也没有视觉模型，报错
                print("[PIP-InstantCharacter] 错误: 无法提取特征，没有提供特征或视觉模型")
                return (model,)
            else:
                # 3. 使用视觉模型提取特征
                print("[PIP-InstantCharacter] 正在使用CLIP视觉模型提取图像特征...")
                try:
                    # 尝试使用两个模型提取特征
                    features = self.feature_extractor.extract_features(
                        dinov2_model=DINOv2,
                        siglip_model=SigLIP,
                        image=reference_image,
                        low_memory_mode=low_memory_mode
                    )
                except Exception as e:
                    print(f"[PIP-InstantCharacter] 特征提取错误: {e}")
                    print("[PIP-InstantCharacter] 尝试项目回退方案...")
                    try:
                        # 尝试只使用DinoV2提取特征
                        if DINOv2 is not None:
                            features = self.feature_extractor.extract_features(
                                dinov2_model=DINOv2,
                                siglip_model=None,
                                image=reference_image,
                                low_memory_mode=low_memory_mode
                            )
                        # 如果没有DinoV2，尝试只使用SigLIP
                        elif SigLIP is not None:
                            features = self.feature_extractor.extract_features(
                                dinov2_model=None,
                                siglip_model=SigLIP,
                                image=reference_image,
                                low_memory_mode=low_memory_mode
                            )
                        else:
                            raise RuntimeError(f"InstantCharacter特征提取失败，没有有效的视觉模型: {e}")
                    except Exception as e2:
                        print(f"[PIP-InstantCharacter] 最终尝试也失败: {e2}")
                        raise RuntimeError(f"InstantCharacter特征提取失败，无法继续处理: {e2}")
            
            # 检查特征字典
            if isinstance(features, dict) and len(features) > 0:
                feature_keys = list(features.keys())
                print(f"[PIP-InstantCharacter] 成功提取特征字典: {feature_keys}")
            else:
                print("[PIP-InstantCharacter] 警告: 特征字典为空或格式不正确")
            
            # 获取IP适配器路径
            try:
                adapter_path = folder_paths.get_full_path("ipadapter", ip_adapter_name)
                print(f"[PIP-InstantCharacter] 检测到IP适配器文件: {adapter_path}")
            except Exception as e:
                print(f"[PIP-InstantCharacter] 无法使用folder_paths找到IP适配器: {e}")
                
                # 如果无法通过ComfyUI路径找到，尝试在自定义路径中寻找
                adapter_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ipadapter", ip_adapter_name)
                if not os.path.exists(adapter_path):
                    # 如果在当前节点目录下没有，尝试ComfyUI顶层目录
                    adapter_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "ipadapter", ip_adapter_name)
                
                print(f"[PIP-InstantCharacter] 尝试使用路径: {adapter_path}")
                
                if not os.path.exists(adapter_path):
                    raise FileNotFoundError(f"IP适配器文件不存在: {ip_adapter_name}")
            
            # 创建IP适配器
            print(f"[PIP-InstantCharacter] 初始化IP适配器: {os.path.basename(adapter_path)}")
            adapter = InstantCharacterIPAdapter(adapter_path, device=device)
            
            # 检查特征是否有效
            if features is None or len(features) == 0:
                print("[PIP-InstantCharacter] 错误: 特征字典为空")
                return (model,)
            
            # 将InstantCharacter应用到模型 - 调用apply方法
            # 注意：IP适配器现在需要特征字典而不是单个特征向量
            print(f"[PIP-InstantCharacter] 应用InstantCharacter到模型，强度: {subject_scale}")
            try:
                patched_model = adapter.apply(model, features, scale=subject_scale)
            except Exception as e:
                print(f"[PIP-InstantCharacter] 应用特征到模型失败: {e}")
                traceback.print_exc()
                return (model,)
            
            # 清理内存
            try:
                if low_memory_mode and hasattr(self.feature_extractor, 'unload_models'):
                    self.feature_extractor.unload_models()
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[PIP-InstantCharacter] 清理内存时出错（非致命）: {e}")
                
            return (patched_model,)
            
        except Exception as e:
            print(f"[PIP-InstantCharacter] 应用InstantCharacter时出错: {str(e)}")
            traceback.print_exc()
            return (model,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "PIPInstantCharacter": PIPApplyInstantCharacter,
}

# 注册节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIPInstantCharacter": "PIP InstantCharacter 应用",
}
