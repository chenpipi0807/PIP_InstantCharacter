import os
import torch
import sys
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor

class ModelLoader:
    """
    原版InstantCharacter模型加载器
    负责加载SigLIP和DinoV2模型及其处理器
    完全遵循原版InstantCharacter的实现，但优先使用ComfyUI的模型路径
    """
    def __init__(self):
        # 检测并设置ComfyUI环境
        self.comfy_path = self._find_comfyui_path()
        self.siglip_model = None
        self.siglip_processor = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.initialized = False
    
    def _find_comfyui_path(self):
        """找到ComfyUI根目录路径"""
        # 从当前目录向上查找
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 假设当前在custom_nodes/PIP_InstantCharacter/models目录下
        parent_dir = os.path.dirname(current_dir)  # PIP_InstantCharacter
        comfy_nodes_dir = os.path.dirname(parent_dir)  # custom_nodes
        comfy_dir = os.path.dirname(comfy_nodes_dir)  # ComfyUI根目录
        
        if os.path.exists(os.path.join(comfy_dir, "models")):
            return comfy_dir
        
        # 如果未找到，返回None
        print("[PIP-InstantCharacter] 警告: 未找到ComfyUI模型目录，将尝试从Hugging Face加载")
        return None
    
    def _get_model_path(self, model_type, model_name):
        """获取模型路径，优先使用ComfyUI标准路径
        
        参数:
            model_type: 模型类型 ('siglip' 或 'dinov2')
            model_name: 模型名称或Hugging Face ID
        """
        # 如果找到了ComfyUI路径
        if self.comfy_path:
            # 检查ComfyUI标准模型路径
            # clip_vision目录下的模型
            clip_vision_path = os.path.join(self.comfy_path, "models", "clip_vision")
            
            # 检查SigLIP模型
            if model_type.lower() == "siglip":
                # 检查标准SigLIP模型名称
                siglip_models = {
                    "siglip-base-patch16-384": "siglip_so400m.pth", 
                    "google/siglip-base-patch16-384": "siglip_so400m.pth"
                }
                
                if model_name in siglip_models:
                    local_path = os.path.join(clip_vision_path, siglip_models[model_name])
                    if os.path.exists(local_path):
                        print(f"[PIP-InstantCharacter] 使用ComfyUI本地SigLIP模型: {local_path}")
                        return local_path
            
            # 检查DinoV2模型
            elif model_type.lower() == "dinov2":
                # 检查标准DinoV2模型名称
                dinov2_models = {
                    "dinov2-base": "dinov2-vitl14.bin",
                    "facebook/dinov2-base": "dinov2-vitl14.bin"
                }
                
                if model_name in dinov2_models:
                    local_path = os.path.join(clip_vision_path, dinov2_models[model_name])
                    if os.path.exists(local_path):
                        print(f"[PIP-InstantCharacter] 使用ComfyUI本地DinoV2模型: {local_path}")
                        return local_path
        
        # 如果没有找到本地模型，使用Hugging Face ID
        print(f"[PIP-InstantCharacter] 未找到本地{model_type}模型，将从Hugging Face加载: {model_name}")
        return model_name
    
    def _load_local_model(self, model_path, model_type="vision", device="cuda", dtype=torch.float16):
        """从本地.bin或.pth文件加载模型
        
        这个函数处理单个模型文件的加载，而不是直接使用from_pretrained
        """
        if not os.path.exists(model_path):
            print(f"[PIP-InstantCharacter] 错误: 本地模型文件不存在: {model_path}")
            return None
            
        # 判断是哪种类型的模型
        is_siglip = "siglip" in model_path.lower()
        is_dinov2 = "dinov2" in model_path.lower()
        
        try:
            if is_siglip:
                # 使用Hugging Face ID获取配置，但加载本地模型文件
                config = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-384").config
                processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-384")
                
                # 加载模型权重
                print(f"[PIP-InstantCharacter] 从本地加载SigLIP模型: {model_path}")
                state_dict = torch.load(model_path, map_location="cpu")
                model = SiglipVisionModel(config)
                model.load_state_dict(state_dict, strict=False)
                
                # 移动到设备并设置为评估模式
                model = model.to(device, dtype=dtype)
                model.eval()
                
                return model, processor
                
            elif is_dinov2:
                # 使用Hugging Face ID获取配置，但加载本地模型文件
                config = AutoModel.from_pretrained("facebook/dinov2-base").config
                processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                
                # 设置处理器参数，与原版InstantCharacter保持一致
                processor.crop_size = {"height": 384, "width": 384}
                processor.size = {"shortest_edge": 384}
                
                # 加载模型权重
                print(f"[PIP-InstantCharacter] 从本地加载DinoV2模型: {model_path}")
                state_dict = torch.load(model_path, map_location="cpu")
                model = AutoModel.from_config(config)
                model.load_state_dict(state_dict, strict=False)
                
                # 移动到设备并设置为评估模式
                model = model.to(device, dtype=dtype)
                model.eval()
                
                return model, processor
            else:
                print(f"[PIP-InstantCharacter] 无法识别模型类型: {model_path}")
                return None, None
                
        except Exception as e:
            print(f"[PIP-InstantCharacter] 加载本地模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_models(self, siglip_model_id="google/siglip-base-patch16-384", dinov2_model_id="facebook/dinov2-base", device="cuda", dtype=torch.float16):
        """
        加载SigLIP和DinoV2模型及其处理器
        
        参数:
            siglip_model_id: SigLIP模型ID或本地路径
            dinov2_model_id: DinoV2模型ID或本地路径
            device: 计算设备
            dtype: 数据类型
        """
        try:
            print(f"[PIP-InstantCharacter] 开始加载SigLIP模型: {siglip_model_id}")
            siglip_model_path = self._get_model_path("siglip", siglip_model_id)
            
            # 判断是否是本地文件
            if os.path.isfile(siglip_model_path) and (siglip_model_path.endswith(".bin") or siglip_model_path.endswith(".pth")):
                # 如果是本地文件，使用自定义加载函数
                self.siglip_model, self.siglip_processor = self._load_local_model(siglip_model_path, "siglip", device, dtype)
                if self.siglip_model is None:
                    print(f"[PIP-InstantCharacter] 本地SigLIP模型加载失败，尝试仏Hugging Face加载")
                    self.siglip_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-384").to(device, dtype=dtype)
                    self.siglip_processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-384")
                    self.siglip_model.eval()
            else:
                # 否则使用from_pretrained
                self.siglip_model = SiglipVisionModel.from_pretrained(siglip_model_path)
                self.siglip_processor = SiglipImageProcessor.from_pretrained(siglip_model_path)
                
                # 将模型移动到指定设备和数据类型
                self.siglip_model = self.siglip_model.to(device, dtype=dtype)
                self.siglip_model.eval()
            
            print(f"[PIP-InstantCharacter] SigLIP模型加载成功")
            
            print(f"[PIP-InstantCharacter] 开始加载DinoV2模型: {dinov2_model_id}")
            dinov2_model_path = self._get_model_path("dinov2", dinov2_model_id)
            
            # 判断是否是本地文件
            if os.path.isfile(dinov2_model_path) and (dinov2_model_path.endswith(".bin") or dinov2_model_path.endswith(".pth")):
                # 如果是本地文件，使用自定义加载函数
                self.dinov2_model, self.dinov2_processor = self._load_local_model(dinov2_model_path, "dinov2", device, dtype)
                if self.dinov2_model is None:
                    print(f"[PIP-InstantCharacter] 本地DinoV2模型加载失败，尝试仏Hugging Face加载")
                    self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device, dtype=dtype)
                    self.dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                    
                    # 设置处理器参数，与原版InstantCharacter保持一致
                    self.dinov2_processor.crop_size = {"height": 384, "width": 384}
                    self.dinov2_processor.size = {"shortest_edge": 384}
                    
                    self.dinov2_model.eval()
            else:
                # 否则使用from_pretrained
                self.dinov2_model = AutoModel.from_pretrained(dinov2_model_path)
                self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_path)
                
                # 设置处理器参数，与原版InstantCharacter保持一致
                self.dinov2_processor.crop_size = {"height": 384, "width": 384}
                self.dinov2_processor.size = {"shortest_edge": 384}
                
                # 将模型移动到指定设备和数据类型
                self.dinov2_model = self.dinov2_model.to(device, dtype=dtype)
                self.dinov2_model.eval()
            
            print(f"[PIP-InstantCharacter] DinoV2模型加载成功")
            
            self.initialized = True
            return True
            
        except Exception as e:
            import traceback
            print(f"[PIP-InstantCharacter] 模型加载失败: {e}")
            traceback.print_exc()
            return False
    
    def get_models(self):
        """获取加载的模型和处理器"""
        if not self.initialized:
            print("[PIP-InstantCharacter] 警告: 模型尚未初始化，请先调用load_models方法")
            return None, None, None, None
        
        return self.siglip_model, self.siglip_processor, self.dinov2_model, self.dinov2_processor
