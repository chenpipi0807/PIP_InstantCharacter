import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from einops import rearrange

class OriginalFeatureExtractor:
    """
    原版InstantCharacter特征提取器
    完全遵循原版InstantCharacter的特征提取逻辑
    """
    def __init__(self, model_loader=None):
        self.model_loader = model_loader
        self.siglip_model = None
        self.siglip_processor = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.initialized = False
    
    def initialize(self, model_loader):
        """初始化特征提取器，使用模型加载器提供的模型和处理器"""
        if model_loader is None:
            print("[PIP-InstantCharacter] 错误: 未提供模型加载器")
            return False
        
        self.model_loader = model_loader
        self.siglip_model, self.siglip_processor, self.dinov2_model, self.dinov2_processor = model_loader.get_models()
        
        if self.siglip_model is None or self.dinov2_model is None:
            print("[PIP-InstantCharacter] 错误: 模型未正确加载")
            return False
        
        self.initialized = True
        print("[PIP-InstantCharacter] 原版特征提取器初始化成功")
        return True
    
    def encode_siglip_image_emb(self, siglip_image, device="cuda", dtype=torch.float16):
        """
        提取SigLIP图像特征
        完全复制原版InstantCharacter的实现
        """
        if not self.initialized:
            print("[PIP-InstantCharacter] 错误: 特征提取器未初始化")
            return None, None
        
        try:
            # 确保图像在正确的设备和数据类型上
            siglip_image = siglip_image.to(device, dtype=dtype)
            
            # 提取特征，包括隐藏状态
            res = self.siglip_model(siglip_image, output_hidden_states=True)
            
            # 获取最后一层隐藏状态作为深层特征
            siglip_image_embeds = res.last_hidden_state
            
            # 按原版方式组合多层特征作为浅层特征
            # 原版InstantCharacter使用层7, 13, 26的特征
            # 但我们需要根据实际模型层数调整
            total_layers = len(res.hidden_states)
            print(f"[PIP-InstantCharacter] SigLIP模型有{total_layers}层隐藏状态")
            
            # 动态选择三个均匀分布的层作为浅层特征
            if total_layers >= 27:  # 原始InstantCharacter假设的层数
                layer_indices = [7, 13, 26]  # 原始索引
            else:
                # 如果层数较少，使用均匀分布的三层
                layer_indices = [
                    total_layers // 4,
                    total_layers // 2,
                    total_layers - 1
                ]
                print(f"[PIP-InstantCharacter] 自动调整SigLIP隐藏层索引为: {layer_indices}")
                
            siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in layer_indices], dim=1)
            
            return siglip_image_embeds, siglip_image_shallow_embeds
            
        except Exception as e:
            import traceback
            print(f"[PIP-InstantCharacter] SigLIP特征提取失败: {e}")
            traceback.print_exc()
            return None, None
    
    def encode_dinov2_image_emb(self, dinov2_image, device="cuda", dtype=torch.float16):
        """
        提取DinoV2图像特征
        完全复制原版InstantCharacter的实现
        """
        if not self.initialized:
            print("[PIP-InstantCharacter] 错误: 特征提取器未初始化")
            return None, None
        
        try:
            # 确保图像在正确的设备和数据类型上
            dinov2_image = dinov2_image.to(device, dtype=dtype)
            
            # 提取特征，包括隐藏状态
            res = self.dinov2_model(dinov2_image, output_hidden_states=True)
            
            # 获取最后一层隐藏状态作为深层特征，去除CLS token
            dinov2_image_embeds = res.last_hidden_state[:, 1:]
            
            # 按原版方式组合多层特征作为浅层特征
            # 原版InstantCharacter使用层9, 19, 29的特征
            # 但我们需要根据实际模型层数调整
            total_layers = len(res.hidden_states)
            print(f"[PIP-InstantCharacter] DinoV2模型有{total_layers}层隐藏状态")
            
            # 动态选择三个均匀分布的层作为浅层特征
            if total_layers >= 30:  # 原始InstantCharacter假设的层数
                layer_indices = [9, 19, 29]  # 原始索引
            else:
                # 如果层数较少，使用均匀分布的三层
                layer_indices = [
                    max(1, total_layers // 4),
                    max(1, total_layers // 2),
                    total_layers - 1
                ]
                print(f"[PIP-InstantCharacter] 自动调整DinoV2隐藏层索引为: {layer_indices}")
                
            dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in layer_indices], dim=1)
            
            return dinov2_image_embeds, dinov2_image_shallow_embeds
            
        except Exception as e:
            import traceback
            print(f"[PIP-InstantCharacter] DinoV2特征提取失败: {e}")
            traceback.print_exc()
            return None, None
    
    def prepare_multi_scale_images(self, image, low_res_size=384, high_res_size=768):
        """
        准备多尺度图像，按照原版InstantCharacter的实现
        1. 低分辨率图像 - 384x384
        2. 高分辨率图像 - 将图像分成4个区域处理
        """
        try:
            # 确保图像是PIL.Image
            if not isinstance(image, Image.Image):
                print("[PIP-InstantCharacter] 警告: 输入不是PIL图像，尝试转换")
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                elif isinstance(image, torch.Tensor):
                    if image.ndim == 4:
                        image = image.squeeze(0)
                    image = image.cpu().numpy().transpose(1, 2, 0)
                    image = Image.fromarray((image * 255).astype(np.uint8))
            
            # 记录原始图像尺寸
            original_width, original_height = image.size
            print(f"[PIP-InstantCharacter] 原始图像尺寸: {image.size}")
            
            # 1. 处理低分辨率图像 (384x384)
            # 使用SigLIP图像处理器处理图像
            low_res_image_pil = image.resize((low_res_size, low_res_size), Image.LANCZOS)
            
            # 2. 处理高分辨率图像 (768x768分成4个区域)
            # 首先调整图像尺寸
            high_res_image_pil = image.resize((high_res_size, high_res_size), Image.LANCZOS)
            
            # 将图像分成4个区域
            region_size = high_res_size // 2
            high_res_regions = []
            
            for y in range(2):
                for x in range(2):
                    left = x * region_size
                    top = y * region_size
                    right = left + region_size
                    bottom = top + region_size
                    
                    region = high_res_image_pil.crop((left, top, right, bottom))
                    high_res_regions.append(region)
            
            return {
                'low_res': low_res_image_pil,
                'high_res_regions': high_res_regions
            }
            
        except Exception as e:
            import traceback
            print(f"[PIP-InstantCharacter] 图像预处理失败: {e}")
            traceback.print_exc()
            return None
    
    def extract_features(self, image, device="cuda", dtype=torch.float16):
        """
        提取多尺度特征，完全按照原版InstantCharacter的实现
        
        参数:
            image: 输入图像 (PIL.Image)
            device: 计算设备
            dtype: 数据类型
        
        返回:
            包含不同级别特征的字典
        """
        if not self.initialized:
            print("[PIP-InstantCharacter] 错误: 特征提取器未初始化")
            return None
        
        try:
            print("[PIP-InstantCharacter] 开始提取原版多尺度特征...")
            
            # 1. 准备多尺度图像
            images_low_res = self.prepare_multi_scale_images(
                image=image,
                low_res_size=384,
                high_res_size=768
            )
            
            if images_low_res is None or 'low_res' not in images_low_res:
                print("[PIP-InstantCharacter] 错误: 图像预处理失败")
                return None
            
            print(f"[PIP-InstantCharacter] 图像预处理完成: 低分辨率(384x384)和高分辨率区域(768x768切分为4个384x384区域)")
            
            # 2. 提取低分辨率特征
            # 2.1 SigLIP低分辨率特征
            siglip_low_res_pixel_values = self.siglip_processor(images=images_low_res['low_res'], return_tensors="pt").pixel_values
            siglip_low_res_embeds = self.encode_siglip_image_emb(
                siglip_low_res_pixel_values,
                device=device,
                dtype=dtype
            )
            
            if siglip_low_res_embeds[0] is None:
                print("[PIP-InstantCharacter] 错误: SigLIP低分辨率特征提取失败")
                return None
            
            # 2.2 DinoV2低分辨率特征
            dinov2_low_res_pixel_values = self.dinov2_processor(images=images_low_res['low_res'], return_tensors="pt").pixel_values
            dinov2_low_res_embeds = self.encode_dinov2_image_emb(
                dinov2_low_res_pixel_values,
                device=device,
                dtype=dtype
            )
            
            if dinov2_low_res_embeds[0] is None:
                print("[PIP-InstantCharacter] 错误: DinoV2低分辨率特征提取失败")
                return None
            
            # 3. 合并低分辨率特征
            siglip_deep = siglip_low_res_embeds[0]
            dinov2_deep = dinov2_low_res_embeds[0]
            siglip_shallow = siglip_low_res_embeds[1]
            dinov2_shallow = dinov2_low_res_embeds[1]
            
            # 打印特征尺寸
            print(f"[PIP-InstantCharacter] SigLIP深层特征尺寸: {siglip_deep.shape}")
            print(f"[PIP-InstantCharacter] DinoV2深层特征尺寸: {dinov2_deep.shape}")
            print(f"[PIP-InstantCharacter] SigLIP浅层特征尺寸: {siglip_shallow.shape}")
            print(f"[PIP-InstantCharacter] DinoV2浅层特征尺寸: {dinov2_shallow.shape}")
            
            # 调整特征尺寸以确保匹配
            # 判断哪个尺寸更小，将更大的重新采样到更小的尺寸
            siglip_deep_len = siglip_deep.shape[1]
            dinov2_deep_len = dinov2_deep.shape[1]
            siglip_shallow_len = siglip_shallow.shape[1]
            dinov2_shallow_len = dinov2_shallow.shape[1]
            
            print(f"[PIP-InstantCharacter] 重新采样深层特征...")
            if siglip_deep_len != dinov2_deep_len:
                # 重新采样深层特征
                target_len = min(siglip_deep_len, dinov2_deep_len)
                if siglip_deep_len > target_len:
                    print(f"[PIP-InstantCharacter] 将SigLIP深层特征从{siglip_deep_len}采样到{target_len}")
                    siglip_deep = torch.nn.functional.interpolate(
                        siglip_deep.permute(0, 2, 1), size=target_len, mode='linear'
                    ).permute(0, 2, 1)
                if dinov2_deep_len > target_len:
                    print(f"[PIP-InstantCharacter] 将DinoV2深层特征从{dinov2_deep_len}采样到{target_len}")
                    dinov2_deep = torch.nn.functional.interpolate(
                        dinov2_deep.permute(0, 2, 1), size=target_len, mode='linear'
                    ).permute(0, 2, 1)
                print(f"[PIP-InstantCharacter] 采样后尺寸 - SigLIP: {siglip_deep.shape}, DinoV2: {dinov2_deep.shape}")
            
            print(f"[PIP-InstantCharacter] 重新采样浅层特征...")
            if siglip_shallow_len != dinov2_shallow_len:
                # 重新采样浅层特征
                target_len = min(siglip_shallow_len, dinov2_shallow_len)
                if siglip_shallow_len > target_len:
                    print(f"[PIP-InstantCharacter] 将SigLIP浅层特征从{siglip_shallow_len}采样到{target_len}")
                    siglip_shallow = torch.nn.functional.interpolate(
                        siglip_shallow.permute(0, 2, 1), size=target_len, mode='linear'
                    ).permute(0, 2, 1)
                if dinov2_shallow_len > target_len:
                    print(f"[PIP-InstantCharacter] 将DinoV2浅层特征从{dinov2_shallow_len}采样到{target_len}")
                    dinov2_shallow = torch.nn.functional.interpolate(
                        dinov2_shallow.permute(0, 2, 1), size=target_len, mode='linear'
                    ).permute(0, 2, 1)
                print(f"[PIP-InstantCharacter] 采样后尺寸 - SigLIP: {siglip_shallow.shape}, DinoV2: {dinov2_shallow.shape}")
            
            # 按原版方式合并特征
            image_embeds_low_res_deep = torch.cat([siglip_deep, dinov2_deep], dim=2)
            image_embeds_low_res_shallow = torch.cat([siglip_shallow, dinov2_shallow], dim=2)
            
            print(f"[PIP-InstantCharacter] 成功合并低分辨率特征，深层特征尺寸: {image_embeds_low_res_deep.shape}")
            
            # 4. 提取高分辨率区域特征
            high_res_embeds_list = []
            nb_split_image = len(images_low_res['high_res_regions'])
            
            # 4.1 处理SigLIP高分辨率区域特征
            siglip_high_res_pixel_values = [
                self.siglip_processor(images=region, return_tensors="pt").pixel_values
                for region in images_low_res['high_res_regions']
            ]
            
            # 将区域特征整合成一个批次
            siglip_high_res_pixel_values = torch.cat(siglip_high_res_pixel_values, dim=0)
            siglip_high_res_pixel_values = siglip_high_res_pixel_values.unsqueeze(0)  # 添加批次维度
            siglip_high_res_pixel_values = rearrange(siglip_high_res_pixel_values, 'b n c h w -> (b n) c h w')
            
            # 提取SigLIP高分辨率特征
            siglip_high_res_embeds = self.encode_siglip_image_emb(siglip_high_res_pixel_values, device, dtype)
            siglip_high_res_deep = rearrange(siglip_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
            
            # 4.2 处理DinoV2高分辨率区域特征
            dinov2_high_res_pixel_values = [
                self.dinov2_processor(images=region, return_tensors="pt").pixel_values
                for region in images_low_res['high_res_regions']
            ]
            
            # 将区域特征整合成一个批次
            dinov2_high_res_pixel_values = torch.cat(dinov2_high_res_pixel_values, dim=0)
            dinov2_high_res_pixel_values = dinov2_high_res_pixel_values.unsqueeze(0)  # 添加批次维度
            dinov2_high_res_pixel_values = rearrange(dinov2_high_res_pixel_values, 'b n c h w -> (b n) c h w')
            
            # 提取DinoV2高分辨率特征
            dinov2_high_res_embeds = self.encode_dinov2_image_emb(dinov2_high_res_pixel_values, device, dtype)
            dinov2_high_res_deep = rearrange(dinov2_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
            
            # 4.3 确保高分辨率特征尺寸匹配
            print(f"[PIP-InstantCharacter] 高分辨率特征尺寸 - SigLIP: {siglip_high_res_deep.shape}, DinoV2: {dinov2_high_res_deep.shape}")
            
            # 重新采样高分辨率特征以确保匹配
            siglip_high_res_len = siglip_high_res_deep.shape[1]
            dinov2_high_res_len = dinov2_high_res_deep.shape[1]
            
            print(f"[PIP-InstantCharacter] 重新采样高分辨率特征...")
            if siglip_high_res_len != dinov2_high_res_len:
                # 重新采样高分辨率特征
                target_len = min(siglip_high_res_len, dinov2_high_res_len)
                if siglip_high_res_len > target_len:
                    print(f"[PIP-InstantCharacter] 将SigLIP高分辨率特征从{siglip_high_res_len}采样到{target_len}")
                    siglip_high_res_deep = torch.nn.functional.interpolate(
                        siglip_high_res_deep.permute(0, 2, 1), size=target_len, mode='linear'
                    ).permute(0, 2, 1)
                if dinov2_high_res_len > target_len:
                    print(f"[PIP-InstantCharacter] 将DinoV2高分辨率特征从{dinov2_high_res_len}采样到{target_len}")
                    dinov2_high_res_deep = torch.nn.functional.interpolate(
                        dinov2_high_res_deep.permute(0, 2, 1), size=target_len, mode='linear'
                    ).permute(0, 2, 1)
                print(f"[PIP-InstantCharacter] 采样后尺寸 - SigLIP: {siglip_high_res_deep.shape}, DinoV2: {dinov2_high_res_deep.shape}")
                
            # 合并高分辨率特征
            image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov2_high_res_deep], dim=2)
            print(f"[PIP-InstantCharacter] 成功合并高分辨率区域特征，结果尺寸: {image_embeds_high_res_deep.shape}")
            
            # 5. 返回特征字典
            features = {
                'image_embeds_low_res_shallow': image_embeds_low_res_shallow,
                'image_embeds_low_res_deep': image_embeds_low_res_deep,
                'image_embeds_high_res_deep': image_embeds_high_res_deep,
                'deep_features': {
                    'low_res': image_embeds_low_res_deep,
                    'high_res': image_embeds_high_res_deep
                },
                'combined': image_embeds_low_res_deep  # 默认使用低分辨率深层特征
            }
            
            return features
            
        except Exception as e:
            import traceback
            print(f"[PIP-InstantCharacter] 特征提取失败: {e}")
            traceback.print_exc()
            return None
