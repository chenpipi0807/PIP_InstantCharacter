"""
测试多尺度处理器和特征提取器的兼容性
"""

import os
import sys
import torch
from PIL import Image

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# 添加父目录到路径 (ComfyUI路径)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 尝试导入时显示详细路径信息
print(f"当前目录: {current_dir}")
print(f"Python路径: {sys.path}")

# 直接使用相对导入
try:
    # 导入多尺度处理器和特征提取器
    from utils.multi_scale_processor import MultiScaleProcessor
    from utils.feature_extractor import InstantCharacterFeatureExtractor
    print("成功导入模块")
except Exception as e:
    print(f"导入错误: {e}")
    # 尝试绝对导入
    try:
        from PIP_InstantCharacter.utils.multi_scale_processor import MultiScaleProcessor
        from PIP_InstantCharacter.utils.feature_extractor import InstantCharacterFeatureExtractor
        print("成功使用绝对路径导入模块")
    except Exception as e:
        print(f"绝对导入也失败: {e}")
        # 尝试直接导入
        sys.path.append(os.path.join(current_dir, 'utils'))
        try:
            from multi_scale_processor import MultiScaleProcessor
            from feature_extractor import InstantCharacterFeatureExtractor
            print("成功使用直接导入")
        except Exception as e:
            print(f"所有导入方式都失败: {e}")
            raise

def test_multi_scale_processor():
    """测试多尺度处理器"""
    print("=== 测试多尺度处理器 ===")
    
    # 加载测试图像
    test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.png")
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return
        
    test_image = Image.open(test_image_path)
    print(f"测试图像尺寸: {test_image.size}")
    
    # 创建多尺度处理器
    processor = MultiScaleProcessor(device="cpu")
    
    # 测试prepare_multi_scale_images方法
    result = processor.prepare_multi_scale_images(test_image)
    
    if result is None:
        print("多尺度图像处理失败")
        return
        
    print(f"处理结果类型: {type(result)}")
    print(f"处理结果键: {result.keys()}")
    print(f"低分辨率图像尺寸: {result['low_res'].size}")
    print(f"高分辨率区域数量: {len(result['high_res_regions'])}")
    for i, region in enumerate(result['high_res_regions']):
        print(f"  区域 {i+1} 尺寸: {region.size}")
    
    # 测试编码方法 (假设有一个虚拟模型)
    class DummyModel:
        def __init__(self):
            self.name = "DummyModel"
            
        def encode_image(self, image):
            # 返回一个虚拟特征张量
            return torch.randn(1, 768)
            
        def __call__(self, pixel_values, output_hidden_states=False):
            # 模拟模型调用
            class ModelOutput:
                def __init__(self):
                    self.last_hidden_state = torch.randn(1, 16, 768)
                    self.hidden_states = [torch.randn(1, 16, 768) for _ in range(30)]
            return ModelOutput()
    
    dummy_model = DummyModel()
    
    print("\n测试SigLIP编码...")
    siglip_embeds = processor.encode_siglip_image(
        siglip_processor=None,
        siglip_model=dummy_model,
        images=result['low_res'],
        device="cpu"
    )
    
    if siglip_embeds[0] is not None:
        print(f"SigLIP特征形状: {siglip_embeds[0].shape}")
    else:
        print("SigLIP特征提取失败")
    
    print("\n测试DinoV2编码...")
    dino_embeds = processor.encode_dinov2_image(
        dino_processor=None,
        dino_model=dummy_model,
        images=result['low_res'],
        device="cpu"
    )
    
    if dino_embeds[0] is not None:
        print(f"DinoV2特征形状: {dino_embeds[0].shape}")
    else:
        print("DinoV2特征提取失败")
    
    print("\n测试多尺度特征提取...")
    features = processor.encode_multi_scale_features(
        siglip_processor=None,
        siglip_model=dummy_model,
        dino_processor=None,
        dino_model=dummy_model,
        image=test_image,
        device="cpu"
    )
    
    if features is not None:
        print(f"多尺度特征键: {features.keys()}")
        for key, value in features.items():
            print(f"  {key} 形状: {value.shape}")
    else:
        print("多尺度特征提取失败")
    
def test_feature_extractor():
    """测试特征提取器"""
    print("\n=== 测试特征提取器 ===")
    
    # 加载测试图像
    test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.png")
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return
        
    test_image = Image.open(test_image_path)
    
    # 创建特征提取器
    extractor = InstantCharacterFeatureExtractor(device="cpu")
    
    # 创建虚拟模型
    class DummyModel:
        def __init__(self):
            self.name = "DummyModel"
            
        def encode_image(self, image):
            # 返回一个虚拟特征张量
            return torch.randn(1, 768)
            
        def __call__(self, pixel_values, output_hidden_states=False):
            # 模拟模型调用
            class ModelOutput:
                def __init__(self):
                    self.last_hidden_state = torch.randn(1, 16, 768)
                    self.hidden_states = [torch.randn(1, 16, 768) for _ in range(30)]
            return ModelOutput()
    
    dummy_model = DummyModel()
    
    # 测试特征提取
    print("测试特征提取...")
    features = extractor.extract_features(
        dinov2_model=dummy_model,
        siglip_model=dummy_model,
        image=test_image
    )
    
    if features is not None and len(features) > 0:
        print(f"特征提取成功，特征键: {features.keys()}")
        
        for key in features:
            if isinstance(features[key], torch.Tensor):
                print(f"  {key} 形状: {features[key].shape}")
            elif isinstance(features[key], dict):
                print(f"  {key} 内容: {features[key].keys()}")
    else:
        print("特征提取失败")

if __name__ == "__main__":
    try:
        test_multi_scale_processor()
        test_feature_extractor()
        print("\n所有测试完成!")
    except Exception as e:
        import traceback
        print(f"测试过程中出错: {e}")
        traceback.print_exc()
