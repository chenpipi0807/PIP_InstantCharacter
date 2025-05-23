import os
import sys
import torch
from PIL import Image
import numpy as np
import traceback

# 添加当前目录到系统路径，确保能找到自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(f"添加路径: {current_dir}")

# 导入ComfyUI路径
comfy_dir = os.path.join(os.path.dirname(current_dir), "..")
if comfy_dir not in sys.path:
    sys.path.append(comfy_dir)
    print(f"添加ComfyUI路径: {comfy_dir}")

# 尝试导入原版特征提取器和模型加载器
try:
    from models.model_loader import ModelLoader
    from models.feature_extractor_original import OriginalFeatureExtractor
    print("成功导入原版模型加载器和特征提取器")
except ImportError as e:
    print(f"导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_original_extractor():
    print("\n=== 测试原版特征提取器 ===")
    
    # 创建模型加载器
    model_loader = ModelLoader()
    
    # 加载模型（使用CPU以便在任何环境中运行）
    # 在生产环境中可使用GPU加速: device="cuda"
    success = model_loader.load_models(
        siglip_model_id="google/siglip-base-patch16-384",
        dinov2_model_id="facebook/dinov2-base",
        device="cpu",
        dtype=torch.float32  # 使用float32以便在CPU上运行更快
    )
    
    if not success:
        print("模型加载失败，测试终止")
        return
    
    # 创建原版特征提取器
    extractor = OriginalFeatureExtractor()
    extractor.initialize(model_loader)
    
    # 测试图像路径
    image_path = os.path.join(current_dir, "test_images", "test_image.jpg")
    
    # 如果测试图像不存在，创建一个简单的测试图像
    if not os.path.exists(image_path):
        # 创建测试图像目录
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # 创建一个测试图像
        test_image = Image.new("RGB", (512, 512), color=(255, 255, 255))
        # 添加一些简单的内容
        for i in range(0, 512, 32):
            for j in range(0, 512, 32):
                color = (i % 256, j % 256, (i + j) % 256)
                for x in range(i, i + 16):
                    for y in range(j, j + 16):
                        if x < 512 and y < 512:
                            test_image.putpixel((x, y), color)
        
        test_image.save(image_path)
        print(f"创建测试图像: {image_path}")
    
    # 加载测试图像
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"加载测试图像: {image_path}, 尺寸: {image.size}")
    except Exception as e:
        print(f"图像加载失败: {e}")
        return
    
    # 提取特征
    print("开始提取特征...")
    features = extractor.extract_features(
        image=image,
        device="cpu",
        dtype=torch.float32
    )
    
    # 检查特征
    if features is None:
        print("特征提取失败")
        return
    
    # 打印特征信息
    print("特征提取成功，特征键:", features.keys())
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key} 形状:", value.shape)
        elif isinstance(value, dict):
            print(f"  {key} 内容:", value.keys())
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    print("当前目录:", os.getcwd())
    print("Python路径:", sys.path)
    try:
        test_original_extractor()
    except Exception as e:
        print(f"测试过程中出错: {e}")
        traceback.print_exc()
