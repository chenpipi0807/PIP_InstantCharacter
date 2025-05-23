"""
InstantCharacter 张量处理工具
提供各种张量转换和图像处理函数
"""

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

def tensor2pil(tensor):
    """ 
    将张量转换为PIL图像 (BHWC -> PIL)
    
    Args:
        tensor: BHWC格式的张量，值范围[0,1]
        
    Returns:
        PIL图像
    """
    # 先确保数据类型为float32
    tensor = tensor.cpu().detach().float()
    
    # 如果是批次图像，取第一张
    if len(tensor.shape) == 4:
        tensor = tensor[0]
        
    # 将数值范围限制在0-1之间
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # 如果是单通道图像(掩码)，转换为L模式
    if tensor.shape[2] == 1:
        array = tensor.squeeze().numpy() * 255.0
        return Image.fromarray(array.astype(np.uint8), mode='L')
    else:
        # 将张量转为numpy数组，并转换为0-255范围
        array = tensor.numpy() * 255.0
        return Image.fromarray(array.astype(np.uint8), mode='RGB')

def pil2tensor(image):
    """ 
    将PIL图像转换为张量 (PIL -> BHWC) 
    
    Args:
        image: PIL图像
        
    Returns:
        BHWC格式的张量，值范围[0,1]
    """
    # 判断格式
    if image.mode == 'L':
        # 单通道图像
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.unsqueeze(2)  # 添加通道维度
    else:
        # RGB图像
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(2)  # 添加通道维度
        if tensor.shape[2] == 4:  # RGBA图像
            tensor = tensor[:, :, :3]  # 去除alpha通道
    
    # 添加batch维度
    return tensor.unsqueeze(0)

def ensure_bhwc_format(tensor):
    """
    确保张量是BHWC格式
    
    Args:
        tensor: 输入张量
        
    Returns:
        BHWC格式的张量
    """
    # 如果是HWC格式(3维)，添加批次维度
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
        
    # 如果是BCHW格式，转换为BHWC
    if len(tensor.shape) == 4 and tensor.shape[1] == 3:
        tensor = tensor.permute(0, 2, 3, 1)
        
    return tensor

def ensure_bchw_format(tensor):
    """
    确保张量是BCHW格式
    
    Args:
        tensor: 输入张量
        
    Returns:
        BCHW格式的张量
    """
    # 如果是HWC格式(3维)，添加批次维度并转换通道
    if len(tensor.shape) == 3:
        if tensor.shape[2] == 3:  # HWC
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # 转换为BCHW
        else:  # CHW
            tensor = tensor.unsqueeze(0)  # 添加批次维度
            
    # 如果是BHWC格式，转换为BCHW
    if len(tensor.shape) == 4 and tensor.shape[3] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
        
    return tensor

def resize_tensor(tensor, size, mode='bilinear'):
    """
    调整张量尺寸
    
    Args:
        tensor: BCHW或BHWC格式的张量
        size: 目标尺寸 (高度, 宽度)
        mode: 插值模式
        
    Returns:
        调整后的张量，保持原始格式
    """
    is_bhwc = False
    if len(tensor.shape) == 4 and tensor.shape[3] == 3:  # BHWC格式
        is_bhwc = True
        tensor = tensor.permute(0, 3, 1, 2)  # 转换为BCHW用于处理
        
    # 调整大小
    resized = F.interpolate(tensor, size=size, mode=mode, align_corners=False if mode != 'nearest' else None)
    
    # 如果原始为BHWC格式，转换回去
    if is_bhwc:
        resized = resized.permute(0, 2, 3, 1)
        
    return resized

def normalize_tensor(tensor, mean=None, std=None, min_max=False):
    """
    规范化张量值
    
    Args:
        tensor: 输入张量
        mean: 均值
        std: 标准差
        min_max: 是否使用最小最大值规范化
        
    Returns:
        规范化后的张量
    """
    if min_max:
        # 最小最大值规范化到[0,1]
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        if min_val < 0 or max_val > 1.0:
            tensor = (tensor - min_val) / (max_val - min_val + 1e-6)
    elif mean is not None and std is not None:
        # 使用均值和标准差规范化
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean).view(1, len(mean), 1, 1).to(tensor.device)
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std).view(1, len(std), 1, 1).to(tensor.device)
        tensor = (tensor - mean) / std
    
    return tensor
