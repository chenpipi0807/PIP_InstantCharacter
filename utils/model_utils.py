"""
InstantCharacter模型工具函数 - 辅助模型加载和处理
"""

import os
import torch
import json
import requests
import tqdm
from typing import Dict, List, Optional, Union, Any
from huggingface_hub import hf_hub_download


def download_model_from_hf(repo_id: str, filename: str, cache_dir: str):
    """
    从Hugging Face下载模型文件
    
    Args:
        repo_id: 仓库ID
        filename: 文件名
        cache_dir: 缓存目录
    
    Returns:
        下载文件的路径
    """
    os.makedirs(cache_dir, exist_ok=True)
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True,
        )
        return file_path
    except Exception as e:
        print(f"从Hugging Face下载模型失败: {str(e)}")
        print("尝试从HF镜像站下载...")
        return download_from_hf_mirror(repo_id, filename, cache_dir)


def download_from_hf_mirror(repo_id: str, filename: str, cache_dir: str):
    """
    从HF镜像站下载模型（当无法直接从Hugging Face下载时使用）
    
    Args:
        repo_id: 仓库ID
        filename: 文件名
        cache_dir: 缓存目录
    
    Returns:
        下载文件的路径
    """
    mirror_url = f"https://hf-mirror.com/{repo_id}/resolve/main/{filename}"
    local_path = os.path.join(cache_dir, os.path.basename(filename))
    
    try:
        response = requests.get(mirror_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB
        
        with open(local_path, 'wb') as f:
            with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
        
        print(f"已从镜像站下载模型到: {local_path}")
        return local_path
    except Exception as e:
        print(f"从镜像站下载失败: {str(e)}")
        raise e


def apply_lora_to_model(model, lora_path: str, scale: float = 0.8):
    """
    将LoRA权重应用到模型
    
    Args:
        model: 要应用LoRA的模型
        lora_path: LoRA文件路径
        scale: LoRA强度
    """
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA文件不存在: {lora_path}")
    
    # 记录原始权重
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    # 加载LoRA权重
    lora_state_dict = torch.load(lora_path, map_location="cpu")
    
    # 应用LoRA权重
    # 这里是简化版本，实际应用中需要处理不同的LoRA格式
    for name, param in model.named_parameters():
        if name in lora_state_dict:
            lora_weight = lora_state_dict[name]
            if lora_weight.shape == param.shape:
                param.data += scale * lora_weight
    
    return original_weights


def restore_original_weights(model, original_weights):
    """
    恢复模型原始权重（移除LoRA效果）
    
    Args:
        model: 要恢复的模型
        original_weights: 原始权重字典
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_weights:
                param.copy_(original_weights[name])
