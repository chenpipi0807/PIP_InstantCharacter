#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InstantCharacter模型下载脚本
使用阿里云镜像加速国内下载DinoV2和SigLIP模型
"""

import os
import sys
import shutil
import requests
from tqdm import tqdm
import zipfile
import tarfile
import argparse

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ComfyUI目录
COMFYUI_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# 目标模型目录
TARGET_DIR = os.path.join(COMFYUI_DIR, "models", "clip_vision")

# 阿里云镜像地址
ALIYUN_MIRROR = "https://hf-mirror.com"

# 模型信息
MODELS = {
    "siglip": {
        "repo_id": "google/siglip-base-patch16-384",
        "local_dir": os.path.join(TARGET_DIR),
        "files": [
            {
                "url": "https://huggingface.co/google/siglip-base-patch16-384/resolve/main/pytorch_model.bin",
                "mirror_url": None,  # 暂时不使用镜像
                "dest": "siglip_so400m.pth"
            }
        ]
    },
    "dinov2": {
        "repo_id": "facebook/dinov2-base",
        "local_dir": os.path.join(TARGET_DIR),
        "files": [
            {
                "url": "https://huggingface.co/facebook/dinov2-base/resolve/main/pytorch_model.bin",
                "mirror_url": None,  # 暂时不使用镜像
                "dest": "dinov2-vitl14.bin"
            }
        ]
    }
}

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        print(f"创建目录: {directory}")
        os.makedirs(directory)

def download_file(url, dest_path, desc=None):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"下载失败: {url}, 状态码: {response.status_code}")
        return False
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            if data:  # 过滤保持连接的新块
                f.write(data)
                bar.update(len(data))
    
    return True

def download_model(model_name, use_mirror=True, force=False):
    """下载指定的模型"""
    if model_name not in MODELS:
        print(f"错误: 未知模型 '{model_name}'")
        return False
    
    model_info = MODELS[model_name]
    local_dir = model_info["local_dir"]
    
    # 检查目标目录是否已存在
    if os.path.exists(local_dir):
        print(f"目标目录已存在: {local_dir}")
    else:
        # 创建目标目录
        ensure_dir(local_dir)
    
    # 下载所有需要的文件
    for file_info in model_info["files"]:
        # 获取URL和目标文件名
        if use_mirror and file_info.get("mirror_url"):
            # 如果存在镜像链接并使用镜像模式
            url = file_info["mirror_url"]
        else:
            # 否则使用原始URL
            url = file_info["url"]
        
        # 构建目标路径
        dest_path = os.path.join(local_dir, file_info["dest"])
        
        # 检查文件是否已存在
        if os.path.exists(dest_path) and not force:
            print(f"文件已存在: {file_info['dest']}")
            choice = input(f"是否覆盖 {file_info['dest']}? (y/n): ").strip().lower()
            if choice != 'y':
                print(f"跳过 {file_info['dest']}")
                continue
        
        print(f"下载文件: {file_info['dest']} (源自 {os.path.basename(url)})")
        if not download_file(url, dest_path, f"下载 {file_info['dest']}"):
            print(f"下载 {file_info['dest']} 失败")
            return False
    
    print(f"模型 '{model_name}' 下载完成，保存至: {local_dir}")
    return True

def download_all_models(use_mirror=True, force=False):
    """下载所有模型"""
    success = True
    for model_name in MODELS:
        print(f"\n===== 开始下载模型: {model_name} =====")
        if not download_model(model_name, use_mirror, force):
            success = False
    
    if success:
        print("\n所有模型下载完成！")
    else:
        print("\n部分模型下载失败，请检查错误信息并重试。")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载InstantCharacter所需的模型")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()) + ["all"], default="all",
                        help="要下载的模型名称，或'all'下载所有模型")
    parser.add_argument("--no-mirror", action="store_true", 
                        help="不使用镜像，直接从Hugging Face下载")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖所有已存在的文件，不再提示")
    
    args = parser.parse_args()
    use_mirror = not args.no_mirror
    
    print("原版InstantCharacter模型下载工具")
    print(f"目标目录: {TARGET_DIR}")
    print(f"使用阿里云镜像: {'否' if args.no_mirror else '是'}")
    print(f"强制覆盖: {'是' if args.force else '否'}\n")
    print("说明:当运行原版InstantCharacter特征提取器时，所需的模型文件是:")
    print("  - siglip_so400m.pth (从 SigLIP 模型转换)")
    print("  - dinov2-vitl14.bin (从 DinoV2 模型转换)\n")
    
    ensure_dir(TARGET_DIR)
    
    if args.model == "all":
        download_all_models(use_mirror, args.force)
    else:
        download_model(args.model, use_mirror, args.force)
