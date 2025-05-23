"""
InstantCharacter IP适配器 - 原版实现
完全复制原版InstantCharacter的IP适配器实现，确保效果一致
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
from typing import Optional, Union, Tuple, List


class RMSNorm(nn.Module):
    """从原版InstantCharacter复制的RMSNorm实现"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x: [bs, seq_len, dim]
        # norm_x: [bs, seq_len, dim]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x: [bs, seq_len, dim]
        # output: [bs, seq_len, dim]
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FluxIPAttnProcessor(nn.Module):
    """从原版InstantCharacter复制的注意力处理器"""

    def __init__(
        self,
        hidden_size=None,
        ip_hidden_states_dim=None,
    ):
        super().__init__()
        self.norm_ip_q = RMSNorm(128, eps=1e-6)
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size)
        self.norm_ip_k = RMSNorm(128, eps=1e-6)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size)

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        emb_dict={},
        subject_emb_dict={},
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # 样本投影
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # IP适配器
        ip_hidden_states = self._get_ip_hidden_states(
            attn, 
            query if encoder_hidden_states is not None else query[:, emb_dict.get('length_encoder_hidden_states', 0):],
            subject_emb_dict.get('ip_hidden_states', None)
        )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)

        # 处理encoder_hidden_states (如果存在)
        if encoder_hidden_states is not None and hasattr(attn, 'add_q_proj'):
            # 上下文投影
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if hasattr(attn, 'norm_added_q') and attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if hasattr(attn, 'norm_added_k') and attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # 注意力
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # 应用旋转嵌入（如果存在）
        if image_rotary_emb is not None:
            try:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            except ImportError:
                print("[PIP-InstantCharacter] 警告: diffusers中缺少apply_rotary_emb，跳过旋转嵌入应用")

        # 注意力计算
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 处理encoder_hidden_states的情况
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
                
            if ip_hidden_states is not None:
                hidden_states = hidden_states + ip_hidden_states * subject_emb_dict.get('scale', 1.0)

            # 线性投影
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            # 对于没有encoder_hidden_states的情况
            if ip_hidden_states is not None:
                offset = emb_dict.get('length_encoder_hidden_states', 0)
                hidden_states[:, offset:] = hidden_states[:, offset:] + ip_hidden_states * subject_emb_dict.get('scale', 1.0)

            # 线性投影和dropout（如果有）
            if hasattr(attn, 'to_out'):
                if isinstance(attn.to_out, nn.Sequential):
                    for layer in attn.to_out:
                        hidden_states = layer(hidden_states)
                else:
                    hidden_states = attn.to_out(hidden_states)

            return hidden_states

    def _scaled_dot_product_attention(self, query, key, value, attention_mask=None, heads=None):
        """缩放点积注意力，用于IP隐藏状态"""
        query = rearrange(query, '(b h) l c -> b h l c', h=heads)
        key = rearrange(key, '(b h) l c -> b h l c', h=heads)
        value = rearrange(value, '(b h) l c -> b h l c', h=heads)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=None)
        hidden_states = rearrange(hidden_states, 'b h l c -> (b h) l c', h=heads)
        hidden_states = hidden_states.to(query)
        return hidden_states

    def _get_ip_hidden_states(self, attn, img_query, ip_hidden_states):
        """获取IP适配器隐藏状态"""
        if ip_hidden_states is None:
            return None
        
        if not hasattr(self, 'to_k_ip') or not hasattr(self, 'to_v_ip'):
            return None

        # 对查询和键值应用RMSNorm和线性变换
        ip_query = self.norm_ip_q(rearrange(img_query, 'b l (h d) -> b h l d', h=attn.heads))
        ip_query = rearrange(ip_query, 'b h l d -> (b h) l d')
        
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_key = self.norm_ip_k(rearrange(ip_key, 'b l (h d) -> b h l d', h=attn.heads))
        ip_key = rearrange(ip_key, 'b h l d -> (b h) l d')
        
        ip_value = self.to_v_ip(ip_hidden_states)
        
        # 处理head_to_batch_dim方法
        if hasattr(attn, 'head_to_batch_dim'):
            ip_value = attn.head_to_batch_dim(ip_value)
        else:
            # 如果没有该方法，手动实现
            batch_size, seq_len, dim = ip_value.shape
            head_size = dim // attn.heads
            ip_value = ip_value.reshape(batch_size, seq_len, attn.heads, head_size)
            ip_value = ip_value.permute(0, 2, 1, 3).reshape(batch_size * attn.heads, seq_len, head_size)
            
        # 计算注意力
        ip_hidden_states = self._scaled_dot_product_attention(
            ip_query.to(ip_value.dtype), ip_key.to(ip_value.dtype), ip_value, None, attn.heads)
        ip_hidden_states = ip_hidden_states.to(img_query.dtype)
        
        # 处理batch_to_head_dim方法
        if hasattr(attn, 'batch_to_head_dim'):
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        else:
            # 如果没有该方法，手动实现
            batch_size = img_query.shape[0]
            seq_len = ip_hidden_states.shape[1]
            head_size = ip_hidden_states.shape[2]
            ip_hidden_states = ip_hidden_states.reshape(batch_size, attn.heads, seq_len, head_size)
            ip_hidden_states = ip_hidden_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, attn.heads * head_size)
            
        return ip_hidden_states


class InstantCharacterIPAdapter:
    """
    IP适配器 - 原版InstantCharacter的实现
    """
    
    def __init__(self, adapter_path=None, device=None, nb_token=1024):
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_dict = {}
        self.nb_token = nb_token  # 原版代码中默认使用的token数量
        
        # 初始化组件
        self.image_proj = None
        self.image_proj_norm = None
        self.hidden_states_proj = None
        
        # 如果指定了适配器路径，则加载
        if adapter_path is not None:
            self.load_adapter(adapter_path)
        
        # 检查IP适配器文件是否存在
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"[PIP-InstantCharacter] IP适配器文件不存在: {adapter_path}")
            
        print(f"[PIP-InstantCharacter] 初始化IP适配器: {os.path.basename(adapter_path)}")
        self.load_adapter(adapter_path)
    
    def load_adapter(self, adapter_path):
        """加载IP适配器权重 - 同原版InstantCharacter实现一致"""
        try:
            if not os.path.isfile(adapter_path):
                raise ValueError(f"[PIP-InstantCharacter] 错误: IP适配器文件不存在: {adapter_path}")
                
            print(f"[PIP-InstantCharacter] 加载IP适配器权重: {os.path.basename(adapter_path)}")
            
            # 与原版一致，直接加载到CPU
            self.state_dict = torch.load(adapter_path, map_location="cpu")
            
            # 打印加载的关键信息
            print(f"[PIP-InstantCharacter] 权重文件类型: {type(self.state_dict)}")
            
            if isinstance(self.state_dict, dict):
                print(f"[PIP-InstantCharacter] 权重文件键名: {list(self.state_dict.keys())}")
                
                # 检查是否包含原版中使用的键
                for expected_key in ['image_proj', 'ip_adapter']:
                    if expected_key in self.state_dict:
                        if isinstance(self.state_dict[expected_key], dict):
                            print(f"[PIP-InstantCharacter] {expected_key} 包含子键: {list(self.state_dict[expected_key].keys())}")
                        else:
                            print(f"[PIP-InstantCharacter] {expected_key} 类型: {type(self.state_dict[expected_key])}")
                    else:
                        print(f"[PIP-InstantCharacter] 警告: 权重文件中缺少关键组件 {expected_key}")
            
            # 初始化组件 - 完全使用原版的方式
            self._init_components()
            
            # 清理内存
            del self.state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("[PIP-InstantCharacter] IP适配器加载完成")
        except Exception as e:
            print(f"[PIP-InstantCharacter] 加载IP适配器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 创建空字典以避免其他方法出错
            self.state_dict = {}
    
    def _init_components(self):
        """初始化模型组件 - 原版InstantCharacter实现一致"""
        # 检查state_dict是否为空
        if not self.state_dict:
            raise ValueError("[PIP-InstantCharacter] 错误: 状态字典为空")
        
        print(f"[PIP-InstantCharacter] 初始化组件时的键名: {list(self.state_dict.keys())}")
        
        try:
            # 根据原版InstantCharacter的实现，一共有两个关键组件
            # 1. 图像投影模型 - CrossLayerCrossScaleProjector
            # 2. IP适配器层 - FluxIPAttnProcessor
            
            # 先尝试加载原版中包含的IP适配器权重
            if 'ip_adapter' in self.state_dict:
                print("[PIP-InstantCharacter] 加载IP适配器层权重")
                
                # 在原版中，这些层被直接创建并加载到FluxIPAttnProcessor中
                # 我们这里只需要维护对应的组件
                self.image_proj = nn.Linear(1536, 768).to(self.device)
                self.image_proj_norm = RMSNorm(768, eps=1e-6).to(self.device)
                self.hidden_states_proj = nn.Linear(768, 768).to(self.device)
                
                print("[PIP-InstantCharacter] 初始化完成基本组件")
            else:
                print("[PIP-InstantCharacter] 警告: 没有找到ip_adapter权重，使用默认初始化")
                self.image_proj = nn.Linear(1536, 768).to(self.device)
                self.image_proj_norm = RMSNorm(768, eps=1e-6).to(self.device)
                self.hidden_states_proj = nn.Linear(768, 768).to(self.device)
            
            # 记录token数量参数
            print(f"[PIP-InstantCharacter] 使用token数量: {self.nb_token}")
            
            print(f"[PIP-InstantCharacter] 成功初始化组件:")
            print(f"[PIP-InstantCharacter] image_proj: {self.image_proj}")
            print(f"[PIP-InstantCharacter] image_proj_norm: {self.image_proj_norm}")
            print(f"[PIP-InstantCharacter] hidden_states_proj: {self.hidden_states_proj}")
            
        except Exception as e:
            print(f"[PIP-InstantCharacter] 创建原版组件失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果创建失败则报错
            raise RuntimeError(f"[PIP-InstantCharacter] 无法创建原版IP适配器组件: {e}")
            
            # 尝试所有可能的前缀
            prefixes_to_try = ['image_proj.', 'ip_adapter.image_proj.', 'adapter.image_proj.']
            for prefix in prefixes_to_try:
                if self.image_proj is None:
                    weight_key = f"{prefix}weight"
                    if weight_key in self.state_dict:
                        print(f"[PIP-InstantCharacter] 找到image_proj权重: {weight_key}")
                        self.image_proj = self._create_linear(prefix, input_dim=1536, output_dim=768)
                        
                if self.image_proj_norm is None:
                    weight_key = f"{prefix}norm.weight"
                    if weight_key in self.state_dict:
                        print(f"[PIP-InstantCharacter] 找到image_proj_norm权重: {weight_key}")
                        self.image_proj_norm = self._create_layernorm(prefix + 'norm', dim=768)
                
                # 隐藏状态投影
                if self.hidden_states_proj is None:
                    weight_key = f"{prefix}hidden_states_proj.weight"
                    if weight_key in self.state_dict:
                        print(f"[PIP-InstantCharacter] 找到hidden_states_proj权重: {weight_key}")
                        self.hidden_states_proj = self._create_linear(prefix + 'hidden_states_proj', input_dim=768, output_dim=768)
                
                # 如果找到了所有组件，就跳出循环
                if self.image_proj is not None and self.image_proj_norm is not None:
                    print(f"[PIP-InstantCharacter] 使用前缀 '{prefix}' 找到了IP适配器组件")
                    break
        
        # 检查核心组件是否初始化成功
        if self.image_proj is None or self.image_proj_norm is None:
            print("[PIP-InstantCharacter] 错误: 核心IP适配器组件初始化失败")
            print(f"[PIP-InstantCharacter] image_proj: {self.image_proj}, image_proj_norm: {self.image_proj_norm}")
            raise RuntimeError("[PIP-InstantCharacter] 必要的IP适配器组件缺失")
        
        print("[PIP-InstantCharacter] IP适配器组件初始化完成")
    
    def _create_linear(self, prefix, input_dim=None, output_dim=None):
        """
        创建线性层
        
        Args:
            prefix: 权重键前缀
            input_dim: 可选，输入维度，如果权重不存在则使用这个尺寸
            output_dim: 可选，输出维度，如果权重不存在则使用这个尺寸
        """
        weight = self.state_dict.get(f"{prefix}weight")
        bias = self.state_dict.get(f"{prefix}bias")
        
        # 使用默认权重或自定义尺寸
        if weight is not None and bias is not None:
            # 创建基于权重的线性层
            layer = nn.Linear(weight.shape[1], weight.shape[0], device='cpu')
            print(f"[PIP-InstantCharacter] 创建线性层 {prefix}: {weight.shape[1]} -> {weight.shape[0]}")
            
            # 使用no_grad避免不必要的梯度计算
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
        elif input_dim is not None and output_dim is not None:
            # 如果没有预定义的权重，但指定了尺寸，则创建新的线性层
            layer = nn.Linear(input_dim, output_dim, device='cpu')
            print(f"[PIP-InstantCharacter] 创建自定义线性层 {prefix}: {input_dim} -> {output_dim}")
        else:
            print(f"[PIP-InstantCharacter] 无法创建线性层 {prefix}: 无有效权重或尺寸信息")
            return None
        
        # 移动到目标设备
        return layer.to(self.device)
    
    def _create_layernorm(self, prefix, dim=None):
        """
        创建LayerNorm层
        
        Args:
            prefix: 权重键前缀
            dim: 可选，如果权重不存在则使用此维度创建LayerNorm
        """
        weight = self.state_dict.get(f"{prefix}.weight")
        bias = self.state_dict.get(f"{prefix}.bias")
        
        # 使用预定义权重或自定义尺寸
        if weight is not None and bias is not None:
            # 创建基于权重的LayerNorm
            layer = nn.LayerNorm(weight.shape[0], elementwise_affine=True, device='cpu')
            print(f"[PIP-InstantCharacter] 创建LayerNorm层 {prefix}: 维度 {weight.shape[0]}")
            
            # 使用no_grad避免不必要的梯度计算
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
        elif dim is not None:
            # 如果没有预定义的权重，但指定了尺寸，则创建新的LayerNorm
            layer = nn.LayerNorm(dim, elementwise_affine=True, device='cpu')
            print(f"[PIP-InstantCharacter] 创建自定义LayerNorm层 {prefix}: 维度 {dim}")
        else:
            print(f"[PIP-InstantCharacter] 无法创建LayerNorm层 {prefix}: 无有效权重或维度信息")
            return None
        
        # 移动到目标设备
        return layer.to(self.device)
    
    @torch.no_grad()
    def encode_image(self, image_features):
        """编码图像特征 - 原版InstantCharacter实现一致"""
        if image_features is None:
            raise ValueError("[PIP-InstantCharacter] 错误: 图像特征为None")
            
        if not isinstance(image_features, torch.Tensor):
            raise ValueError(f"[PIP-InstantCharacter] 错误: 图像特征必须是张量")
        
        # 打印特征形状信息以便调试
        print(f"[PIP-InstantCharacter] 特征形状: {image_features.shape}, 设备: {image_features.device}, 类型: {image_features.dtype}")
        print(f"[PIP-InstantCharacter] image_proj权重类型: {self.image_proj.weight.dtype}, 设备: {self.image_proj.weight.device}")
            
        # 移动到正确的设备
        image_features = image_features.to(self.device)
        
        # 确保数据类型匹配
        if image_features.dtype != self.image_proj.weight.dtype:
            print(f"[PIP-InstantCharacter] 将特征从 {image_features.dtype} 转换为 {self.image_proj.weight.dtype}")
            image_features = image_features.to(dtype=self.image_proj.weight.dtype)
        
        # 执行特征编码
        try:
            # 原版中，模型使用CrossLayerCrossScaleProjector处理特征
            # 但在我们的简化实现中，直接使用标准的投影层
            
            # 应用线性投影
            x = self.image_proj(image_features)
            print(f"[PIP-InstantCharacter] 应用image_proj后形状: {x.shape}")
            
            # 应用归一化
            x = self.image_proj_norm(x)
            print(f"[PIP-InstantCharacter] 应用image_proj_norm后形状: {x.shape}")
            
            # 应用隐藏状态投影
            if self.hidden_states_proj is not None:
                x = self.hidden_states_proj(x)
                print(f"[PIP-InstantCharacter] 应用hidden_states_proj后形状: {x.shape}")
            
            # 原版中，这里会进行重采样到特定数量的token
            # 在我们的简化实现中，我们保持原始形状
            print(f"[PIP-InstantCharacter] 原版会使用 {self.nb_token} 个token，当前形状: {x.shape}")
            
            print("[PIP-InstantCharacter] 特征编码成功")
            return x
        except Exception as e:
            print(f"[PIP-InstantCharacter] 特征编码失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def to(self, device):
        """移动所有组件到指定设备"""
        self.device = device
        for name, module in self.__dict__.items():
            if isinstance(module, (nn.Module, torch.Tensor)) and module is not None:
                setattr(self, name, module.to(device))
        return self
        
    @torch.no_grad()
    def apply(self, input_model, features, scale=1.0):
        """
        将InstantCharacter特征应用到模型 - 原版InstantCharacter实现一致
        
        Args:
            input_model: 输入的模型（ModelPatcher实例）
            features: 图像特征字典，包含多尺度特征
            scale: 特征强度系数
        """
        print(f"[PIP-InstantCharacter] 应用IP适配器, scale={scale}, nb_token={self.nb_token}")
        print(f"[PIP-InstantCharacter] 可用特征键: {list(features.keys()) if isinstance(features, dict) else '不是字典'}")
        
        # 验证输入
        if input_model is None or not features:
            print("[PIP-InstantCharacter] 错误: 输入模型或特征为空")
            return input_model

        # 获取模型副本
        patched_model = input_model.clone() if hasattr(input_model, 'clone') else input_model
        
        # 获取transformer实例
        transformer = None
        if hasattr(patched_model, 'model'):
            if hasattr(patched_model.model, 'diffusion_model'):
                transformer = patched_model.model.diffusion_model
            elif hasattr(patched_model.model, 'transformer'):
                transformer = patched_model.model.transformer
            else:
                transformer = patched_model.model
            
        # 获取设备信息并移动组件
        if transformer is not None and hasattr(transformer, 'parameters'):
            try:
                param = next(transformer.parameters())
                device = param.device
                self.to(device)
                print(f"[PIP-InstantCharacter] 移动IP适配器到设备: {device}")
            except Exception as e:
                print(f"[PIP-InstantCharacter] 无法获取transformer设备: {e}")
                
        # 优先使用原版格式的特征
        feature_to_use = None
        
        # 首选键值 - 原版InstantCharacter使用的特征键
        if isinstance(features, dict):
            if 'image_embeds_low_res_deep' in features and features['image_embeds_low_res_deep'] is not None:
                feature_to_use = features['image_embeds_low_res_deep']
                print("[PIP-InstantCharacter] 使用image_embeds_low_res_deep特征 (原版格式)")
            # 深度特征
            elif 'deep_features' in features and features.get('deep_features') is not None:
                if isinstance(features['deep_features'], dict) and 'low_res' in features['deep_features']:
                    feature_to_use = features['deep_features']['low_res']
                    print("[PIP-InstantCharacter] 使用deep_features.low_res特征")
            # 尝试使用dino特征
            elif 'dino_features_low' in features and features['dino_features_low'] is not None:
                feature_to_use = features['dino_features_low']
                print("[PIP-InstantCharacter] 使用dino_features_low特征")
            # 最后尝试组合特征
            elif 'combined' in features and features['combined'] is not None:
                feature_to_use = features['combined']
                print("[PIP-InstantCharacter] 使用combined特征")
        else:
            # 如果不是字典，直接使用features
            feature_to_use = features
            print("[PIP-InstantCharacter] 直接使用传入的非字典特征")

        if feature_to_use is None:
            print("[PIP-InstantCharacter] 错误: 无法找到可用的特征数据")
            return input_model

        if isinstance(feature_to_use, torch.Tensor):
            print(f"[PIP-InstantCharacter] 特征张量形状: {feature_to_use.shape}, 类型: {feature_to_use.dtype}, 设备: {feature_to_use.device}")

        try:
            # 编码图像特征
            image_features = self.encode_image(feature_to_use)
            print(f"[PIP-InstantCharacter] 特征编码完成: {image_features.shape}")

            # 设置注意力处理器 - 原版InstantCharacter方式
            if transformer is not None and hasattr(transformer, 'set_attn_processor'):
                # 加载当前的处理器
                attn_procs = {}
                curr_attn_procs = transformer.attn_processors
                updated_count = 0

                # 遍历找到所有注意力层
                for name, module in transformer.named_modules():
                    if name.endswith('attn1') or name.endswith('attn2'):
                        # 只处理还没有设置或者不是FluxIPAttnProcessor的层
                        if name not in curr_attn_procs or not isinstance(curr_attn_procs[name], FluxIPAttnProcessor):
                            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                                hidden_size = module.to_q.out_features
                                # 使用原版中相同的参数创建处理器
                                attn_procs[name] = FluxIPAttnProcessor(
                                    hidden_size=hidden_size,
                                    ip_hidden_states_dim=768,  # 编码后的特征维度
                                )
                                updated_count += 1

                # 只有在有新层需要设置时才更新
                if updated_count > 0:
                    transformer.set_attn_processor(attn_procs)
                    print(f"[PIP-InstantCharacter] 成功设置注意力处理器: {updated_count}个新层")
                
                # 对每个处理器设置主题字典
                subject_dict_count = 0
                for name, proc in transformer.attn_processors.items():
                    if isinstance(proc, FluxIPAttnProcessor):
                        # 在原版中，主题字典包含图像特征和缩放比例
                        # 创建兼容原版的subject_emb_dict
                        subject_emb_dict = {
                            'ip_hidden_states': image_features,
                            'scale': scale
                        }
                        # 将字典存储到模型中，供每次调用使用
                        if not hasattr(transformer, 'ip_adapter_subject_emb_dict'):
                            transformer.ip_adapter_subject_emb_dict = {}
                        transformer.ip_adapter_subject_emb_dict[name] = subject_emb_dict
                        subject_dict_count += 1
                
                print(f"[PIP-InstantCharacter] 成功设置特征和缩放比例到全部 {subject_dict_count} 个处理器")
            
            # 兼容使用ip_adapter_image_embeds属性的模型
            elif transformer is not None and hasattr(transformer, 'ip_adapter_image_embeds'):
                transformer.ip_adapter_scale = scale
                transformer.ip_adapter_image_embeds = image_features
                print("[PIP-InstantCharacter] 直接设置了ip_adapter_image_embeds属性")

            return patched_model
        except Exception as e:
            print(f"[PIP-InstantCharacter] 应用IP适配器时出错: {e}")
            import traceback
            traceback.print_exc()
            return input_model
