import torch.nn as nn
import torch
import math

from .norm_layer import RMSNorm


def FeedForward(dim, mult=4):
    """前馈神经网络模块"""
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

    
def reshape_tensor(x, heads):
    """重新塑形张量以用于多头注意力"""
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    """感知器注意力模块"""
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, shift=None, scale=None):
        """
        Args:
            x (torch.Tensor): 图像特征，形状 (b, n1, D)
            latent (torch.Tensor): 潜在特征，形状 (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        if shift is not None and scale is not None:
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # 注意力计算
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # 比之后除更稳定
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class ReshapeExpandToken(nn.Module):
    """重塑和扩展令牌模块"""
    def __init__(self, expand_token, token_dim):
        super().__init__()
        self.expand_token = expand_token
        self.token_dim = token_dim

    def forward(self, x):
        x = x.reshape(-1, self.expand_token, self.token_dim)
        return x


class TimeResampler(nn.Module):
    """时间重采样器模块"""
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        timestep_in_dim=320,
        timestep_flip_sin_to_cos=True,
        timestep_freq_shift=0,
        expand_token=None,
        extra_dim=None,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.expand_token = expand_token is not None
        if expand_token:
            self.expand_proj = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(embedding_dim * 2, embedding_dim * expand_token),
                ReshapeExpandToken(expand_token, embedding_dim),
                RMSNorm(embedding_dim, eps=1e-8),
            )

        self.proj_in = nn.Linear(embedding_dim, dim)
        
        self.extra_feature = extra_dim is not None
        if extra_dim is not None:
            self.extra_proj = nn.Linear(extra_dim, dim)

        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))
                
        self.proj_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim)
        )

        try:
            from diffusers.models.embeddings import Timesteps, TimestepEmbedding
            self.time_embedding = TimestepEmbedding(timestep_in_dim, dim, act_fn="silu")
        except ImportError:
            print("[PIP-InstantCharacter] 警告: 无法导入TimestepEmbedding，将使用自定义实现")
            # 简化版TimestepEmbedding
            self.time_embedding = nn.Sequential(
                nn.Linear(timestep_in_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

    def forward(self, x, timestep, need_temb=False, extra_feature=None):
        # 将输入投影到潜在空间
        if self.expand_token:
            x = self.expand_proj(x)
        
        x = self.proj_in(x)
        
        if self.extra_feature and extra_feature is not None:
            extra_x = self.extra_proj(extra_feature)
            x = torch.cat([x, extra_x], dim=1)
        
        # 获取潜在向量
        b = x.shape[0]
        latents = self.latents.repeat(b, 1, 1)
        
        # 计算时间嵌入
        emb = self.embedding_time(x, timestep)
            
        # 潜在块
        for attn, ff in self.blocks:
            latents = attn(x, latents, emb, None) + latents
            latents = ff(latents) + latents
            
        # 投影输出
        latents = self.proj_out(latents)
        
        if need_temb:
            return latents, emb
        return latents

    def embedding_time(self, sample, timestep):
        if timestep is None:
            return None
            
        if not torch.is_tensor(timestep):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timestep.expand(sample.shape[0])
        
        try:
            t_emb = self.time_embedding(timestep)
        except Exception:
            # 如果原始实现失败，使用简化版
            timestep = timestep.float() / 1000.0
            half_dim = self.time_embedding[0].in_features // 2
            emb = torch.log(torch.tensor(10000.0)).to(sample.device) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=sample.device) * -emb)
            emb = timestep[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            t_emb = self.time_embedding(emb)
            
        return t_emb


class CrossLayerCrossScaleProjector(nn.Module):
    """跨层跨尺度投影器"""
    def __init__(
        self,
        inner_dim=2688,
        num_attention_heads=42,
        attention_head_dim=64,
        cross_attention_dim=2688,
        num_layers=4,

        # resampler
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=1024,
        embedding_dim=1152 + 1536,
        output_dim=4096,
        ff_mult=4,
        timestep_in_dim=320,
        timestep_flip_sin_to_cos=True,
        timestep_freq_shift=0,
    ):
        super().__init__()
        
        # 尝试导入所需组件
        try:
            from diffusers.models.transformers.transformer_2d import BasicTransformerBlock
            from timm.models.vision_transformer import Mlp
            
            # 跨层块
            self.cross_layer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=0,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn="geglu",
                        num_embeds_ada_norm=None,
                        attention_bias=False,
                        only_cross_attention=False,
                        double_self_attention=False,
                        upcast_attention=False,
                        norm_type='layer_norm',
                        norm_elementwise_affine=True,
                        norm_eps=1e-6,
                        attention_type="default",
                    )
                    for _ in range(num_layers)
                ]
            )

            # 跨尺度块
            self.cross_scale_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=0,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn="geglu",
                        num_embeds_ada_norm=None,
                        attention_bias=False,
                        only_cross_attention=False,
                        double_self_attention=False,
                        upcast_attention=False,
                        norm_type='layer_norm',
                        norm_elementwise_affine=True,
                        norm_eps=1e-6,
                        attention_type="default",
                    )
                    for _ in range(num_layers)
                ]
            )

            # 投影层
            self.proj = Mlp(
                in_features=inner_dim, 
                hidden_features=int(inner_dim*2), 
                act_layer=lambda: nn.GELU(approximate="tanh"), 
                drop=0
            )

            self.proj_cross_layer = Mlp(
                in_features=inner_dim, 
                hidden_features=int(inner_dim*2), 
                act_layer=lambda: nn.GELU(approximate="tanh"), 
                drop=0
            )

            self.proj_cross_scale = Mlp(
                in_features=inner_dim, 
                hidden_features=int(inner_dim*2), 
                act_layer=lambda: nn.GELU(approximate="tanh"), 
                drop=0
            )
            
        except ImportError:
            print("[PIP-InstantCharacter] 警告: 无法导入所需组件，将使用简化版实现")
            
            # 简化版BasicTransformerBlock
            class SimpleTransformerBlock(nn.Module):
                def __init__(self, dim, heads=8, dim_head=64, mlp_dim=None):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(dim)
                    self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
                    self.norm2 = nn.LayerNorm(dim)
                    mlp_dim = mlp_dim or dim * 4
                    self.mlp = nn.Sequential(
                        nn.Linear(dim, mlp_dim),
                        nn.GELU(),
                        nn.Linear(mlp_dim, dim)
                    )
                
                def forward(self, x, encoder_hidden_states=None, cross_attention_kwargs=None):
                    if encoder_hidden_states is not None:
                        # Cross-attention
                        x = x + self.attn(self.norm1(x), self.norm1(encoder_hidden_states), self.norm1(encoder_hidden_states))[0]
                    else:
                        # Self-attention
                        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                    x = x + self.mlp(self.norm2(x))
                    return x
            
            # 简化版Mlp
            class SimpleMlp(nn.Module):
                def __init__(self, in_features, hidden_features):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_features, hidden_features),
                        nn.GELU(),
                        nn.Linear(hidden_features, in_features)
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            # 跨层块
            self.cross_layer_blocks = nn.ModuleList(
                [SimpleTransformerBlock(inner_dim, num_attention_heads, attention_head_dim // num_attention_heads)
                 for _ in range(num_layers)]
            )
            
            # 跨尺度块
            self.cross_scale_blocks = nn.ModuleList(
                [SimpleTransformerBlock(inner_dim, num_attention_heads, attention_head_dim // num_attention_heads)
                 for _ in range(num_layers)]
            )
            
            # 投影层
            self.proj = SimpleMlp(inner_dim, inner_dim * 2)
            self.proj_cross_layer = SimpleMlp(inner_dim, inner_dim * 2)
            self.proj_cross_scale = SimpleMlp(inner_dim, inner_dim * 2)

        # 重采样器
        self.resampler = TimeResampler(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            ff_mult=ff_mult,
            timestep_in_dim=timestep_in_dim,
            timestep_flip_sin_to_cos=timestep_flip_sin_to_cos,
            timestep_freq_shift=timestep_freq_shift,
        )

    def forward(self, low_res_shallow, low_res_deep, high_res_deep, timesteps, cross_attention_kwargs=None, need_temb=True):
        '''
            low_res_shallow [bs, 729*l, c]
            low_res_deep    [bs, 729, c]
            high_res_deep   [bs, 729*4, c]
        '''

        cross_layer_hidden_states = low_res_deep
        for block in self.cross_layer_blocks:
            cross_layer_hidden_states = block(
                cross_layer_hidden_states,
                encoder_hidden_states=low_res_shallow,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        cross_layer_hidden_states = self.proj_cross_layer(cross_layer_hidden_states)

        cross_scale_hidden_states = low_res_deep
        for block in self.cross_scale_blocks:
            cross_scale_hidden_states = block(
                cross_scale_hidden_states,
                encoder_hidden_states=high_res_deep,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        cross_scale_hidden_states = self.proj_cross_scale(cross_scale_hidden_states)
        
        hidden_states = self.proj(low_res_deep) + cross_scale_hidden_states
        hidden_states = torch.cat([hidden_states, cross_layer_hidden_states], dim=1)

        hidden_states, timestep_emb = self.resampler(hidden_states, timesteps, need_temb=True)
        return hidden_states, timestep_emb
