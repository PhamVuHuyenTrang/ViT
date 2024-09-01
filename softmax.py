import math
import logging
from functools import partial
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import init_weights_vit_timm, init_weights_vit_jax, _load_weights
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from utils import named_apply
import copy
import wandb
import cfg
import argparse
parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[cfg.get_args_parser()])
args = parser.parse_args()

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 robust=False, layerth=0, n=1, lambd=0, layer=0, u = args.u_momentum, s = args.s_momentum, method = 'momentum'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.n = n
        self.lambd = lambd
        self.layer = layer
        self.scale = head_dim ** -0.5
        self.layerth = layerth

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.robust = robust
        self.num_heads = num_heads
        self.p = None
        self.method = method
        # for momentum only
        self.u = u
        self.s = s

    def forward(self, x, accum_prev_gradient = None):
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x_clone = x.view(B, N, self.num_heads, C // self.num_heads)
        
        if self.method == 'momentum':
            if self.num_heads % 2 == 0:
                u_tensor = torch.tensor([self.u] * (self.num_heads // 2) + [1] * (self.num_heads // 2)).view(1, 1, self.num_heads, 1).to(x.device)
                s_tensor = torch.tensor([self.s] * (self.num_heads // 2) + [1] * (self.num_heads // 2)).view(1, 1, self.num_heads, 1).to(x.device)
            else:
                u_tensor = torch.tensor([self.u] * ((self.num_heads // 2) + 1) + [1] * (self.num_heads // 2)).view(1, 1, self.num_heads, 1).to(x.device)
                s_tensor = torch.tensor([self.s] * ((self.num_heads // 2) + 1) + [1] * (self.num_heads // 2)).view(1, 1, self.num_heads, 1).to(x.device)
                
            if self.layerth > 0:
                x = (u_tensor * accum_prev_gradient + s_tensor * x_clone).reshape(B, N, C)
                  
            accum_prev_gradient = x.view(B, N, self.num_heads, C // self.num_heads)
            
            return x, accum_prev_gradient
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerth=None, 
                 robust=False, n=1, lambd=0, layer=0, hyperparam_1=0.9, hyperparam_2=0.1, method = 'vit'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, robust=robust, 
                              layerth=layerth, n=n, lambd=lambd, layer=layer, method = method)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layerth = layerth
        self.num_heads = num_heads
        self.method = method
        if method == 'moving_average':
            self.momentum_attn = self.generate_hyperparams(num_heads, hyperparam_1, hyperparam_2)
            if hyperparam_1 == 1.0 and hyperparam_2 == 1.0:
                self.momentum_attn = self.generate_hyperparams(num_heads, 1.0, 1.0)
                self.momentum_input = self.generate_hyperparams(num_heads, 1.0, 1.0)
            elif hyperparam_1 != 1.0 and hyperparam_2 == 1.0:
                self.momentum_input = self.generate_hyperparams(num_heads, 1.0 - hyperparam_1, 1.0)
            else:
                self.momentum_input = self.generate_hyperparams(num_heads, 1.0 - hyperparam_1, 1.0 - hyperparam_2)
    # for moving_average only
    def generate_hyperparams(self, num_heads, hyperparam_1, hyperparam_2):
        if num_heads % 2 == 0:
            first_half = [hyperparam_1] * (num_heads // 2)
            second_half = [hyperparam_2] * (num_heads // 2)
        else:
            first_half = [hyperparam_1] * (num_heads // 2 + 1)
            second_half = [hyperparam_2] * (num_heads // 2)
        return torch.tensor(first_half + second_half, dtype=torch.float32).view(1, 1, num_heads, 1)

    def forward(self, x, accum_prev_gradient = None):
        if self.method == 'moving_average':
            B, N, C = x.shape
            head_dim = C // self.num_heads
            # original X shape: (B, N, C)
            x_clone = x.view(B, N, self.num_heads, head_dim)
            attn_output = self.drop_path(self.attn(self.norm1(x)))
            attn_output = attn_output.view(B, N, self.num_heads, head_dim)
            x = x_clone * (self.momentum_input.to(x.device)) + attn_output * (self.momentum_attn.to(x.device))
            x = x.reshape(B, N, C) 
        elif self.method == 'vit':
            x = x + self.drop_path(self.attn(self.norm1(x)))
        elif self.method == 'momentum':
            if self.layerth == 0:
                attn_output_before_drop_path, accum_prev_gradient = self.attn(self.norm1(x))
            else:
                attn_output_before_drop_path, accum_prev_gradient = self.attn(self.norm1(x), accum_prev_gradient)
                x = x + self.drop_path(attn_output_before_drop_path)
                  
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.method == 'momentum':
            return x, accum_prev_gradient
        else:
            return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',pretrained_cfg=None,pretrained_cfg_overlay=None,robust=False,n=1,lambd=0,layer=0,
                 hyperparam_1=args.hyperparam_1, hyperparam_2=args.hyperparam_2, method = args.method):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            hyperparam_1 (float): first hyperparameter for attention weights
            hyperparam_2 (float): second hyperparameter for attention weights
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.lambd = lambd
        self.n = n
        self.layer = layer
        self.hyperparam_1 = hyperparam_1
        self.hyperparam_2 = hyperparam_2
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.method = method

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, 
                layerth=i, robust=robust, n=self.n, lambd=self.lambd, layer=self.layer,
                hyperparam_1=self.hyperparam_1, hyperparam_2=self.hyperparam_2, method = self.method)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.weight_init(weight_init)

    def weight_init(self, mode=''):
        if mode == 'jax':
            self.apply(init_weights_vit_jax)
        elif mode == 'moco':
            self.apply(init_weights_vit_moco)
        elif mode == 'timm':
            self.apply(init_weights_vit_timm)
        else:
            trunc_normal_(self.pos_embed, std=.02)
            if self.dist_token is not None:
                trunc_normal_(self.dist_token, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if self.method == 'momentum':
            for block in self.blocks:
                if block.layerth == 0: 
                    x, accum_prev_gradient = block(x)
                else:
                    x, accum_prev_gradient = block(x, accum_prev_gradient)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
