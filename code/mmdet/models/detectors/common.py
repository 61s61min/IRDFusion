# YOLOv5 common modules
import math
from functools import partial

import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from torch.nn import init
from einops import rearrange, repeat
from typing import Callable, Any
from timm.models.layers import DropPath
from torch.utils import checkpoint
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    

class NiNfusion(nn.Module):
    def __init__(self, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()
        c1=c2*2
        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, rgb, ir):
        y = self.concat([rgb,ir])
        y = self.act(self.conv(y))

        return y
class NiNfusion_RGB(nn.Module):
    def __init__(self, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion_RGB, self).__init__()
        # c1=c2
        # self.concat = Concat(dimension=1)
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.act = nn.SiLU()

    def forward(self, rgb, ir):
        # # y = self.concat([rgb,ir])
        # y = self.act(self.conv(rgb))

        return rgb

class NiNfusion_IR(nn.Module):
    def __init__(self, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion_IR, self).__init__()
        # c1=c2
        # self.concat = Concat(dimension=1)
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.act = nn.SiLU()

    def forward(self, rgb, ir):
        # # y = self.concat([rgb,ir])
        # y = self.act(self.conv(ir))

        return ir

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


###################################################################################
####################### clw note: 只用红外的输出
class CrossTransformerBlockForOnlyTir(nn.Module):    # clw note: CFE模块, 
    # def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
    def __init__(self, d_model, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlockForOnlyTir, self).__init__()
        self.loops = loops_num
        # self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.crossatt = CrossAttentionForOnlyTir(d_model, h, attn_pdrop, resid_pdrop)  # clw modify
        # self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
        #                              # nn.SiLU(),  # changed from GELU
        #                              nn.GELU(),  # changed from GELU
        #                              nn.Linear(block_exp * d_model, d_model),
        #                              nn.Dropout(resid_pdrop),
        #                              )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )

        # Layer norm
        # self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        # self.coefficient1 = LearnableCoefficient()
        # self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        # self.coefficient5 = LearnableCoefficient()
        # self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        # bs, nx, c = rgb_fea_flat.size()
        # h = w = int(math.sqrt(nx))

        for _ in range(self.loops):
            # with Learnable Coefficient
            #rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])   # clw modify: CFE模块只输出红外
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

        return ir_fea_flat
    


    
class TransformerFusionBlockForOnlyTir(nn.Module):              # clw modify： 只用红外特征分支（把可见光向红外融合）
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlockForOnlyTir, self).__init__()

        self.n_embd = d_model   # 512
        self.vert_anchors = vert_anchors   # 20
        self.horz_anchors = horz_anchors   # 20
        # d_k = d_model  # 512          # clw delete: no use, and value wrong，可以查看self.d_k
        # d_v = d_model  # 512

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))   # 20*20, 512 
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
        # self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')   # clw note: this has bug ??
        # self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        #self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        self.crosstransformer = nn.Sequential(*[CrossTransformerBlockForOnlyTir(d_model, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])  # clw modify

        # Concat
        # self.concat = Concat(dimension=1)

        # conv1x1
        #self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.conv1x1_out = Conv(c1=d_model, c2=d_model, k=1, s=1, p=0, g=1, act=True)   # clw modify TODO: 只用红外分支的CFE输出 
        

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, x):   # clw modify: 只使用红外分支的CFE模块
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        # import pdb; pdb.set_trace()
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        # import pdb; pdb.set_trace()
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis   # (h*w, c)，论文里的T_R

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir  # (h*w, c)，论文里的T_

        # rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])  # clw note: 这里相当于CFE模块在FFN前面的部分；
        ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])  

        # rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        # if self.training == True:
        #     rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        # else:
        #     rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        # new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        # new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_ir_fea)
        
        return new_fea




class CrossAttentionForOnlyTir(nn.Module):       # clw modify: 只输出红外部分
    #def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
    def __init__(self, d_model, h, attn_pdrop=.1, resid_pdrop=.1):   # clw modify TODO
        '''
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        '''
        super(CrossAttentionForOnlyTir, self).__init__()
        # assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h   # clw note: Dimensionality of queries and keys,  512 // 8 = 64
        self.d_v = d_model // h   #             Dimensionality of values
        assert self.d_k % h == 0
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)   # clw note TODO：论文里没有写LN
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        ir_fea_flat = self.LN2(ir_fea_flat)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)

        # get attention matrix      
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)   # clw note TODO: 默认dropout参数0.1 ??

        # output
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)    # clw note TODO: 默认dropout参数0.1 ??

        return out_ir
#############################################################




#####################################################
################ clw note: 原版实现
class CrossAttention(nn.Module):   
    #def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
    def __init__(self, d_model, h, attn_pdrop=.1, resid_pdrop=.1):   # clw modify TODO
        '''
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        # assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h   # clw note: Dimensionality of queries and keys,  512 // 8 = 64
        self.d_v = d_model // h   #             Dimensionality of values
        assert self.d_k % h == 0
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)   # clw note TODO：论文里没有写LN
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)          # clw note TODO: 默认dropout参数0.1 ??
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):    # clw note: CFE模块
    # def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
    def __init__(self, d_model, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        # self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.crossatt = CrossAttention(d_model, h, attn_pdrop, resid_pdrop)  # clw modify
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        # bs, nx, c = rgb_fea_flat.size()
        # h = w = int(math.sqrt(nx))

        for _ in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            # rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN1(rgb_att_out)))   # clw modify TODO
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

            # without Learnable Coefficient
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.LN2(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.LN2(ir_att_out))

        return [rgb_fea_flat, ir_fea_flat]


    
class TransformerFusionBlock(nn.Module):         # clw note: DMFF，原版实现； yolov5原版代码里面，三个stride特征层分别加了该模块，分别设置 [[512, 20, 20], [1024, 16, 16], [2048, 10, 10]]
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model   # 512
        self.vert_anchors = vert_anchors   # 20
        self.horz_anchors = horz_anchors   # 20
        # d_k = d_model  # 512          # clw delete: no use, and value wrong，可以查看self.d_k
        # d_v = d_model  # 512

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))   # 20*20, 512 
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')   # clw note: origin version
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        #self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])  # clw modify

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, rgb_fea, ir_fea):
    # def forward(self, x):
        # rgb_fea = x[0]
        # ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis   # (h*w, c)，论文里的T_R

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir  # (h*w, c)，论文里的T_

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])  # clw note: 这里相当于CFE模块在FFN前面的部分；

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        # import pdb; pdb.set_trace()
        if self.training == True:
            # rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')   # clw note TODO：为啥训练和推理插值方式不一样？并且训练用的低精度插值方式 nearest ?? 
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')     #              实测：训练时使用bilinear，推理时使用bilinear或者nearest精度基本没有差别
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
            # rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            # ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
            # ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        # ------------------------- feature visulization -----------------------#
        # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
        # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
        # fea_ir_new = torch.mean(new_ir_fea, dim=1)
        # fea_new = torch.mean(new_fea, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
        # -----------------------------------------------------------------------------#
        
        return new_fea




 
class TransformerFusionBlockV2(nn.Module):         # clw note: DMFF，原版实现； yolov5原版代码里面，三个stride特征层分别加了该模块，分别设置 [[512, 20, 20], [1024, 16, 16], [2048, 10, 10]]
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlockV2, self).__init__()

        self.n_embd = d_model   # 512
        self.vert_anchors = vert_anchors   # 20
        self.horz_anchors = horz_anchors   # 20

        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))   # 20*20, 512 
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')   # clw note: origin version
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        #self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])  # clw modify


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, rgb_fea, ir_fea):
    # def forward(self, x):
        # rgb_fea = x[0]
        # ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis   # (h*w, c)，论文里的T_R

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir  # (h*w, c)，论文里的T_

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])  # clw note: 这里相当于CFE模块在FFN前面的部分；

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        # import pdb; pdb.set_trace()
        if self.training == True:
            # rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')   # clw note TODO：为啥训练和推理插值方式不一样？并且训练用的低精度插值方式 nearest ?? 
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')     #              实测：训练时使用bilinear，推理时使用bilinear或者nearest精度基本没有差别
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
            # rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            # ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
            # ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        new_ir_fea = ir_fea_CFE + ir_fea

        return new_rgb_fea, new_ir_fea
############################################################################


    

class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

class Add(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x1,x2):
        return x1 * self.w + x2 * (1 - self.w)

#================================================IDAT=====================================================
class Attention_Crossmodal(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(Attention_Crossmodal, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()
        self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]
        lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
        lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
        lambda_init = lambda_init_fn(1)
        lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init

        lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
            rgb_fea_flat)
        lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
        lambda_init = lambda_init_fn(1)
        lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)


        att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)


        # output
        out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                       self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)

        return [out_vis, out_ir]

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
class Iterative_Differential_TransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(Iterative_Differential_TransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
        # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()
        self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        temp_rgb = rgb_fea_flat
        temp_ir = ir_fea_flat
        dif_fea_rgb = x[0]
        dif_fea_ir = x[1]
        lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
            rgb_fea_flat)
        lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
            rgb_fea_flat)
        lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
            rgb_fea_flat)
        lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
            rgb_fea_flat)


        for loop in range(self.loops):

            lambda_init = lambda_init_fn(loop)
            lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
            dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
            # dif_fea_ir = ir_fea_out - rgb_fea_out
            rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))

        for loop in range(self.loops):
            lambda_init = lambda_init_fn(loop)
            lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
            # with Learnable Coefficient
            temp_rgb_fea_out, temp_ir_fea_out = self.crossatt_rgb([temp_rgb, temp_ir])
            dif_fea_rgb = temp_rgb_fea_out - lambda_ir_full * temp_ir_fea_out
            # dif_fea_rgb = temp_rgb_fea_out - temp_ir_fea_out
            temp_ir = self.coefficient5(temp_ir) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))


        rgb_fea_final = rgb_fea_out + dif_fea_ir
        ir_fea_final = temp_ir_fea_out + dif_fea_rgb

        # return [dif_fea_rgb, dif_fea_ir]
        return [rgb_fea_final,ir_fea_final]


class Iterative_Differential_TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Iterative_Differential_TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])

        # Concat
        # self.concat = Concat(dimension=1)
        self.fused = NiNfusion(d_model)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, rgb_fea, ir_fea):
        # rgb_fea = x[0]
        # ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.fused(new_rgb_fea, new_ir_fea)
        # new_fea = self.conv1x1_out(new_fea)

        # ------------------------- feature visulization -----------------------#
        # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
        # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
        # fea_ir_new = torch.mean(new_ir_fea, dim=1)
        # fea_new = torch.mean(new_fea, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
        # -----------------------------------------------------------------------------#

        return new_fea

#
#     # ========================================================SAM================================================

# class Attention_Crossmodal(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(Attention_Crossmodal, self).__init__()
#         assert d_k % h == 0
#         self.d_model = d_model
#         self.d_k = d_model // h
#         self.d_v = d_model // h
#         self.h = h
#
#         # key, query, value projections for all heads
#         self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
#         self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection
#
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#
#         # layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#
#         self.init_weights()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes Self-Attention
#         Args:
#             x (tensor): input (token) dim:(b_s, nx, c),
#                 b_s means batch size
#                 nx means length, for CNN, equals H*W, i.e. the length of feature maps
#                 c means channel, i.e. the channel of feature maps
#             attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#             attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         Return:
#             output (tensor): dim:(b_s, nx, c)
#         '''
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         b_s, nq = rgb_fea_flat.shape[:2]
#         nk = rgb_fea_flat.shape[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#
#         # Self-Attention
#         rgb_fea_flat = self.LN1(rgb_fea_flat)
#         q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         ir_fea_flat = self.LN2(ir_fea_flat)
#         q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#
#         att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
#         att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)
#
#         # get attention matrix
#         att_vis = torch.softmax(att_vis, -1)
#         att_vis = self.attn_drop(att_vis)
#         att_ir = torch.softmax(att_ir, -1)
#         att_ir = self.attn_drop(att_ir)
#
#
#         # output
#         out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
#         out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
#         out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
#                                                                                        self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
#         out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)
#
#         return [out_vis, out_ir]
#
# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)
# class Iterative_Differential_TransformerBlock(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
#         """
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         :param block_exp: Expansion factor for MLP (feed foreword network)
#         """
#         super(Iterative_Differential_TransformerBlock, self).__init__()
#         self.loops = loops_num
#         self.ln_input = nn.LayerNorm(d_model)
#         self.ln_output = nn.LayerNorm(d_model)
#         self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                      # nn.SiLU(),  # changed from GELU
#                                      nn.GELU(),  # changed from GELU
#                                      nn.Linear(block_exp * d_model, d_model),
#                                      nn.Dropout(resid_pdrop),
#                                      )
#         self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                     # nn.SiLU(),  # changed from GELU
#                                     nn.GELU(),  # changed from GELU
#                                     nn.Linear(block_exp * d_model, d_model),
#                                     nn.Dropout(resid_pdrop),
#                                     )
#         self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                  # nn.SiLU(),  # changed from GELU
#                                  nn.GELU(),  # changed from GELU
#                                  nn.Linear(block_exp * d_model, d_model),
#                                  nn.Dropout(resid_pdrop),
#                                  )
#
#         # Layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#         # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
#         # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)
#
#         # Learnable Coefficient
#         # self.coefficient1 = LearnableCoefficient()
#         # self.coefficient2 = LearnableCoefficient()
#         # self.coefficient3 = LearnableCoefficient()
#         # self.coefficient4 = LearnableCoefficient()
#         # self.coefficient5 = LearnableCoefficient()
#         # self.coefficient6 = LearnableCoefficient()
#         # self.coefficient7 = LearnableCoefficient()
#         # self.coefficient8 = LearnableCoefficient()
#         # self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         # self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         # self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         # self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         # self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         # self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         # self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         # self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def forward(self, x):
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         # assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
#         # temp_rgb = rgb_fea_flat
#         # temp_ir = ir_fea_flat
#         # dif_fea_rgb = x[0]
#         # dif_fea_ir = x[1]
#         # lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
#         #     rgb_fea_flat)
#         # lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
#         #     rgb_fea_flat)
#         # lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#         #     rgb_fea_flat)
#         # lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
#         #     rgb_fea_flat)
#
#         rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
#         # for loop in range(self.loops):
#         #
#         #     lambda_init = lambda_init_fn(loop)
#         #     lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#         #     # with Learnable Coefficient
#         #     rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
#         #     dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
#         #     # dif_fea_ir = ir_fea_out - rgb_fea_out
#         #     rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))
#         #
#         # for loop in range(self.loops):
#         #     lambda_init = lambda_init_fn(loop)
#         #     lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#         #     # with Learnable Coefficient
#         #     temp_rgb_fea_out, temp_ir_fea_out = self.crossatt_rgb([temp_rgb, temp_ir])
#         #     dif_fea_rgb = temp_rgb_fea_out - lambda_ir_full * temp_ir_fea_out
#         #     # dif_fea_rgb = temp_rgb_fea_out - temp_ir_fea_out
#         #     temp_ir = self.coefficient5(temp_ir) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))
#
#
#         # rgb_fea_final = rgb_fea_out + dif_fea_ir
#         # ir_fea_final = temp_ir_fea_out + dif_fea_rgb
#
#         # return [dif_fea_rgb, dif_fea_ir]
#         return [rgb_fea_out,ir_fea_out]
#
#
# class Iterative_Differential_TransformerFusionBlock(nn.Module):
#     def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super(Iterative_Differential_TransformerFusionBlock, self).__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#         self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         # downsampling
#         # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
#         # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
#
#         self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
#         self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
#
#         # LearnableCoefficient
#         self.vis_coefficient = LearnableWeights()
#         self.ir_coefficient = LearnableWeights()
#
#         # init weights
#         self.apply(self._init_weights)
#
#         # cross transformer
#         self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
#
#         # Concat
#         # self.concat = Concat(dimension=1)
#         self.fused = NiNfusion(d_model)
#
#         # conv1x1
#         self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, rgb_fea,ir_fea):
#
#         assert rgb_fea.shape[0] == ir_fea.shape[0]
#         bs, c, h, w = rgb_fea.shape
#
#         # ------------------------- cross-modal feature fusion -----------------------#
#         #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
#         new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
#         new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
#         rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
#
#         #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
#         new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
#         ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
#
#         rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
#
#         rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
#         new_rgb_fea = rgb_fea_CFE + rgb_fea
#         ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
#         new_ir_fea = ir_fea_CFE + ir_fea
#
#         new_fea = self.fused(new_rgb_fea, new_ir_fea)
#         # new_fea = self.conv1x1_out(new_fea)
#
#         # ------------------------- feature visulization -----------------------#
#         # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
#         # fea_rgb = torch.mean(rgb_fea, dim=1)
#         # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
#         # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
#         # fea_ir = torch.mean(ir_fea, dim=1)
#         # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
#         # fea_ir_new = torch.mean(new_ir_fea, dim=1)
#         # fea_new = torch.mean(new_fea, dim=1)
#         # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
#         # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
#         # plt.figure()
#         # for i in range(len(block)):
#         #     feature = transforms.ToPILImage()(block[i].squeeze())
#         #     ax = plt.subplot(3, 3, i + 1)
#         #     ax.set_xticks([])
#         #     ax.set_yticks([])
#         #     ax.set_title(black_name[i], fontsize=8)
#         #     plt.imshow(feature)
#         # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
#         # -----------------------------------------------------------------------------#
#
#         return new_fea




#=====================================IDFM===================================================
#
# class Attention_Crossmodal(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(Attention_Crossmodal, self).__init__()
#         assert d_k % h == 0
#         self.d_model = d_model
#         self.d_k = d_model // h
#         self.d_v = d_model // h
#         self.h = h
#
#         # key, query, value projections for all heads
#         self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
#         self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection
#
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#
#         # layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#
#         self.init_weights()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes Self-Attention
#         Args:
#             x (tensor): input (token) dim:(b_s, nx, c),
#                 b_s means batch size
#                 nx means length, for CNN, equals H*W, i.e. the length of feature maps
#                 c means channel, i.e. the channel of feature maps
#             attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#             attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         Return:
#             output (tensor): dim:(b_s, nx, c)
#         '''
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         b_s, nq = rgb_fea_flat.shape[:2]
#         nk = rgb_fea_flat.shape[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#
#         # Self-Attention
#         rgb_fea_flat = self.LN1(rgb_fea_flat)
#         q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         ir_fea_flat = self.LN2(ir_fea_flat)
#         q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#
#         att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
#         att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)
#
#         # get attention matrix
#         att_vis = torch.softmax(att_vis, -1)
#         att_vis = self.attn_drop(att_vis)
#         att_ir = torch.softmax(att_ir, -1)
#         att_ir = self.attn_drop(att_ir)
#
#
#         # output
#         out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
#         out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
#         out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
#                                                                                        self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
#         out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)
#
#         return [out_vis, out_ir]
#
# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)
# class Iterative_Differential_TransformerBlock(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
#         """
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         :param block_exp: Expansion factor for MLP (feed foreword network)
#         """
#         super(Iterative_Differential_TransformerBlock, self).__init__()
#         self.loops = loops_num
#         self.ln_input = nn.LayerNorm(d_model)
#         self.ln_output = nn.LayerNorm(d_model)
#         # self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         # self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                      # nn.SiLU(),  # changed from GELU
#                                      nn.GELU(),  # changed from GELU
#                                      nn.Linear(block_exp * d_model, d_model),
#                                      nn.Dropout(resid_pdrop),
#                                      )
#         self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                     # nn.SiLU(),  # changed from GELU
#                                     nn.GELU(),  # changed from GELU
#                                     nn.Linear(block_exp * d_model, d_model),
#                                     nn.Dropout(resid_pdrop),
#                                     )
#         self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                  # nn.SiLU(),  # changed from GELU
#                                  nn.GELU(),  # changed from GELU
#                                  nn.Linear(block_exp * d_model, d_model),
#                                  nn.Dropout(resid_pdrop),
#                                  )
#
#         # Layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#         # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
#         # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)
#
#         # Learnable Coefficient
#         self.coefficient1 = LearnableCoefficient()
#         self.coefficient2 = LearnableCoefficient()
#         self.coefficient3 = LearnableCoefficient()
#         self.coefficient4 = LearnableCoefficient()
#         self.coefficient5 = LearnableCoefficient()
#         self.coefficient6 = LearnableCoefficient()
#         self.coefficient7 = LearnableCoefficient()
#         self.coefficient8 = LearnableCoefficient()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def forward(self, x):
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         # assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
#         temp_rgb = rgb_fea_flat
#         temp_ir = ir_fea_flat
#         dif_fea_rgb = x[0]
#         dif_fea_ir = x[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#
#         for loop in range(self.loops):
#
#             lambda_init = lambda_init_fn(loop)
#             lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#             # with Learnable Coefficient
#             rgb_fea_out = rgb_fea_flat
#             ir_fea_out = ir_fea_flat
#             dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
#             # dif_fea_ir = ir_fea_out - rgb_fea_out
#             rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))
#
#         for loop in range(self.loops):
#             lambda_init = lambda_init_fn(loop)
#             lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#             # with Learnable Coefficient
#             temp_rgb_fea_out =temp_rgb
#             temp_ir_fea_out = temp_ir
#             dif_fea_rgb = temp_rgb_fea_out - lambda_ir_full * temp_ir_fea_out
#             # dif_fea_rgb = temp_rgb_fea_out - temp_ir_fea_out
#             temp_ir = self.coefficient5(temp_ir) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))
#
#
#         rgb_fea_final = rgb_fea_out + dif_fea_ir
#         ir_fea_final = temp_ir_fea_out + dif_fea_rgb
#
#         # return [dif_fea_rgb, dif_fea_ir]
#         return [rgb_fea_final,ir_fea_final]
#
#
# class Iterative_Differential_TransformerFusionBlock(nn.Module):
#     def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super(Iterative_Differential_TransformerFusionBlock, self).__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#         self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         # downsampling
#         # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
#         # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
#
#         self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
#         self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
#
#         # LearnableCoefficient
#         self.vis_coefficient = LearnableWeights()
#         self.ir_coefficient = LearnableWeights()
#
#         # init weights
#         self.apply(self._init_weights)
#
#         # cross transformer
#         self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
#
#         # Concat
#         # self.concat = Concat(dimension=1)
#         self.fused = NiNfusion(d_model)
#
#         # conv1x1
#         self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, rgb_fea, ir_fea):
#         assert rgb_fea.shape[0] == ir_fea.shape[0]
#         bs, c, h, w = rgb_fea.shape
#
#         # ------------------------- cross-modal feature fusion -----------------------#
#         #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
#         new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
#         new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
#         rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
#
#         #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
#         new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
#         ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
#
#         rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
#
#         rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
#         new_rgb_fea = rgb_fea_CFE + rgb_fea
#         ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
#         new_ir_fea = ir_fea_CFE + ir_fea
#
#         new_fea = self.fused(new_rgb_fea, new_ir_fea)
#         # new_fea = self.conv1x1_out(new_fea)
#
#         # ------------------------- feature visulization -----------------------#
#         # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
#         # fea_rgb = torch.mean(rgb_fea, dim=1)
#         # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
#         # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
#         # fea_ir = torch.mean(ir_fea, dim=1)
#         # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
#         # fea_ir_new = torch.mean(new_ir_fea, dim=1)
#         # fea_new = torch.mean(new_fea, dim=1)
#         # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
#         # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
#         # plt.figure()
#         # for i in range(len(block)):
#         #     feature = transforms.ToPILImage()(block[i].squeeze())
#         #     ax = plt.subplot(3, 3, i + 1)
#         #     ax.set_xticks([])
#         #     ax.set_yticks([])
#         #     ax.set_title(black_name[i], fontsize=8)
#         #     plt.imshow(feature)
#         # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
#         # -----------------------------------------------------------------------------#
#
#         return new_fea

# =========================================ISDF_RGB====================================
#
# class Attention_Crossmodal(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(Attention_Crossmodal, self).__init__()
#         assert d_k % h == 0
#         self.d_model = d_model
#         self.d_k = d_model // h
#         self.d_v = d_model // h
#         self.h = h
#
#         # key, query, value projections for all heads
#         self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
#         self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection
#
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#
#         # layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#
#         self.init_weights()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes Self-Attention
#         Args:
#             x (tensor): input (token) dim:(b_s, nx, c),
#                 b_s means batch size
#                 nx means length, for CNN, equals H*W, i.e. the length of feature maps
#                 c means channel, i.e. the channel of feature maps
#             attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#             attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         Return:
#             output (tensor): dim:(b_s, nx, c)
#         '''
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         b_s, nq = rgb_fea_flat.shape[:2]
#         nk = rgb_fea_flat.shape[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#
#         # Self-Attention
#         rgb_fea_flat = self.LN1(rgb_fea_flat)
#         q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         ir_fea_flat = self.LN2(ir_fea_flat)
#         q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#
#         att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
#         att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)
#
#         # get attention matrix
#         att_vis = torch.softmax(att_vis, -1)
#         att_vis = self.attn_drop(att_vis)
#         att_ir = torch.softmax(att_ir, -1)
#         att_ir = self.attn_drop(att_ir)
#
#
#         # output
#         out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
#         out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
#         out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
#                                                                                        self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
#         out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)
#
#         return [out_vis, out_ir]
#
# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)
# class Iterative_Differential_TransformerBlock(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
#         """
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         :param block_exp: Expansion factor for MLP (feed foreword network)
#         """
#         super(Iterative_Differential_TransformerBlock, self).__init__()
#         self.loops = loops_num
#         self.ln_input = nn.LayerNorm(d_model)
#         self.ln_output = nn.LayerNorm(d_model)
#         self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                      # nn.SiLU(),  # changed from GELU
#                                      nn.GELU(),  # changed from GELU
#                                      nn.Linear(block_exp * d_model, d_model),
#                                      nn.Dropout(resid_pdrop),
#                                      )
#         self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                     # nn.SiLU(),  # changed from GELU
#                                     nn.GELU(),  # changed from GELU
#                                     nn.Linear(block_exp * d_model, d_model),
#                                     nn.Dropout(resid_pdrop),
#                                     )
#         self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                  # nn.SiLU(),  # changed from GELU
#                                  nn.GELU(),  # changed from GELU
#                                  nn.Linear(block_exp * d_model, d_model),
#                                  nn.Dropout(resid_pdrop),
#                                  )
#
#         # Layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#         # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
#         # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)
#
#         # Learnable Coefficient
#         self.coefficient1 = LearnableCoefficient()
#         self.coefficient2 = LearnableCoefficient()
#         self.coefficient3 = LearnableCoefficient()
#         self.coefficient4 = LearnableCoefficient()
#         self.coefficient5 = LearnableCoefficient()
#         self.coefficient6 = LearnableCoefficient()
#         self.coefficient7 = LearnableCoefficient()
#         self.coefficient8 = LearnableCoefficient()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def forward(self, x):
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
#         temp_rgb = rgb_fea_flat
#         temp_ir = ir_fea_flat
#         dif_fea_rgb = x[0]
#         dif_fea_ir = x[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#
#
#         for loop in range(self.loops):
#
#             lambda_init = lambda_init_fn(loop)
#             lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#             # with Learnable Coefficient
#             rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
#             dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
#             # dif_fea_ir = ir_fea_out - rgb_fea_out
#             rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))
#
#         # for loop in range(self.loops):
#         #     lambda_init = lambda_init_fn(loop)
#         #     lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#         #     # with Learnable Coefficient
#         #     temp_rgb_fea_out, temp_ir_fea_out = self.crossatt_rgb([temp_rgb, temp_ir])
#         #     dif_fea_rgb = temp_rgb_fea_out - lambda_ir_full * temp_ir_fea_out
#         #     # dif_fea_rgb = temp_rgb_fea_out - temp_ir_fea_out
#         #     temp_ir = self.coefficient5(temp_ir) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))
#
#
#         rgb_fea_final = rgb_fea_out + dif_fea_ir
#         # ir_fea_final = temp_ir_fea_out + dif_fea_rgb
#
#         # return [dif_fea_rgb, dif_fea_ir]
#         return [rgb_fea_final,ir_fea_out]
#
#
# class Iterative_Differential_TransformerFusionBlock_RGB(nn.Module):
#     def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super(Iterative_Differential_TransformerFusionBlock_RGB, self).__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#         self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         # downsampling
#         # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
#         # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
#
#         self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
#         self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
#
#         # LearnableCoefficient
#         self.vis_coefficient = LearnableWeights()
#         self.ir_coefficient = LearnableWeights()
#
#         # init weights
#         self.apply(self._init_weights)
#
#         # cross transformer
#         self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
#
#         # Concat
#         # self.concat = Concat(dimension=1)
#         self.fused = NiNfusion(d_model)
#
#         # conv1x1
#         self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, rgb_fea, ir_fea):
#         # rgb_fea = x[0]
#         # ir_fea = x[1]
#         assert rgb_fea.shape[0] == ir_fea.shape[0]
#         bs, c, h, w = rgb_fea.shape
#
#         # ------------------------- cross-modal feature fusion -----------------------#
#         #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
#         new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
#         new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
#         rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
#
#         #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
#         new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
#         ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
#
#         rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
#
#         rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
#         new_rgb_fea = rgb_fea_CFE + rgb_fea
#         # ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         # if self.training == True:
#         #     ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
#         # else:
#         #     ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
#         # new_ir_fea = ir_fea_CFE + ir_fea
#         #
#         # new_fea = self.fused(new_rgb_fea, new_ir_fea)
#         # new_fea = self.conv1x1_out(new_fea)
#
#
#         return new_rgb_fea

# =================================ISDF_IR==========================================
# class Attention_Crossmodal(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(Attention_Crossmodal, self).__init__()
#         assert d_k % h == 0
#         self.d_model = d_model
#         self.d_k = d_model // h
#         self.d_v = d_model // h
#         self.h = h
#
#         # key, query, value projections for all heads
#         self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
#         self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection
#
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#
#         # layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#
#         self.init_weights()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes Self-Attention
#         Args:
#             x (tensor): input (token) dim:(b_s, nx, c),
#                 b_s means batch size
#                 nx means length, for CNN, equals H*W, i.e. the length of feature maps
#                 c means channel, i.e. the channel of feature maps
#             attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#             attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         Return:
#             output (tensor): dim:(b_s, nx, c)
#         '''
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         b_s, nq = rgb_fea_flat.shape[:2]
#         nk = rgb_fea_flat.shape[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#
#         # Self-Attention
#         rgb_fea_flat = self.LN1(rgb_fea_flat)
#         q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         ir_fea_flat = self.LN2(ir_fea_flat)
#         q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#
#         att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
#         att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)
#
#         # get attention matrix
#         att_vis = torch.softmax(att_vis, -1)
#         att_vis = self.attn_drop(att_vis)
#         att_ir = torch.softmax(att_ir, -1)
#         att_ir = self.attn_drop(att_ir)
#
#
#         # output
#         out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
#         out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
#         out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
#                                                                                        self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
#         out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)
#
#         return [out_vis, out_ir]
#
# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)
# class Iterative_Differential_TransformerBlock(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
#         """
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         :param block_exp: Expansion factor for MLP (feed foreword network)
#         """
#         super(Iterative_Differential_TransformerBlock, self).__init__()
#         self.loops = loops_num
#         self.ln_input = nn.LayerNorm(d_model)
#         self.ln_output = nn.LayerNorm(d_model)
#         self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                      # nn.SiLU(),  # changed from GELU
#                                      nn.GELU(),  # changed from GELU
#                                      nn.Linear(block_exp * d_model, d_model),
#                                      nn.Dropout(resid_pdrop),
#                                      )
#         self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                     # nn.SiLU(),  # changed from GELU
#                                     nn.GELU(),  # changed from GELU
#                                     nn.Linear(block_exp * d_model, d_model),
#                                     nn.Dropout(resid_pdrop),
#                                     )
#         self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                  # nn.SiLU(),  # changed from GELU
#                                  nn.GELU(),  # changed from GELU
#                                  nn.Linear(block_exp * d_model, d_model),
#                                  nn.Dropout(resid_pdrop),
#                                  )
#
#         # Layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#         # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
#         # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)
#
#         # Learnable Coefficient
#         self.coefficient1 = LearnableCoefficient()
#         self.coefficient2 = LearnableCoefficient()
#         self.coefficient3 = LearnableCoefficient()
#         self.coefficient4 = LearnableCoefficient()
#         self.coefficient5 = LearnableCoefficient()
#         self.coefficient6 = LearnableCoefficient()
#         self.coefficient7 = LearnableCoefficient()
#         self.coefficient8 = LearnableCoefficient()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def forward(self, x):
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
#         temp_rgb = rgb_fea_flat
#         temp_ir = ir_fea_flat
#         dif_fea_rgb = x[0]
#         dif_fea_ir = x[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#
#
#         # for loop in range(self.loops):
#         #
#         #     lambda_init = lambda_init_fn(loop)
#         #     lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#         #     # with Learnable Coefficient
#         #     rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
#         #     dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
#         #     # dif_fea_ir = ir_fea_out - rgb_fea_out
#         #     rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))
#
#         for loop in range(self.loops):
#             lambda_init = lambda_init_fn(loop)
#             lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#             # with Learnable Coefficient
#             temp_rgb_fea_out, temp_ir_fea_out = self.crossatt_rgb([temp_rgb, temp_ir])
#             dif_fea_rgb = temp_rgb_fea_out - lambda_ir_full * temp_ir_fea_out
#             # dif_fea_rgb = temp_rgb_fea_out - temp_ir_fea_out
#             temp_ir = self.coefficient5(temp_ir) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))
#
#
#         # rgb_fea_final = rgb_fea_out + dif_fea_ir
#         ir_fea_final = temp_ir_fea_out + dif_fea_rgb
#
#         # return [dif_fea_rgb, dif_fea_ir]
#         return [temp_rgb_fea_out,ir_fea_final]
#
#
# class Iterative_Differential_TransformerFusionBlock_IR(nn.Module):
#     def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super(Iterative_Differential_TransformerFusionBlock_IR, self).__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#         self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         # downsampling
#         # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
#         # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
#
#         self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
#         self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
#
#         # LearnableCoefficient
#         self.vis_coefficient = LearnableWeights()
#         self.ir_coefficient = LearnableWeights()
#
#         # init weights
#         self.apply(self._init_weights)
#
#         # cross transformer
#         self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
#
#         # Concat
#         # self.concat = Concat(dimension=1)
#         self.fused = NiNfusion(d_model)
#
#         # conv1x1
#         self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, rgb_fea, ir_fea):
#         # rgb_fea = x[0]
#         # ir_fea = x[1]
#         assert rgb_fea.shape[0] == ir_fea.shape[0]
#         bs, c, h, w = rgb_fea.shape
#
#         # ------------------------- cross-modal feature fusion -----------------------#
#         #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
#         new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
#         new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
#         rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
#
#         #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
#         new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
#         ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
#
#         rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
#
#         # rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         # if self.training == True:
#         #     rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
#         # else:
#         #     rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
#         # new_rgb_fea = rgb_fea_CFE + rgb_fea
#         ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
#         new_ir_fea = ir_fea_CFE + ir_fea
#         #
#         # new_fea = self.fused(new_rgb_fea, new_ir_fea)
#         # new_fea = self.conv1x1_out(new_fea)
#
#
#         return new_ir_fea
# ===========================================ISDF_ADD================================
# class Attention_Crossmodal(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(Attention_Crossmodal, self).__init__()
#         assert d_k % h == 0
#         self.d_model = d_model
#         self.d_k = d_model // h
#         self.d_v = d_model // h
#         self.h = h
#
#         # key, query, value projections for all heads
#         self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
#         self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection
#
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#
#         # layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#
#         self.init_weights()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes Self-Attention
#         Args:
#             x (tensor): input (token) dim:(b_s, nx, c),
#                 b_s means batch size
#                 nx means length, for CNN, equals H*W, i.e. the length of feature maps
#                 c means channel, i.e. the channel of feature maps
#             attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#             attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         Return:
#             output (tensor): dim:(b_s, nx, c)
#         '''
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         b_s, nq = rgb_fea_flat.shape[:2]
#         nk = rgb_fea_flat.shape[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#
#         # Self-Attention
#         rgb_fea_flat = self.LN1(rgb_fea_flat)
#         q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         ir_fea_flat = self.LN2(ir_fea_flat)
#         q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#
#         att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
#         att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)
#
#         # get attention matrix
#         att_vis = torch.softmax(att_vis, -1)
#         att_vis = self.attn_drop(att_vis)
#         att_ir = torch.softmax(att_ir, -1)
#         att_ir = self.attn_drop(att_ir)
#
#
#         # output
#         out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
#         out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
#         out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
#                                                                                        self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
#         out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)
#
#         return [out_vis, out_ir]
#
# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)
# class Iterative_Differential_TransformerBlock(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
#         """
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         :param block_exp: Expansion factor for MLP (feed foreword network)
#         """
#         super(Iterative_Differential_TransformerBlock, self).__init__()
#         self.loops = loops_num
#         self.ln_input = nn.LayerNorm(d_model)
#         self.ln_output = nn.LayerNorm(d_model)
#         self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                      # nn.SiLU(),  # changed from GELU
#                                      nn.GELU(),  # changed from GELU
#                                      nn.Linear(block_exp * d_model, d_model),
#                                      nn.Dropout(resid_pdrop),
#                                      )
#         self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                     # nn.SiLU(),  # changed from GELU
#                                     nn.GELU(),  # changed from GELU
#                                     nn.Linear(block_exp * d_model, d_model),
#                                     nn.Dropout(resid_pdrop),
#                                     )
#         self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                  # nn.SiLU(),  # changed from GELU
#                                  nn.GELU(),  # changed from GELU
#                                  nn.Linear(block_exp * d_model, d_model),
#                                  nn.Dropout(resid_pdrop),
#                                  )
#
#         # Layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#         # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
#         # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)
#
#         # Learnable Coefficient
#         self.coefficient1 = LearnableCoefficient()
#         self.coefficient2 = LearnableCoefficient()
#         self.coefficient3 = LearnableCoefficient()
#         self.coefficient4 = LearnableCoefficient()
#         self.coefficient5 = LearnableCoefficient()
#         self.coefficient6 = LearnableCoefficient()
#         self.coefficient7 = LearnableCoefficient()
#         self.coefficient8 = LearnableCoefficient()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def forward(self, x):
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
#         temp_rgb = rgb_fea_flat
#         temp_ir = ir_fea_flat
#         dif_fea_rgb = x[0]
#         dif_fea_ir = x[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#
#
#         for loop in range(self.loops):
#
#             lambda_init = lambda_init_fn(loop)
#             lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#             # with Learnable Coefficient
#             rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
#             dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
#             # dif_fea_ir = ir_fea_out - rgb_fea_out
#             rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))
#
#         for loop in range(self.loops):
#             lambda_init = lambda_init_fn(loop)
#             lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#             # with Learnable Coefficient
#             temp_rgb_fea_out, temp_ir_fea_out = self.crossatt_rgb([temp_rgb, temp_ir])
#             dif_fea_rgb = temp_rgb_fea_out - lambda_ir_full * temp_ir_fea_out
#             # dif_fea_rgb = temp_rgb_fea_out - temp_ir_fea_out
#             temp_ir = self.coefficient5(temp_ir) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))
#
#
#         rgb_fea_final = rgb_fea_out + dif_fea_ir
#         ir_fea_final = temp_ir_fea_out + dif_fea_rgb
#
#         # return [dif_fea_rgb, dif_fea_ir]
#         return [rgb_fea_final,ir_fea_final]
#
#
# class Iterative_Differential_TransformerFusionBlock_ADD(nn.Module):
#     def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super(Iterative_Differential_TransformerFusionBlock_ADD, self).__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#         self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         # downsampling
#         # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
#         # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
#
#         self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
#         self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
#
#         # LearnableCoefficient
#         self.vis_coefficient = LearnableWeights()
#         self.ir_coefficient = LearnableWeights()
#
#         # init weights
#         self.apply(self._init_weights)
#
#         # cross transformer
#         self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
#
#         # Concat
#         # self.concat = Concat(dimension=1)
#         # self.fused = NiNfusion(d_model)
#
#         # conv1x1
#         self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, rgb_fea, ir_fea):
#         # rgb_fea = x[0]
#         # ir_fea = x[1]
#         assert rgb_fea.shape[0] == ir_fea.shape[0]
#         bs, c, h, w = rgb_fea.shape
#
#         # ------------------------- cross-modal feature fusion -----------------------#
#         #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
#         new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
#         new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
#         rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
#
#         #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
#         new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
#         ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
#
#         rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
#
#         rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
#         new_rgb_fea = rgb_fea_CFE + rgb_fea
#         ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
#         new_ir_fea = ir_fea_CFE + ir_fea
#
#         new_fea = new_rgb_fea + new_ir_fea
#         # new_fea = self.conv1x1_out(new_fea)
#
#         # ------------------------- feature visulization -----------------------#
#         # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
#         # fea_rgb = torch.mean(rgb_fea, dim=1)
#         # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
#         # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
#         # fea_ir = torch.mean(ir_fea, dim=1)
#         # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
#         # fea_ir_new = torch.mean(new_ir_fea, dim=1)
#         # fea_new = torch.mean(new_fea, dim=1)
#         # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
#         # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
#         # plt.figure()
#         # for i in range(len(block)):
#         #     feature = transforms.ToPILImage()(block[i].squeeze())
#         #     ax = plt.subplot(3, 3, i + 1)
#         #     ax.set_xticks([])
#         #     ax.set_yticks([])
#         #     ax.set_title(black_name[i], fontsize=8)
#         #     plt.imshow(feature)
#         # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
#         # -----------------------------------------------------------------------------#
#
#         return new_fea


# ======================ISDF_cross_V-no_hybrid_V
# class Attention_Crossmodal(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(Attention_Crossmodal, self).__init__()
#         assert d_k % h == 0
#         self.d_model = d_model
#         self.d_k = d_model // h
#         self.d_v = d_model // h
#         self.h = h
#
#         # key, query, value projections for all heads
#         self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
#         self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection
#
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#
#         # layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#
#         self.init_weights()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes Self-Attention
#         Args:
#             x (tensor): input (token) dim:(b_s, nx, c),
#                 b_s means batch size
#                 nx means length, for CNN, equals H*W, i.e. the length of feature maps
#                 c means channel, i.e. the channel of feature maps
#             attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#             attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         Return:
#             output (tensor): dim:(b_s, nx, c)
#         '''
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         b_s, nq = rgb_fea_flat.shape[:2]
#         nk = rgb_fea_flat.shape[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#
#         # Self-Attention
#         rgb_fea_flat = self.LN1(rgb_fea_flat)
#         q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         ir_fea_flat = self.LN2(ir_fea_flat)
#         q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#
#         att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
#         att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)
#
#         # get attention matrix
#         att_vis = torch.softmax(att_vis, -1)
#         att_vis = self.attn_drop(att_vis)
#         att_ir = torch.softmax(att_ir, -1)
#         att_ir = self.attn_drop(att_ir)
#
#
#         # output
#         out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         # out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         # out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
#         out_vis = out_vis_ir
#         out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
#         out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         # out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
#         #                                                                                self.h * self.d_v)  # (b_s, nq, h*d_v)
#         # out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
#         out_ir = out_ir_vis
#         out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)
#
#         return [out_vis, out_ir]
#
# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)
# class Iterative_Differential_TransformerBlock(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
#         """
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         :param block_exp: Expansion factor for MLP (feed foreword network)
#         """
#         super(Iterative_Differential_TransformerBlock, self).__init__()
#         self.loops = loops_num
#         self.ln_input = nn.LayerNorm(d_model)
#         self.ln_output = nn.LayerNorm(d_model)
#         self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                      # nn.SiLU(),  # changed from GELU
#                                      nn.GELU(),  # changed from GELU
#                                      nn.Linear(block_exp * d_model, d_model),
#                                      nn.Dropout(resid_pdrop),
#                                      )
#         self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                     # nn.SiLU(),  # changed from GELU
#                                     nn.GELU(),  # changed from GELU
#                                     nn.Linear(block_exp * d_model, d_model),
#                                     nn.Dropout(resid_pdrop),
#                                     )
#         self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                  # nn.SiLU(),  # changed from GELU
#                                  nn.GELU(),  # changed from GELU
#                                  nn.Linear(block_exp * d_model, d_model),
#                                  nn.Dropout(resid_pdrop),
#                                  )
#
#         # Layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#         # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
#         # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)
#
#         # Learnable Coefficient
#         self.coefficient1 = LearnableCoefficient()
#         self.coefficient2 = LearnableCoefficient()
#         self.coefficient3 = LearnableCoefficient()
#         self.coefficient4 = LearnableCoefficient()
#         self.coefficient5 = LearnableCoefficient()
#         self.coefficient6 = LearnableCoefficient()
#         self.coefficient7 = LearnableCoefficient()
#         self.coefficient8 = LearnableCoefficient()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def forward(self, x):
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
#         temp_rgb = rgb_fea_flat
#         temp_ir = ir_fea_flat
#         dif_fea_rgb = x[0]
#         dif_fea_ir = x[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#
#
#         for loop in range(self.loops):
#
#             lambda_init = lambda_init_fn(loop)
#             lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#             # with Learnable Coefficient
#             rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
#             dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
#             # dif_fea_ir = ir_fea_out - rgb_fea_out
#             rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))
#
#         for loop in range(self.loops):
#             lambda_init = lambda_init_fn(loop)
#             lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#             # with Learnable Coefficient
#             temp_rgb_fea_out, temp_ir_fea_out = self.crossatt_rgb([temp_rgb, temp_ir])
#             dif_fea_rgb = temp_rgb_fea_out - lambda_ir_full * temp_ir_fea_out
#             # dif_fea_rgb = temp_rgb_fea_out - temp_ir_fea_out
#             temp_ir = self.coefficient5(temp_ir) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))
#
#
#         rgb_fea_final = rgb_fea_out + dif_fea_ir
#         ir_fea_final = temp_ir_fea_out + dif_fea_rgb
#
#         # return [dif_fea_rgb, dif_fea_ir]
#         return [rgb_fea_final,ir_fea_final]
#
#
# class Iterative_Differential_TransformerFusionBlock_crossV(nn.Module):
#     def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super(Iterative_Differential_TransformerFusionBlock_crossV, self).__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#         self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         # downsampling
#         # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
#         # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
#
#         self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
#         self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
#
#         # LearnableCoefficient
#         self.vis_coefficient = LearnableWeights()
#         self.ir_coefficient = LearnableWeights()
#
#         # init weights
#         self.apply(self._init_weights)
#
#         # cross transformer
#         self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
#
#         # Concat
#         # self.concat = Concat(dimension=1)
#         self.fused = NiNfusion(d_model)
#
#         # conv1x1
#         self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, rgb_fea, ir_fea):
#         # rgb_fea = x[0]
#         # ir_fea = x[1]
#         assert rgb_fea.shape[0] == ir_fea.shape[0]
#         bs, c, h, w = rgb_fea.shape
#
#         # ------------------------- cross-modal feature fusion -----------------------#
#         #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
#         new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
#         new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
#         rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
#
#         #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
#         new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
#         ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
#
#         rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
#
#         rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
#         new_rgb_fea = rgb_fea_CFE + rgb_fea
#         ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
#         new_ir_fea = ir_fea_CFE + ir_fea
#
#         new_fea = self.fused(new_rgb_fea, new_ir_fea)
#         # new_fea = self.conv1x1_out(new_fea)
#
#         # ------------------------- feature visulization -----------------------#
#         # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
#         # fea_rgb = torch.mean(rgb_fea, dim=1)
#         # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
#         # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
#         # fea_ir = torch.mean(ir_fea, dim=1)
#         # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
#         # fea_ir_new = torch.mean(new_ir_fea, dim=1)
#         # fea_new = torch.mean(new_fea, dim=1)
#         # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
#         # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
#         # plt.figure()
#         # for i in range(len(block)):
#         #     feature = transforms.ToPILImage()(block[i].squeeze())
#         #     ax = plt.subplot(3, 3, i + 1)
#         #     ax.set_xticks([])
#         #     ax.set_yticks([])
#         #     ax.set_title(black_name[i], fontsize=8)
#         #     plt.imshow(feature)
#         # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
#         # -----------------------------------------------------------------------------#
#
#         return new_fea

# ============================dual feedback========================

# class Attention_Crossmodal(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(Attention_Crossmodal, self).__init__()
#         assert d_k % h == 0
#         self.d_model = d_model
#         self.d_k = d_model // h
#         self.d_v = d_model // h
#         self.h = h
#
#         # key, query, value projections for all heads
#         self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
#         self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
#         self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection
#
#         self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
#         self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection
#
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#
#         # layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#
#         self.init_weights()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes Self-Attention
#         Args:
#             x (tensor): input (token) dim:(b_s, nx, c),
#                 b_s means batch size
#                 nx means length, for CNN, equals H*W, i.e. the length of feature maps
#                 c means channel, i.e. the channel of feature maps
#             attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#             attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         Return:
#             output (tensor): dim:(b_s, nx, c)
#         '''
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         b_s, nq = rgb_fea_flat.shape[:2]
#         nk = rgb_fea_flat.shape[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
#         lambda_init = lambda_init_fn(1)
#         lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#
#         # Self-Attention
#         rgb_fea_flat = self.LN1(rgb_fea_flat)
#         q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         ir_fea_flat = self.LN2(ir_fea_flat)
#         q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
#         v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#
#         att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
#         att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)
#
#         # get attention matrix
#         att_vis = torch.softmax(att_vis, -1)
#         att_vis = self.attn_drop(att_vis)
#         att_ir = torch.softmax(att_ir, -1)
#         att_ir = self.attn_drop(att_ir)
#
#
#         # output
#         out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
#         out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
#         out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
#                                                                                        self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
#         out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)
#
#         return [out_vis, out_ir]
#
# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)
# class Iterative_Differential_TransformerBlock(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
#         """
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         :param block_exp: Expansion factor for MLP (feed foreword network)
#         """
#         super(Iterative_Differential_TransformerBlock, self).__init__()
#         self.loops = loops_num
#         self.ln_input = nn.LayerNorm(d_model)
#         self.ln_output = nn.LayerNorm(d_model)
#         self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
#         self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                      # nn.SiLU(),  # changed from GELU
#                                      nn.GELU(),  # changed from GELU
#                                      nn.Linear(block_exp * d_model, d_model),
#                                      nn.Dropout(resid_pdrop),
#                                      )
#         self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                     # nn.SiLU(),  # changed from GELU
#                                     nn.GELU(),  # changed from GELU
#                                     nn.Linear(block_exp * d_model, d_model),
#                                     nn.Dropout(resid_pdrop),
#                                     )
#         self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
#                                  # nn.SiLU(),  # changed from GELU
#                                  nn.GELU(),  # changed from GELU
#                                  nn.Linear(block_exp * d_model, d_model),
#                                  nn.Dropout(resid_pdrop),
#                                  )
#
#         # Layer norm
#         self.LN1 = nn.LayerNorm(d_model)
#         self.LN2 = nn.LayerNorm(d_model)
#         # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
#         # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)
#
#         # Learnable Coefficient
#         self.coefficient1 = LearnableCoefficient()
#         self.coefficient2 = LearnableCoefficient()
#         self.coefficient3 = LearnableCoefficient()
#         self.coefficient4 = LearnableCoefficient()
#         self.coefficient5 = LearnableCoefficient()
#         self.coefficient6 = LearnableCoefficient()
#         self.coefficient7 = LearnableCoefficient()
#         self.coefficient8 = LearnableCoefficient()
#         self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
#
#     def forward(self, x):
#         rgb_fea_flat = x[0]
#         ir_fea_flat = x[1]
#         assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
#         temp_rgb = rgb_fea_flat
#         temp_ir = ir_fea_flat
#         dif_fea_rgb = x[0]
#         dif_fea_ir = x[1]
#         lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
#             rgb_fea_flat)
#         lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
#             rgb_fea_flat)
#
#
#         for loop in range(self.loops):
#
#             lambda_init = lambda_init_fn(loop)
#             lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
#             # lambda_init = lambda_init_fn(loop)
#             lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
#             # with Learnable Coefficient
#             rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
#             dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
#             dif_fea_rgb = rgb_fea_out - lambda_ir_full * ir_fea_out
#             rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))
#             ir_fea_flat = self.coefficient5(ir_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))
#
#
#         rgb_fea_final = rgb_fea_out + dif_fea_ir
#         ir_fea_final = ir_fea_out + dif_fea_rgb
#
#         # return [dif_fea_rgb, dif_fea_ir]
#         return [rgb_fea_final,ir_fea_final]
#
#
# class Iterative_Differential_TransformerFusionBlock_FDOMdualfeedback(nn.Module):
#     def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super(Iterative_Differential_TransformerFusionBlock_FDOMdualfeedback, self).__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#         self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         # downsampling
#         # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
#         # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
#
#         self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
#         self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
#
#         # LearnableCoefficient
#         self.vis_coefficient = LearnableWeights()
#         self.ir_coefficient = LearnableWeights()
#
#         # init weights
#         self.apply(self._init_weights)
#
#         # cross transformer
#         self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
#
#         # Concat
#         # self.concat = Concat(dimension=1)
#         self.fused = NiNfusion(d_model)
#
#         # conv1x1
#         self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, rgb_fea, ir_fea):
#         # rgb_fea = x[0]
#         # ir_fea = x[1]
#         assert rgb_fea.shape[0] == ir_fea.shape[0]
#         bs, c, h, w = rgb_fea.shape
#
#         # ------------------------- cross-modal feature fusion -----------------------#
#         #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
#         new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
#         new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
#         rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
#
#         #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
#         new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
#         ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
#
#         rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
#
#         rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
#         new_rgb_fea = rgb_fea_CFE + rgb_fea
#         ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
#         new_ir_fea = ir_fea_CFE + ir_fea
#
#         new_fea = self.fused(new_rgb_fea, new_ir_fea)
#         # new_fea = self.conv1x1_out(new_fea)
#
#         # ------------------------- feature visulization -----------------------#
#         # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
#         # fea_rgb = torch.mean(rgb_fea, dim=1)
#         # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
#         # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
#         # fea_ir = torch.mean(ir_fea, dim=1)
#         # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
#         # fea_ir_new = torch.mean(new_ir_fea, dim=1)
#         # fea_new = torch.mean(new_fea, dim=1)
#         # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
#         # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
#         # plt.figure()
#         # for i in range(len(block)):
#         #     feature = transforms.ToPILImage()(block[i].squeeze())
#         #     ax = plt.subplot(3, 3, i + 1)
#         #     ax.set_xticks([])
#         #     ax.set_yticks([])
#         #     ax.set_title(black_name[i], fontsize=8)
#         #     plt.imshow(feature)
#         # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
#         # -----------------------------------------------------------------------------#
#
#         return new_fea


# ========================mamba fusion================================

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)


# an alternative for mamba_ssm
from selective_scan.selective_scan_interface import selective_scan_fn as selective_scan_fn_v1


# cross selective scan ===============================
if True:
    import selective_scan_cuda_core as selective_scan_cuda


    class SelectiveScan(torch.autograd.Function):
        # @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        # @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)


    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y

        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None


    class CrossScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # B,  d, 2 * H * W
            # ys = ys[:, 0] + ys[:, 1]  # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:H * W].view(B, -1, H, W), ys[:, :, H * W:2 * H * W].view(B, -1, H, W)


    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            return ys[:, :, 0:L // 2], ys[:, :, L // 2:L]

        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            # H, W = ctx.shape
            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2 * L))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2 * L)
            return xs, None, None


    def cross_selective_scan(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)

        y = CrossMerge.apply(ys)
        y=y.half()
        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def selective_scan_1d(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape  # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)

        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def cross_selective_scan_multimodal_k1(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2 * H * W)

        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L // 2] + ys[:, 0, :, L // 2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)

        return y


    def cross_selective_scan_multimodal_k2(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm1: torch.nn.Module = None,
            out_norm2: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e)  # B, C, H, W -> B, 2, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        # @staticmethod
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2 * H * W).to(x_rgb.dtype)

        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)

        return y_rgb, y_e


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try:
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    assert inputs[1].debugName().startswith("dts")  # (B, D, L)
    assert inputs[2].debugName().startswith("As")  # (D, N)
    assert inputs[3].debugName().startswith("Bs")  # (D, N)
    assert inputs[4].debugName().startswith("Cs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


DEV = False


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 4 if not (self.forward_core == self.forward_corev1_share_ssm) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K = 1
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.K2 = self.K if not (self.forward_core == self.forward_corev1_share_a) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K2 = 1
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 1  # 4

        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,  # z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            # return_last_state=False,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = torch.transpose(out_y[:, 0], dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
        selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev1(self, x: torch.Tensor, float32=True):
        # float32 should be true in training!!!! otherwise, the output of selective_scan would be inf...
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        As = -torch.exp(self.A_logs.to(torch.float))  # (k * d, d_state)
        Ds = self.Ds.to(torch.float)  # (k * d)
        dt_projs_bias = self.dt_projs_bias.to(torch.float).view(-1)  # (k * d)

        if float32:
            ys: torch.Tensor = selective_scan(
                xs.to(torch.float),
                dts.to(torch.float),
                As,
                Bs.to(torch.float),
                Cs.to(torch.float),
                Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        else:
            out_y: torch.Tensor = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

            # if torch.isinf(y).any() or torch.isnan(y).any():
            #     for item in [y, xs, dts, As, Bs, Cs, Ds]:
            #         print(torch.isinf(item).any(), torch.isnan(item).any(), item.max(), item.min())
            #     import time; time.sleep(10000)

        return y

    def forward_corev1_share_ssm(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
            return xs

        x_dbl = torch.einsum("b d l, c d -> b c l", x.view(B, -1, L), self.x_proj_weight[0])
        # x_dbl = x_dbl + self.x_proj_bias.view(1, -1, 1)
        dt, BC = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state], dim=1)
        dt = torch.einsum("b r l, d r -> b d l", dt, self.dt_projs_weight[0])
        x_dt_BC = torch.cat([x, dt.view(B, -1, H, W), BC.view(B, -1, H, W)], dim=1)  # (b, -1, h, w)

        x_dt_BCs = cross_scan_2d(x_dt_BC)  # (b, k, d, l)
        xs, dts, Bs, Cs = torch.split(x_dt_BCs, [self.d_inner, self.d_inner, self.d_state, self.d_state], dim=2)

        xs = xs.contiguous().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        As = -torch.exp(self.A_logs.float()).repeat(4, 1)  # (k * d, d_state)
        Ds = self.Ds.repeat(4)  # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1).repeat(4)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 4, -1, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1_share_a(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x, dim=1):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=dim)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=dim)  # (b, k, d, l)
            return xs

        K = 4
        xs = cross_scan_2d(x, dim=1)  # (b, d, k, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        dts = dts + self.dt_projs_bias.to(xs.dtype).view(1, K, -1, 1)

        xs = xs.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        dts = dts.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        As = -torch.exp(self.A_logs.float())  # (D, N)
        Ds = self.Ds.view(-1)  # (D)
        Bs = Bs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)
        Cs = Cs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=None,
            delta_softplus=True,
        ).view(B, -1, 4, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, :, 2:4], dims=[-1]).view(B, -1, 2, L)
        wh_y = torch.transpose(out_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, :, 0].float() + inv_y[:, :, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev2(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
            nrows=nrows,
        )

    def forward_core_1d(self, x: torch.Tensor):
        return selective_scan_1d(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
        )

    # forward_core = forward_core_share_ssm
    # forward_core = forward_core_share_a
    # forward_core = forward_corev1
    forward_core = forward_corev2  # vmamba

    # forward_core = forward_corev0 # ori mamba
    # forward_core = forward_core_1d # ori mamba

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))  # (b, d, h, w)
            y = self.forward_core(x)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = F.silu(x)
            else:
                xz = F.silu(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out



# =====================================================
class SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y



# ====================================================
class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CM_Attention(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_modalx = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_k2(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None),
            self.softmax_version,
            nrows=nrows,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb= x_rgb.permute(0,2,3,1).contiguous()
        x_e =x_e.permute(0,2,3,1).contiguous()
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans))  # (b, d, h, w)
            x_e_conv = self.act(self.conv2d_modalx(x_e_trans))  # (b, d, h, w)
            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv)  # b, d, h, w -> b, h, w, d
            # SE to get attention, scale
            b, d, h, w = x_rgb_trans.shape
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()  # b, 1, 1, d
            x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
            y_rgb = y_rgb * x_e_exitation
            y_e = y_e * x_rgb_exitation
            y = torch.concat([y_rgb, y_e], dim=-1)
        out = self.dropout(self.out_proj(y)).permute(0,3,1,2).contiguous()

        return out



class CMSSF(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CM_Attention(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, x: torch.Tensor):
        x_rgb=x[0]
        x_e = x[1]
        x = x_rgb + x_e + self.drop_path(self.op(x_rgb, x_e))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, x: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)







class DBISSF_Attention(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            Cross=True,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.Cross = Cross
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_share = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        self.dt_proj_share = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.A_log_share = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)
        self.D_share = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)
        self.out_norm_share = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # def visualization(feat_rgb_A,feat_rgb_B,feat_rgb_C,feat_rgb_D,feat_e_A,feat_e_B,feat_e_C,feat_e_D,feat_fusion,h,w):
    #
    #     save_dir = 'visualization'
    #     feat_rgb_A = torch.mean(feat_rgb_A, dim=1)
    #     feat_rgb_B = torch.mean(feat_rgb_B, dim=1)
    #     feat_rgb_C = torch.mean(feat_rgb_C, dim=1)
    #     feat_rgb_D = torch.mean(feat_rgb_D, dim=1)
    #     feat_e_A = torch.mean(feat_e_A, dim=1)
    #     feat_e_B = torch.mean(feat_e_B, dim=1)
    #     feat_e_C = torch.mean(feat_e_C, dim=1)
    #     feat_e_D = torch.mean(feat_e_D, dim=1)
    #     block = [feat_rgb_A,feat_rgb_B,feat_rgb_C,feat_rgb_D,feat_e_A,feat_e_B,feat_e_C,feat_e_D]
    #     black_name = ["feat_rgb_A","feat_rgb_B","feat_rgb_C","feat_rgb_D","feat_e_A","feat_e_B","feat_e_C","feat_e_D"]
    #     plt.figure()
    #     for i in range(len(block)):
    #         feature = transforms.ToPILImage()(block[i].squeeze())
    #         ax = plt.subplot(3, 3, i + 1)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.set_title(black_name[i], fontsize=8)
    #         plt.imshow(feature)
    #     plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)

    def forward(self, x_rgb: torch.Tensor,x_share_rgb, x_e: torch.Tensor,x_share_e):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        x_share_rgb = x_share_rgb.permute(0, 2, 1)
        x_share_e = x_share_e.permute(0, 2, 1)

        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)
        x_dbl_share = self.x_proj_share(rearrange(torch.add(x_share_rgb,x_share_e), "b d l -> (b l) d"))  # (bl d)

        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_share, B_share, C_share = torch.split(x_dbl_share, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_share = self.dt_proj_share.weight @ dt_share.t()

        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        dt_share = rearrange(dt_share, "d (b l) -> b d l", l=L)

        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        A_share = -torch.exp(self.A_log_share.float())  # (k * d, d_state)

        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_share = rearrange(B_share, "(b l) dstate -> b dstate l", l=L).contiguous()

        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_share = rearrange(C_share, "(b l) dstate -> b dstate l", l=L).contiguous()

        if self.Cross:
            y_rgb = selective_scan(
                x_rgb, dt_rgb,
                A_rgb, B_rgb, C_e, self.D_1.float(),
                delta_bias=self.dt_proj_1.bias.float(),
                delta_softplus=True,
            )
            y_e = selective_scan(
                x_e, dt_e,
                A_e, B_e, C_rgb, self.D_2.float(),
                delta_bias=self.dt_proj_2.bias.float(),
                delta_softplus=True,
            )
            y_share_rgb = selective_scan(
                x_rgb, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus = True,
            )
            y_share_e = selective_scan(
                x_e, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
        else:
            y_rgb = selective_scan(
                x_rgb, dt_rgb,
                A_rgb, B_rgb, C_rgb, self.D_1.float(),
                delta_bias=self.dt_proj_1.bias.float(),
                delta_softplus=True,
            )
            y_e = selective_scan(
                x_e, dt_e,
                A_e, B_e, C_e, self.D_2.float(),
                delta_bias=self.dt_proj_2.bias.float(),
                delta_softplus=True,
            )
            y_share_rgb = selective_scan(
                x_rgb, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
            y_share_e = selective_scan(
                x_e, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
        # assert out_y.dtype == torch.float
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        y_share_rgb = rearrange(y_share_rgb, "b d l -> b l d")
        y_share_rgb = self.out_norm_share(y_share_rgb)
        y_share_e = rearrange(y_share_e, "b d l -> b l d")
        y_share_e = self.out_norm_share(y_share_e)
        return y_rgb,y_share_rgb, y_e,y_share_e
class DBISSF_SS(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            Cross=True,
            learnableweight=False,
            scan = False,
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.Cross = Cross
        self.learnableweight=learnableweight
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_share = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        if learnableweight:
            self.learnableweight_rgb = LearnableWeights()
            self.learnableweight_e = LearnableWeights()
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_share = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_share = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.CMA_ssm = DBISSF_Attention(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            Cross=Cross,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,

            **kwargs,
        )
        self.scan = scan
        if scan:
            self.CMA_ssm2 = DBISSF_Attention(
                d_model=self.d_model,
                d_state=self.d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=dt_rank,
                Cross=Cross,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,

                **kwargs,
            )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = x_rgb.permute(0, 2,3,1).contiguous()
        x_e = x_e.permute(0, 2,3,1).contiguous()
        temp_rgb = x_rgb
        temp_e =x_e
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        x_share_rgb =self.in_proj_share(temp_rgb)
        x_share_e = self.in_proj_share(temp_e)

        B, H, W, D = x_rgb.shape
        if self.d_conv > 1:
            x_rgb = x_rgb.permute(0, 3,1,2).contiguous()
            x_e = x_e.permute(0, 3,1,2).contiguous()
            x_share_rgb = x_share_rgb.permute(0, 3, 1, 2).contiguous()
            x_share_e = x_share_e.permute(0, 3, 1, 2).contiguous()

            x_rgb_conv = self.act(self.conv2d(x_rgb))  # (b, d, h, w)
            x_e_conv = self.act(self.conv2d(x_e))  # (b, d, h, w)
            x_share_rgb_conv = self.act(self.conv2d(x_share_rgb))  # (b, d, h, w)
            x_share_e_conv = self.act(self.conv2d(x_share_e))  # (b, d, h, w)

            x_rgb_conv_new = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
            x_e_conv_new = rearrange(x_e_conv, "b d h w -> b (h w) d")
            x_share_rgb_conv_new = rearrange(x_share_rgb_conv, "b d h w -> b (h w) d")
            x_share_e_conv_new = rearrange(x_share_e_conv, "b d h w -> b (h w) d")

            y_rgb, y_share_rgb, y_e, y_share_e = self.CMA_ssm(x_rgb_conv_new,x_share_rgb_conv_new, x_e_conv_new, x_share_e_conv_new)
            # to b, d, h, w
            y_rgb = y_rgb.view(B, H, W, -1)
            y_e = y_e.view(B, H, W, -1)
            y_share_rgb = y_share_rgb.view(B, H, W, -1)
            y_share_e = y_share_e.view(B, H, W, -1)
            if self.scan:
                x_rgb_conv2=x_rgb_conv
                x_e_conv2=x_e_conv
                x_share_rgb_conv2=x_share_rgb
                x_share_e_conv2=x_share_e
                x_rgb_conv_new2 = rearrange(x_rgb_conv2, "b d h w -> b (w h) d")
                x_e_conv_new2 = rearrange(x_e_conv2, "b d h w -> b (w h) d")
                x_share_rgb_conv_new2 = rearrange(x_share_rgb_conv2, "b d h w -> b (w h) d")
                x_share_e_conv_new2 = rearrange(x_share_e_conv2, "b d h w -> b (w h) d")
                y_rgb2, y_share_rgb2, y_e2, y_share_e2 = self.CMA_ssm2(x_rgb_conv_new2, x_share_rgb_conv_new2, x_e_conv_new2,
                                                                  x_share_e_conv_new2)
                y_rgb2 = y_rgb2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_e2 = y_e2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_share_rgb2 = y_share_rgb2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_share_e2 = y_share_e2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_rgb=0.5*y_rgb+0.5*y_rgb2
                y_e = 0.5*y_e+0.5*y_e2
                y_share_rgb=0.5*y_share_rgb+0.5*y_share_rgb2
                y_share_e=0.5*y_share_e+0.5*y_share_e2
        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb)).permute(0, 3,1,2).contiguous()
        out_e = self.dropout_e(self.out_proj_e(y_e)).permute(0, 3,1,2).contiguous()
        out_share_rgb = self.dropout_share(self.out_proj_share(y_share_rgb)).permute(0, 3, 1, 2).contiguous()
        out_share_e = self.dropout_share(self.out_proj_share(y_share_e)).permute(0, 3, 1, 2).contiguous()

        return out_rgb, out_share_rgb, out_e, out_share_e


class DBISSF(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            return_separately:bool = True,
            Cross:bool = True,
            scan:bool = False,
            learnableweight:bool=False,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.return_serparately = return_separately
        self.op = DBISSF_SS(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            Cross = Cross,
            learnableweight=learnableweight,
            scan=scan,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.drop_path_share = DropPath(drop_path)
        # LearnableCoefficient

        self.return_serparately=return_separately
        if not return_separately:
            self.learnable_weight = LearnableWeights()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, x: torch.Tensor):
        x_rgb=x[0]
        x_e =x[1]
        x_rgb_cross,x_share_rgb, x_e_cross,x_share_e = self.op(x_rgb, x_e)

        b,d,h,w= x_e.shape
        x_rgb_new = x_rgb + self.drop_path1(x_rgb_cross)
        x_e_new = x_e + self.drop_path2(x_e_cross)
        x_share_rgb = x_share_rgb +self.drop_path_share(x_share_rgb)
        x_share_e = x_share_e + self.drop_path_share(x_share_e)
        if self.return_serparately:
            return x_rgb_new,x_share_rgb, x_e_new, x_share_e
        else:
            x_cross_feat = self.learnable_weight(x_rgb_new,x_e_new)
        #=================visualization==================
        # save_dir = 'visualization'
        # fea_rgb = torch.mean(x_rgb, dim=1)
        # fea_rgb_CFE = torch.mean(x_rgb_cross, dim=1)
        # fea_rgb_new = torch.mean(x_rgb_new, dim=1)
        # fea_ir = torch.mean(x_e, dim=1)
        # fea_ir_CFE = torch.mean(x_e_cross, dim=1)
        # fea_ir_new = torch.mean(x_e_new, dim=1)
        # fea_new = torch.mean(x_cross_feat, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After cross', 'fea_rgb new', 'fea_ir', 'fea_ir After cross', 'fea_ir new',
        #               'fea_ir mambafusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        #     plt.show()
        # plt.savefig(save_dir + '/fea_{}x{}.png'.format(h, w), dpi=300)
            return x_cross_feat

    def forward(self, x:torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''

        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
class SSF(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            Cross:bool = True,
            scan: bool = False,
            use_learnableweights: bool = False,

            separate_return: bool = True,
            only_cm:bool = False,
            only_dbi:bool=False,

            # =================================================
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.only_cm = only_cm
        self.only_dbi = only_dbi
        if only_cm:
            self.cm_fusion = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
            )
        if only_dbi:
            self.dbi_fusion=DBISSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
                return_separately=separate_return,
                learnableweight=use_learnableweights,
                scan=scan,
                Cross=Cross
            )
        if not only_dbi and not only_cm:
            self.cm_fusion = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.cm_fusion_rgb = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.cm_fusion_e = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.dbi_fusion = DBISSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
                return_separately=separate_return,
                learnableweight=use_learnableweights,
                scan=scan,
                Cross=Cross
            )

    def forward(self,rgb,ir):
        # VSSBlock=======open=====
        # temp =[]
        # temp.append(x[0].permute(0,3,1,2).contiguous())
        # temp.append(x[1].permute(0, 3,1,2).contiguous())
        # x=temp
        # ==========
        x=[rgb,ir]
        if self.only_dbi:
            x_feat=self.dbi_fusion(x)
        elif self.only_cm:
            x_feat = self.cm_fusion(x)
        elif not self.only_cm and not self.only_dbi:
            x_cross_rgb, x_share_rgb, x_cross_e, x_share_e = self.dbi_fusion(x)
            x_fuse_rgb = []
            x_fuse_e = []

            x_fuse_rgb.append(x_cross_rgb)
            x_fuse_rgb.append(x_share_rgb)

            x_fuse_e.append(x_cross_e)
            x_fuse_e.append(x_share_e)

            x_rgb = self.cm_fusion_rgb(x_fuse_rgb)
            x_e = self.cm_fusion_e(x_fuse_e)

            x_new=[]
            x_new.append(x_rgb)
            x_new.append(x_e)
            x_feat = self.cm_fusion(x_new)
        return x_feat

