import torch
import torch.nn as nn
import numpy as np
import math
from diffusion_policy.model.vision.model_getter import get_linear

class ScorerNetworkMLP(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.device = self.cfg['device']

        # action mlp config
        action_mlp_cfg = self.cfg["action_mlp"]
        action_units = action_mlp_cfg["units"]
        action_activation_function = action_mlp_cfg.get("activation_function", None)
        action_norm_func_name = action_mlp_cfg.get("norm_func_name", None)
        action_lastnorm_func_name = action_mlp_cfg.get("lastnorm_func_name", None)

        # label mlp config
        label_mlp_cfg = self.cfg["label_mlp"]
        label_units = label_mlp_cfg["units"]
        label_activation_function = label_mlp_cfg.get("activation_function", None)
        label_norm_func_name = label_mlp_cfg.get("norm_func_name", None)
        label_lastnorm_func_name = label_mlp_cfg.get("lastnorm_func_name", None)

        # model
        self.action_model = get_linear(action_units, action_activation_function, action_norm_func_name, action_lastnorm_func_name).to(self.device)
        self.label_model = get_linear(label_units, label_activation_function, label_norm_func_name, label_lastnorm_func_name).to(self.device)
        self.loss_function = nn.MSELoss().to(self.device)    

    def forward(self, res_dict):
        bsz = res_dict['action'].shape[0]
        action_feature = self.action_model(res_dict['action']).reshape(bsz, -1)
        inpu_feature = torch.cat([res_dict['image'], action_feature], dim=-1)
        output = self.label_model(inpu_feature)
        return output

class ScorerNetworkTransformer(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.device = self.cfg['device']
        self.score_step = self.cfg['score_step']

        # Transformer config
        transformer_cfg = self.cfg["transformer"]
        self.is_cross = transformer_cfg["is_cross"]

        # action mlp config
        action_mlp_cfg = self.cfg["action_mlp"]
        units = action_mlp_cfg["units"]
        assert units[0] == transformer_cfg["embedding_dim"]
        activation_function = action_mlp_cfg.get("activation_function", None)
        norm_func_name = action_mlp_cfg.get("norm_func_name", None)
        lastnorm_func_name = action_mlp_cfg.get("lastnorm_func_name", None)

        # model
        if self.is_cross:
            self.transform_model = CrossAttention(self.device, transformer_cfg).to(self.device)
        else:
            self.transform_model = SelfAttention(self.device, transformer_cfg).to(self.device)
        self.action_model = get_linear(units, activation_function, norm_func_name, lastnorm_func_name).to(self.device)
        self.loss_function = nn.MSELoss().to(self.device)

    def forward(self, res_dict):
        feature = self.transform_model(res_dict)
        output = self.action_model(feature)
        return output

    def forward_cls(self,res_dict):
        feature = self.transform_model.forward_cls(res_dict)
        output = self.action_model(feature)
        return output

    def finetune_forward(self, res_dict):
        with torch.no_grad():
            feature = self.transform_model(res_dict)
        output = self.action_model(feature)
        return output

class SelfAttention(nn.Module):
    def __init__(self, device, cfg):
        super(SelfAttention, self).__init__()
        time_step = cfg["time_step"]
        embedding_dim = cfg["embedding_dim"]
        pos_embedding_dim = cfg["pos_embedding_dim"]
        n_head = cfg["n_head"]
        depth = cfg["depth"]
        ffd = cfg["ffd"]
        self.use_encoder = cfg["use_encoder_for_action"]
        use_pe = cfg.get("use_pe", False)
        self.use_pe = use_pe
        if self.use_pe:
            self.position_encoder = PositionalEncoding(device, pos_embedding_dim, time_step+1)

        if self.use_encoder:
            self.encoder_ffd_for_action = get_linear(cfg["encoder_mlp"]).to(device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_head,
            batch_first=True,
            dim_feedforward=ffd
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )
        self.cls_token = torch.nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )  # "global information"
        torch.nn.init.normal_(self.cls_token, std=0.02)

    def forward_cls(self, x):
        # cls_token = torch.randn((x.shape[0], 1, x.shape[-1]), device=x.device)
        # x = torch.column_stack((cls_token, x))  # tokens is of shape [B, 1+T, F]
        x = x['qkv']
        if self.use_encoder:
            x = self.encoder_ffd(x)
        x = torch.column_stack((self.cls_token.repeat(x.shape[0], 1, 1), x))  # tokens is of shape [B, 1+T, F]
        if self.use_pe:
            x = self.position_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :] # batch_size * steps * dim
        return x

    def forward(self, x):
        # cls_token = torch.randn((x.shape[0], 1, x.shape[-1]), device=x.device)
        # x = torch.column_stack((cls_token, x))  # tokens is of shape [B, 1+T, F]
        x = x['qkv']
        if self.use_encoder:
            action_feature = self.encoder_ffd_for_action(x[:,:,-8:])
            all_feature = torch.cat([x[:,:,:-8], action_feature], dim=-1)
        if self.use_pe:
            x = self.position_encoder(all_feature)
        x = self.transformer_encoder(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, device, cfg):
        super(CrossAttention, self).__init__()
        self.use_pe = cfg.get("use_pe", False)
        time_step = cfg["time_step"]
        embedding_dim = cfg["embedding_dim"]
        pos_embedding_dim = cfg["pos_embedding_dim"]
        ffd = cfg["ffd"] * embedding_dim
        depth = cfg["depth"]
        n_head = cfg["n_head"]
        dropout = cfg.get("dropout", 0.1)
        Q_encoder_cfg = cfg["Q_encoder"]
        K_encoder_cfg = cfg["K_encoder"]
        V_encoder_cfg = cfg["V_encoder"]

        if self.use_pe:
            # for k and v
            self.position_encoder = PositionalEncoding(device, pos_embedding_dim, time_step)
        self.Q_encoder = get_linear(Q_encoder_cfg["units"], activation_function = Q_encoder_cfg["activation_function"])
        self.K_encoder = get_linear(K_encoder_cfg["units"], activation_function = K_encoder_cfg["activation_function"])
        self.V_encoder = get_linear(V_encoder_cfg["units"], activation_function = V_encoder_cfg["activation_function"])
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                embedding_dim=embedding_dim,
                n_head=n_head,
                ffd=ffd,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embedding_dim))
        torch.nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, res_dict):
        q = res_dict['q']
        kv = res_dict['kv']
        if self.use_pe:
            kv = self.position_encoder(kv)
        query = self.Q_encoder(q)
        key = self.K_encoder(kv)
        value = self.V_encoder(kv)
        # feature, feature_weight = self.cross_attention(query, key, value)
        for layer in self.cross_attention_layers:
            feature = layer(query, key, value)  
            query = feature      
        return feature

    def forward_cls(self, res_dict):
        q = res_dict['q']      # (B, Lq, ?)
        kv = res_dict['kv']    # (B, Lk, ?)

        if self.use_pe:
            kv = self.position_encoder(kv)

        query = self.Q_encoder(q)   # (B, Lq, D)
        key   = self.K_encoder(kv)  # (B, Lk, D)
        value = self.V_encoder(kv)  # (B, Lk, D)

        # ---- prepend cls token to query sequence ----
        B = query.shape[0]
        query = torch.column_stack((self.cls_token.repeat(B, 1, 1), query))  # (B, 1+Lq, D)

        for layer in self.cross_attention_layers:
            feature = layer(query, key, value)  # (B, 1+Lq, D)
            query = feature

        # ---- return global token ----
        feature = feature[:, 0, :]  # (B, D)
        return feature

class CrossAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, n_head, ffd, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_head,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffd),
            nn.ReLU(),
            nn.Linear(ffd, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # q: (B, Lq, D), kv: (B, Lk, D)
        attn_out, _ = self.attn(q, k, v)        
        x = self.norm1(q + self.dropout(attn_out))  

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))   
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, device, d_model: int, max_len: int = 5000):
        nn.Module.__init__(self)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe[None].to(device)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return x