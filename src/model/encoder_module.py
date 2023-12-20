import torch
import sys
import torch.nn as nn
from . import attn

class Encoder(nn.Module):
    def __init__(self,
                 num_attrs,
                 n_bins,
                 query_part_feature_dim,
                 pooling_mlp_hidden_dim,
                 num_s_self_attn_layers,
                 num_q_cross_attn_layers,
                 num_attn_heads,
                 attn_head_key_dim,
                 query_emb_dim,
                 attn_mlp_hidden_dim,
                 use_s_ff,
                 use_q_ff,
                 ff_mlp_num_layers,
                 ff_mlp_hidden_dim,
                 dropout_rate,
                 use_float64,
                 num_weight_tied_layers=0):
        super(Encoder, self).__init__()

        self.num_attrs = num_attrs
        self.n_bins = n_bins
        self.query_part_feature_dim = query_part_feature_dim
        self.pooling_mlp_hidden_dim = pooling_mlp_hidden_dim

        self.num_s_self_attn_layers = num_s_self_attn_layers
        self.num_q_cross_attn_layers = num_q_cross_attn_layers
        self.num_attn_heads = num_attn_heads

        if num_weight_tied_layers == 0:
            self.num_s_self_attn_layers += 1
            self.num_q_cross_attn_layers += 1
        self.num_weight_tied_layers = num_weight_tied_layers
        self.attn_head_key_dim = attn_head_key_dim
        self.query_emb_dim = query_emb_dim
        self.attn_mlp_hidden_dim = attn_mlp_hidden_dim

        self.use_s_ff = use_s_ff
        self.use_q_ff = use_q_ff
        self.ff_mlp_num_layers = ff_mlp_num_layers
        self.ff_mlp_hidden_dim = ff_mlp_hidden_dim

        self.dropout_rate = dropout_rate
        self.use_float64 = use_float64

        # S-Transformer
        self.S_pooling = attn._MLP(2, self.n_bins, self.pooling_mlp_hidden_dim, self.attn_head_key_dim,
                                   self.dropout_rate, self.use_float64)

        self.S_self_attn_layers = torch.nn.ModuleList([
            attn.attnBlock(
                self.attn_head_key_dim,
                self.num_attn_heads,
                self.attn_mlp_hidden_dim,
                self.dropout_rate,
                self.use_float64,
                if_self_attn=True
                # # attn_head_key_dim=cfg.encoder.attn_head_key_dim,
                # # num_attn_heads=cfg.encoder.num_attn_heads,
            )
            for _ in range(self.num_s_self_attn_layers)])

        if self.use_s_ff:
            self.S_feed_forward = attn._MLP(n_layers=self.ff_mlp_num_layers,
                                            dim_input=self.attn_head_key_dim,
                                            dim_inner=self.ff_mlp_hidden_dim,
                                            dim_output=self.attn_head_key_dim,
                                            dropout_rate=self.dropout_rate,
                                            use_float64=self.use_float64)


        # Q-Transformer
        self.Q_pooling = attn._MLP(2, self.query_part_feature_dim, self.pooling_mlp_hidden_dim, self.attn_head_key_dim,
                                   self.dropout_rate, self.use_float64)

        self.Q_cross_attn_layers = torch.nn.ModuleList([
            attn.attnBlock(
                self.attn_head_key_dim,
                self.num_attn_heads,
                self.attn_mlp_hidden_dim,
                self.dropout_rate,
                self.use_float64,
                if_self_attn=False
                # attn_head_key_dim=cfg.encoder.attn_head_key_dim,
                # num_attn_heads=cfg.encoder.num_attn_heads,
            )
            for _ in range(self.num_q_cross_attn_layers)])

        if self.num_weight_tied_layers > 0:
            self.S_self_weight_tied_attn = attn.attnBlock(
                self.attn_head_key_dim,
                self.num_attn_heads,
                self.attn_mlp_hidden_dim,
                self.dropout_rate,
                self.use_float64,
                if_self_attn=True
            )
            self.Q_cross_weight_tied_attn = attn.attnBlock(
                self.attn_head_key_dim,
                self.num_attn_heads,
                self.attn_mlp_hidden_dim,
                self.dropout_rate,
                self.use_float64,
                if_self_attn=False
            )

        if self.use_q_ff:
            self.Q_feed_forward = attn._MLP(n_layers=self.ff_mlp_num_layers,
                                            dim_input=self.attn_head_key_dim,
                                            dim_inner=self.ff_mlp_hidden_dim,
                                            dim_output=self.query_emb_dim,
                                            dropout_rate=self.dropout_rate,
                                            use_float64=self.use_float64)

        if self.use_float64:
            self.double()


    def forward(self, s, q):
        """
        :param s: Shape `(batch_size, num_attrs, n_bins)
        :param q: Shape `(batch_size, 1, query_part_features_dim)
        :return:
        """
        s = torch.reshape(s, [-1, self.num_attrs, self.n_bins])
        q = torch.reshape(q, [-1, 1, self.query_part_feature_dim])

        s = self.S_pooling(s)
        q = self.Q_pooling(q)

        # S-Transformer
        for i, layer in enumerate(self.S_self_attn_layers):
            s = layer(s, s)

        if self.num_weight_tied_layers > 0:
            for i in range(self.num_weight_tied_layers):
                s = self.S_self_weight_tied_attn(s, s)

        if self.use_s_ff:
            s = self.S_feed_forward(s)

        # Q-Transformer
        for layer in self.Q_cross_attn_layers:
            q = layer(s, q)

        if self.num_weight_tied_layers > 0:
            for i in range(self.num_weight_tied_layers):
                q = self.Q_cross_weight_tied_attn(s, q)

        if self.use_q_ff:
            q = self.Q_feed_forward(q)

        x = torch.reshape(q, [-1, self.query_emb_dim])

        return x

