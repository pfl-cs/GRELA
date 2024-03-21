import torch
import torch.nn as nn
from . import attn

class Graph(nn.Module):
    def __init__(self,
                 num_attn_layers,
                 num_attn_heads,
                 query_emb_dim,
                 attn_mlp_hidden_dim,
                 use_ff,
                 ff_mlp_num_layers,
                 ff_mlp_hidden_dim,
                 dropout_rate,
                 use_float64,
                 fix_keys_in_attn,
                 num_weight_tied_layers):
        super(Graph, self).__init__()

        self.num_attn_layers = num_attn_layers
        self.num_attn_heads = num_attn_heads
        self.num_weight_tied_layers = num_weight_tied_layers
        if num_weight_tied_layers == 0:
            self.num_task_attn_layers += 1

        self.query_emb_dim = query_emb_dim
        self.attn_mlp_hidden_dim = attn_mlp_hidden_dim

        self.use_ff = use_ff
        self.ff_mlp_num_layers = ff_mlp_num_layers
        self.ff_mlp_hidden_dim = ff_mlp_hidden_dim

        self.dropout_rate = dropout_rate
        self.use_float64 = use_float64
        self.fix_keys_in_attn = fix_keys_in_attn

        self.attn_layers = torch.nn.ModuleList([
            attn.attnBlock(
                self.query_emb_dim,
                self.num_attn_heads,
                self.attn_mlp_hidden_dim,
                self.dropout_rate,
                self.use_float64,
                if_self_attn=False
            )
            for _ in range(self.num_attn_layers)])

        if num_weight_tied_layers > 0:
            self.weight_tied_attn = attn.attnBlock(
                self.query_emb_dim,
                self.num_attn_heads,
                self.attn_mlp_hidden_dim,
                self.dropout_rate,
                self.use_float64,
                if_self_attn=False
            )

        if self.use_ff:
            self.feed_forward = attn._MLP(self.ff_mlp_num_layers,
                                          dim_input=self.query_emb_dim,
                                          dim_inner=self.ff_mlp_hidden_dim,
                                          dim_output=self.query_emb_dim,
                                          dropout_rate=self.dropout_rate,
                                          use_float64=self.use_float64)


    def forward(self, _data_embs, _task_embs, task_self_attn_mask, data_task_attn_mask):
        task_attn_mask = torch.vstack((task_self_attn_mask, data_task_attn_mask)).T
        task_attn_mask = ~task_attn_mask
        attn_keys = torch.vstack((_task_embs, _data_embs))
        attn_keys = attn_keys[None, :, :]
        task_embs = _task_embs[None, :, :]
        # data_embs_3d = _data_embs[None, :, :]
        for i, layer in enumerate(self.attn_layers):
            # attn_keys = torch.cat((task_embs, data_embs_3d), dim=1)
            task_embs = layer(attn_keys, task_embs, attn_mask=task_attn_mask)

        for i in range(self.num_weight_tied_layers):
            if not self.fix_keys_in_attn:
                attn_keys = torch.vstack((torch.squeeze(task_embs), _data_embs))
                attn_keys = attn_keys[None, :, :]
            task_embs = self.weight_tied_attn(attn_keys, task_embs, attn_mask=task_attn_mask)

        task_embs = torch.squeeze(task_embs)
        if self.use_ff:
            task_embs = self.feed_forward(task_embs)

        data_embs = _data_embs
        task_preds = torch.matmul(data_embs, task_embs.transpose(0, 1))

        return data_embs, task_embs, task_preds
