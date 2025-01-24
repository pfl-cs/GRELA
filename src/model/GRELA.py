import torch
import sys

sys.path.append("../../../")
import torch.nn as nn
from . import encoder_module, graph_module


class GRELA(nn.Module):
    def __init__(self, cfg):
        super(GRELA, self).__init__()
        # assume distributed training is not yet supported
        # TODO: refine the code to support distributed training when release
        self.encoder_num_wt_layers = cfg.encoder.num_weight_tied_layers
        self.graph_num_wt_layers = cfg.graph.num_weight_tied_layers
        self.model_name = 'GRELA'

        self.encoder = encoder_module.Encoder(
            num_attrs=cfg.dataset.num_attrs,
            n_bins=cfg.dataset.n_bins,
            query_part_feature_dim=cfg.dataset.query_part_feature_dim,
            pooling_mlp_hidden_dim=cfg.encoder.pooling_mlp_hidden_dim,
            num_s_self_attn_layers=cfg.encoder.num_s_self_attn_layers,
            num_q_cross_attn_layers=cfg.encoder.num_q_cross_attn_layers,
            num_attn_heads=cfg.encoder.num_attn_heads,
            attn_head_key_dim=cfg.encoder.attn_head_key_dim,
            query_emb_dim=cfg.model.query_emb_dim,
            attn_mlp_hidden_dim=cfg.encoder.attn_mlp_hidden_dim,
            ff_mlp_num_layers=cfg.encoder.ff_mlp_num_layers,
            ff_mlp_hidden_dim=cfg.encoder.ff_mlp_hidden_dim,
            use_s_ff=cfg.encoder.use_s_ff,
            use_q_ff=cfg.encoder.use_q_ff,
            dropout_rate=cfg.model.dropout_rate,
            use_float64=cfg.model.use_float64,
            num_weight_tied_layers=cfg.encoder.num_weight_tied_layers
        )

        self.mse_loss = nn.MSELoss(reduction='mean')

        # for the task heads
        # which would be mapped into the task embedding later on
        self.task_heads = torch.Tensor(cfg.dataset.num_task, cfg.model.query_emb_dim)
        self.task_heads = nn.Parameter(self.task_heads)
        self.task_heads.data = nn.init.xavier_uniform_(self.task_heads.data, gain=nn.init.calculate_gain('relu'))

        self.graph = graph_module.Graph(
            num_attn_layers=cfg.graph.num_attn_layers,
            num_attn_heads=cfg.graph.num_attn_heads,
            query_emb_dim=cfg.model.query_emb_dim,
            attn_mlp_hidden_dim=cfg.graph.attn_mlp_hidden_dim,
            use_ff=cfg.graph.use_ff,
            ff_mlp_num_layers=cfg.graph.ff_mlp_num_layers,
            ff_mlp_hidden_dim=cfg.graph.ff_mlp_hidden_dim,
            dropout_rate=cfg.model.dropout_rate,
            use_float64=cfg.model.use_float64,
            fix_keys_in_attn=cfg.graph.fix_keys_in_attn,
            num_weight_tied_layers=cfg.graph.num_weight_tied_layers
        )


    def print_params(self):
        params = self.parameters()
        print('-' * 50)
        for p in params:
            # if p.requires_grad:
            print(p.name)
        print('-' * 50)


    def forward_embs(self, S, Q):
        return self.encoder(S, Q)

    # return the output
    # task_preds.shape [batch_size, num_task]
    def forward(self, S, Q, task_self_attn_mask, data_task_attn_mask):
        # get the query embedding from the Encoder module
        query_embs = self.encoder(S, Q)
        query_embs, task_embs, task_preds = self.graph(query_embs, self.task_heads, task_self_attn_mask, data_task_attn_mask)
        return task_preds
