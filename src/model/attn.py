import torch
import torch.nn as nn

class _MLP(nn.Module):
    def __init__(self,
                 n_layers,
                 dim_input,
                 dim_inner,
                 dim_output,
                 dropout_rate,
                 act='ELU',
                 use_float64=False):
        super(_MLP, self).__init__()

        for i in range(n_layers - 1):
            d_in = dim_input if i == 0 else dim_inner
            # d_out = dim_output if i == (n_layers - 1) else dim_inner
            linear_layer = torch.nn.Linear(in_features=d_in, out_features=dim_inner, bias=True)
            self.add_module(f'Linear_{i}', linear_layer)
            act_layer = torch.nn.ELU() if act == 'ELU' else torch.nn.ReLU()
            self.add_module(f'Act_{i}', act_layer)

        final_linear_layer = torch.nn.Linear(in_features=dim_inner, out_features=dim_output, bias=True)
        self.add_module(f'Linear_{n_layers - 1}', final_linear_layer)
        if use_float64:
            self.double()

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


class attnBlock(nn.Module):
    def __init__(self,
                 attn_head_key_dim,
                 num_attn_heads,
                 attn_mlp_hidden_dim,
                 dropout_rate,
                 use_float64,
                 if_self_attn=True
                 ):
        super(attnBlock, self).__init__()
        self.attn_head_key_dim = attn_head_key_dim
        self.num_attn_heads = num_attn_heads
        self.attn_mlp_hidden_dim = attn_mlp_hidden_dim
        self.if_self_attn = if_self_attn

        self.dropout_rate = dropout_rate

        self.dtype = torch.float32
        if use_float64:
            self.dtype = torch.float64

        # Multi-head self-attention.
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=self.attn_head_key_dim,  # Size of each attention head for query Q and key K.
            num_heads=self.num_attn_heads,
            dropout=self.dropout_rate,
            batch_first=True,
            dtype=self.dtype
        )

        self.ffn = _MLP(2, self.attn_head_key_dim, self.attn_mlp_hidden_dim, self.attn_head_key_dim,
                       self.dropout_rate, 'ReLU', use_float64)
        # Layer normalization.
        self.layernorm1 = torch.nn.LayerNorm(self.attn_head_key_dim, eps=1e-6, dtype=self.dtype)
        self.layernorm2 = torch.nn.LayerNorm(self.attn_head_key_dim, eps=1e-6, dtype=self.dtype)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = torch.nn.Dropout(p=self.dropout_rate)

        if use_float64:
            self.double()

    def forward(self, x, q, attn_mask=None):
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output, _ = self.mha(
            query=q,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attn_mask=attn_mask  # A boolean mask that prevents attention to certain positions.
        )

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(q + attn_output)  # Shape `(batch_size, T, d_model)`

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, T, d_model)`
        ffn_output = self.dropout1(ffn_output)

        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, T, d_model)`.

        return out2
