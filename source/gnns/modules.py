import torch
from torch import nn
from dgl import ops


class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        x = x + x_res

        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, num_inputs=1, hidden_dim_multiplier=1, dropout=0, **kwargs):
        super().__init__()
        input_dim = int(dim * num_inputs)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GCNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier=1, dropout=0, **kwargs):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        in_degrees = graph.in_degrees().float()
        out_degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, out_degrees, in_degrees)
        degree_edge_products[degree_edge_products == 0] = 1
        norm_coefs = 1 / degree_edge_products.sqrt()

        x = ops.u_mul_e_sum(graph, x, norm_coefs)

        x = self.feed_forward_module(graph, x)

        return x


class GraphSAGEModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier=1, dropout=0, **kwargs):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     num_inputs=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        message = ops.copy_u_mean(graph, x)
        x = torch.cat([x, message], axis=1)

        x = self.feed_forward_module(graph, x)

        return x


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')


class GATModule(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim_multiplier=1, dropout=0, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.feed_forward_module(graph, x)

        return x


class GATSepModule(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim_multiplier=1, dropout=0, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     num_inputs=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)

        x = self.feed_forward_module(graph, x)

        return x


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout=0, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_scores_multiplier = 1 / torch.tensor(self.head_dim).sqrt()

        self.attn_qkv_linear = nn.Linear(in_features=dim, out_features=dim * 3)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        qkvs = self.attn_qkv_linear(x)
        qkvs = qkvs.reshape(-1, self.num_heads, self.head_dim * 3)
        queries, keys, values = qkvs.split(split_size=(self.head_dim, self.head_dim, self.head_dim), dim=-1)

        attn_scores = ops.u_dot_v(graph, keys, queries) * self.attn_scores_multiplier
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = ops.u_mul_e_sum(graph, values, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.output_linear(x)
        x = self.dropout(x)

        return x


class TransformerAttentionSepModule(nn.Module):
    def __init__(self, dim, num_heads, dropout=0, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_scores_multiplier = 1 / torch.tensor(self.head_dim).sqrt()

        self.attn_qkv_linear = nn.Linear(in_features=dim, out_features=dim * 3)

        self.output_linear = nn.Linear(in_features=dim * 2, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        qkvs = self.attn_qkv_linear(x)
        qkvs = qkvs.reshape(-1, self.num_heads, self.head_dim * 3)
        queries, keys, values = qkvs.split(split_size=(self.head_dim, self.head_dim, self.head_dim), dim=-1)

        attn_scores = ops.u_dot_v(graph, keys, queries) * self.attn_scores_multiplier
        attn_probs = ops.edge_softmax(graph, attn_scores)

        message = ops.u_mul_e_sum(graph, values, attn_probs)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)

        x = self.output_linear(x)
        x = self.dropout(x)

        return x
