"""
Graph Attention Network (GAT) implementation for MAGIC.
Standalone version adapted from the original MAGIC repository.
"""

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from .utils import create_activation, create_norm


class GAT(nn.Module):
    """
    Multi-layer Graph Attention Network.
    
    Args:
        n_dim: Node feature dimension
        e_dim: Edge feature dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        n_layers: Number of GAT layers
        n_heads: Number of attention heads
        n_heads_out: Number of attention heads for output layer
        activation: Activation function name
        feat_drop: Feature dropout rate
        attn_drop: Attention dropout rate
        negative_slope: Negative slope for LeakyReLU
        residual: Whether to use residual connections
        norm: Normalization layer
        concat_out: Whether to concatenate output heads
        encoding: Whether this is an encoder (affects last layer)
    """
    
    def __init__(self,
                 n_dim,
                 e_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 n_heads,
                 n_heads_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.gats = nn.ModuleList()
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if self.n_layers == 1:
            self.gats.append(GATConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, concat_out=self.concat_out
            ))
        else:
            # First layer
            self.gats.append(GATConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=self.concat_out
            ))
            # Middle layers
            for _ in range(1, self.n_layers - 1):
                self.gats.append(GATConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=self.concat_out
                ))
            # Last layer
            self.gats.append(GATConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
            ))
        self.head = nn.Identity()

    def forward(self, g, input_feature, return_hidden=False):
        """
        Forward pass through GAT layers.
        
        Args:
            g: DGL graph
            input_feature: Input node features
            return_hidden: Whether to return hidden representations from all layers
            
        Returns:
            Output features and optionally hidden features from all layers
        """
        h = input_feature
        hidden_list = []
        for layer in range(self.n_layers):
            h = self.gats[layer](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        """Reset the classifier head for fine-tuning."""
        self.head = nn.Linear(self.n_heads * self.out_dim, num_classes)


class GATConv(nn.Module):
    """
    Graph Attention Convolution Layer.
    
    Implements attention mechanism over node and edge features.
    """
    
    def __init__(self,
                 in_dim,
                 e_dim,
                 out_dim,
                 n_heads,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.src_feat, self.dst_feat = expand_as_pair(in_dim)
        self.edge_feat = e_dim
        self.out_feat = out_dim
        self.allow_zero_in_degree = allow_zero_in_degree
        self.concat_out = concat_out

        # Feature transformation layers
        if isinstance(in_dim, tuple):
            self.fc_node_embedding = nn.Linear(
                self.src_feat, self.out_feat * self.n_heads, bias=False)
            self.fc_src = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
            self.fc_dst = nn.Linear(self.dst_feat, self.out_feat * self.n_heads, bias=False)
        else:
            self.fc_node_embedding = nn.Linear(
                self.src_feat, self.out_feat * self.n_heads, bias=False)
            self.fc = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
        
        # Edge feature transformation
        self.edge_fc = nn.Linear(self.edge_feat, self.out_feat * self.n_heads, bias=False)
        
        # Attention parameters
        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        
        # Dropout and activation
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        else:
            self.register_buffer('bias', None)
        
        # Residual connection
        if residual:
            if self.dst_feat != self.n_heads * self.out_feat:
                self.res_fc = nn.Linear(
                    self.dst_feat, self.n_heads * self.out_feat, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        
        self.reset_parameters()
        self.activation = activation
        self.norm = norm
        if norm is not None:
            self.norm = norm(self.n_heads * self.out_feat)

    def reset_parameters(self):
        """Initialize parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        """Set whether to allow zero in-degree nodes."""
        self.allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        """
        Forward pass through GAT convolution.
        
        Args:
            graph: DGL graph
            feat: Node features
            get_attention: Whether to return attention weights
            
        Returns:
            Updated node features and optionally attention weights
        """
        edge_feature = graph.edata['attr']
        with graph.local_scope():
            # Handle different input formats
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self.n_heads, self.out_feat)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self.n_heads, self.out_feat)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self.n_heads, self.out_feat)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self.n_heads, self.out_feat)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self.n_heads, self.out_feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            
            # Transform edge features
            edge_prefix_shape = edge_feature.shape[:-1]
            feat_edge = self.edge_fc(edge_feature).view(
                *edge_prefix_shape, self.n_heads, self.out_feat)
            
            # Compute attention scores
            eh = (feat_src * self.attn_h).sum(-1).unsqueeze(-1)
            et = (feat_dst * self.attn_t).sum(-1).unsqueeze(-1)
            ee = (feat_edge * self.attn_e).sum(-1).unsqueeze(-1)

            graph.srcdata.update({'hs': feat_src, 'eh': eh})
            graph.dstdata.update({'et': et})
            graph.edata.update({'ee': ee, 'ef': feat_edge})
            
            # Compute attention
            graph.apply_edges(fn.u_add_e('eh', 'ee', 'ee'))
            graph.apply_edges(fn.e_add_v('ee', 'et', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            
            # Apply attention dropout
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            
            # Message passing
            graph.update_all(fn.u_mul_e('hs', 'a', 'm'),
                           fn.sum('m', 'ft'))
            graph.update_all(fn.e_mul_e('ef', 'a', 'em'),
                           fn.sum('em', 'fet'))
            
            rst = graph.dstdata['ft'] + graph.dstdata['fet']
            
            # Apply bias
            if self.bias is not None:
                rst = rst + self.bias
            
            # Residual connection
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, self.n_heads, self.out_feat)
                rst = rst + resval
            
            # Reshape output
            if self.concat_out:
                rst = rst.flatten(1)
            else:
                rst = rst.mean(1)
            
            # Apply activation
            if self.activation:
                rst = self.activation(rst)
            
            # Apply normalization
            if self.norm is not None:
                rst = self.norm(rst)
            
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
