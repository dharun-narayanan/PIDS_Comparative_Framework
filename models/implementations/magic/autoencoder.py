"""
GMAE (Graph Masked Autoencoder) Model for MAGIC.
Implements masked graph representation learning with autoencoder architecture.
"""

import torch
import torch.nn as nn
from functools import partial
from .gat import GAT
from .loss_func import sce_loss
from .utils import create_norm


def build_model(args):
    """
    Build MAGIC model from arguments.
    
    Args:
        args: Model configuration arguments
        
    Returns:
        GMAEModel instance
    """
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    mask_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim

    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        mask_rate=mask_rate,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l
    )
    return model


class GMAEModel(nn.Module):
    """
    Graph Masked Autoencoder Model.
    
    Implements masked graph representation learning where node attributes
    are masked during encoding and reconstructed during decoding.
    
    Args:
        n_dim: Node feature dimension
        e_dim: Edge feature dimension
        hidden_dim: Hidden dimension
        n_layers: Number of GAT layers
        n_heads: Number of attention heads
        activation: Activation function name
        feat_drop: Feature dropout rate
        negative_slope: Negative slope for LeakyReLU
        residual: Whether to use residual connections
        norm: Normalization layer name
        mask_rate: Rate of node masking
        loss_fn: Loss function name
        alpha_l: Alpha parameter for loss
    """
    
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        super(GMAEModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')

        # Initialize weights function
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Edge reconstruction network
        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)

        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads

        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim

        # Build encoder
        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
        )

        # Build decoder for attribute prediction
        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
        )

        # Mask token for encoding
        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)

        # Setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        
        # Store dimensions
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    @property
    def output_hidden_dim(self):
        """Get output hidden dimension."""
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        """Setup loss function."""
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "mse":
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        """
        Apply masking to node features.
        
        Args:
            g: DGL graph
            mask_rate: Rate of nodes to mask
            
        Returns:
            Masked graph and (mask_nodes, keep_nodes) tuple
        """
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # Random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        # Replace masked nodes with mask token
        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token.to(g.device)

        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g):
        """
        Forward pass computing loss.
        
        Args:
            g: DGL graph
            
        Returns:
            Total loss (feature reconstruction + edge reconstruction)
        """
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        """
        Compute reconstruction loss.
        
        Args:
            g: DGL graph
            
        Returns:
            Total reconstruction loss
        """
        # Feature Reconstruction
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        use_g = pre_use_g
        
        # Encode
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        # Decode
        recon = self.decoder(pre_use_g, rep)
        x_init = g.ndata['attr'][mask_nodes]
        x_rec = recon[mask_nodes]

        # Compute feature reconstruction loss
        loss_rec = self.criterion(x_rec, x_init)
        
        # Edge Reconstruction (optional)
        # This can be added if edge reconstruction is needed
        loss_edge = torch.tensor(0.0, device=g.device)
        
        return loss_rec + loss_edge

    def embed(self, g):
        """
        Get node embeddings without masking.
        
        Args:
            g: DGL graph
            
        Returns:
            Node embeddings
        """
        x = g.ndata['attr'].to(g.device)
        enc_rep, all_hidden = self.encoder(g, x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        return enc_rep
    
    def get_embeddings(self, g):
        """Alias for embed() for compatibility."""
        return self.embed(g)
