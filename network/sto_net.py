"""
StoPred Network Implementation

This module implements the StoPred neural network for predicting protein stoichiometry 
and subunit counts from sequence and structural features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import ml_collections
import numpy as np
from typing import Dict, List, Tuple, Optional


def parse_sto(sto: str) -> List[int]:
    """
    Parse stoichiometry string into list of integers.
    
    Args:
        sto (str): Stoichiometry string representation
        
    Returns:
        List[int]: Parsed stoichiometry as a list of integers
    """
    if sto == 'other':
        return ['other']
    sto = str(sto).replace('(', '').replace(')', '')
    sto_counts = [int(i) for i in sto.split(',') if i.strip()]
    return sto_counts


class FeatureBlock(nn.Module):
    """Feature processing block with batch normalization and multiple layers."""
    
    def __init__(
        self, 
        embedding_dim: int, 
        hidden_dim: int, 
        dropout_rate: float, 
        num_layers: int, 
        batch_norm: bool = True
    ):
        """
        Initialize FeatureBlock.
        
        Args:
            embedding_dim (int): Dimension of input embeddings
            hidden_dim (int): Hidden dimension for linear layers
            dropout_rate (float): Dropout rate for regularization
            num_layers (int): Number of layers in the block
            batch_norm (bool): Whether to use batch normalization
        """
        super().__init__()
        self.batch_norm = batch_norm

        self.bn = nn.BatchNorm1d(embedding_dim)
        self.phi = nn.ModuleList()
        self.layernorms = nn.ModuleList()

        for _ in range(num_layers):
            if len(self.phi) == 0:
                input_dim = embedding_dim
            else:
                input_dim = hidden_dim
            
            self.phi.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout_rate)
            ))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the feature block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_subunits, embedding_dim)
            mask (torch.Tensor, optional): Mask for padding positions
            
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # x (batch_size, num_subunits, embedding_dim)
        if self.batch_norm:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        
        for i, layer in enumerate(self.phi):
            # Apply layer
            residual = x
            x = layer(x)
            if i > 0:
                x = x + residual
        return x


class SubunitGAT(nn.Module):
    """Subunit Graph Attention Network."""
    
    def __init__(
        self, 
        feature_dim: int, 
        dropout_rate: float = 0.2, 
        num_heads: int = 6
    ):
        """
        Initialize SubunitGAT.
        
        Args:
            feature_dim (int): Dimension of input features
            dropout_rate (float): Dropout rate for regularization
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Graph attention components
        self.W = nn.Parameter(torch.zeros(num_heads, feature_dim, self.head_dim))
        nn.init.xavier_uniform_(self.W)

        # Attention mechanism
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim, 1))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        # Output projection
        self.linear = nn.Linear(feature_dim * 2, feature_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAT layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_subunits, feature_dim)
            mask (torch.Tensor): Mask for padding positions
            
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        batch_size, num_subunits, _ = x.shape

        # Apply mask
        x_masked = x * mask.unsqueeze(-1)

        # Transform input features
        Wh = torch.einsum('bnd,hdf->bnhf', x_masked, self.W)

        # Compute attention coefficients
        e_src = torch.einsum('bnhf,hfc->bnh', Wh, self.a_src)
        e_dst = torch.einsum('bnhf,hfc->bnh', Wh, self.a_dst)

        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)
        e = self.leakyrelu(e)

        # Mask self-attention
        self_mask = torch.eye(num_subunits, device=x.device).bool().unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(self_mask, float('-inf'))

        # Padding mask (both source and target positions)
        padding_mask_src = (~mask.bool()).unsqueeze(2).unsqueeze(-1)
        padding_mask_dst = (~mask.bool()).unsqueeze(1).unsqueeze(-1)
        e = e.masked_fill(padding_mask_src | padding_mask_dst, float('-inf'))

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = attention.masked_fill(attention != attention, 0)  # replace NaNs
        attention = self.attn_dropout(attention)

        # Aggregate features
        h_prime = torch.einsum('bnmh,bmhf->bnhf', attention, Wh)
        out = h_prime.reshape(batch_size, num_subunits, self.feature_dim)

        # Concatenate with original features
        out = torch.cat([x_masked, out], dim=-1)

        # Final transformation
        out = self.linear(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Residual connection
        out = out + x_masked

        return out

class SubunitGCN(nn.Module):
    """Subunit Graph Convolutional Network."""
    
    def __init__(self, feature_dim: int, dropout_rate: float = 0.1):
        """
        Initialize SubunitGCN.
        
        Args:
            feature_dim (int): Dimension of input features
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        self.linear = nn.Linear(feature_dim*2, feature_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GCN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_subunits, feature_dim)
            mask (torch.Tensor): Mask for padding positions
            
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # x (batch_size, num_subunits, input_dim)
        # mask (batch_size, num_subunits) where 0 is padding

        # Apply mask to ignore padding subunits
        x_masked = x * mask.unsqueeze(-1)
        
        # Sum across subunits, but exclude the current subunit
        x_sum = x_masked.sum(dim=1, keepdim=True) - x
        
        # Fill the mask to 0
        x_sum = x_sum * mask.unsqueeze(-1)
        
        # Stack the sum features with the original features
        x = torch.cat([x, x_sum], dim=-1)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SubunitMLP(nn.Module):
    """Subunit Multi-Layer Perceptron."""
    
    def __init__(self, feature_dim: int, dropout_rate: float = 0.1):
        """
        Initialize SubunitMLP.
        
        Args:
            feature_dim (int): Dimension of input features
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_subunits, feature_dim)
            mask (torch.Tensor): Mask for padding positions
            
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # x (batch_size, num_subunits, input_dim)
        # mask (batch_size, num_subunits) where 0 is padding
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class StoPredNet(pl.LightningModule):
    """Main StoPred network for protein stoichiometry prediction."""
    
    def __init__(self, config: ml_collections.ConfigDict):
        """
        Initialize StoPredNet.
        
        Args:
            config (ml_collections.ConfigDict): Configuration dictionary with model parameters
        """
        super().__init__()
        self.config = config
        self.num_subunits = config['num_subunits']  # e.g., 3
        self.dropout = config['dropout']
        self.features = config['features']  # dict
        self.num_feature_layers = config['num_feature_layers']
        self.hidden_dim = config['hidden_dim']

        self.sto2idx = config['sto2idx']
        self.count2label = config['count2label']
        self.label2idx = config['label2idx']
        self.num_labels = len(self.label2idx)

        self.num_gnn_layers = config.get('num_gnn_layers', 2)
        self.config['num_gnn_layers'] = self.num_gnn_layers
        self.num_heads = config.get('num_heads', 6)
        self.config['num_heads'] = self.num_heads
        
        # Get the agg_methods, gcn or gat
        self.agg_methods = config.get('agg_methods', 'gat')
        self.use_moe = config.get('use_moe', False)
        self.use_global_state = config.get('use_global_state', True)
        num_features = len(self.features)

        # learning rate, weight_local, weight_global
        self.learning_rate = config.get('learning_rate', 0.0005)
        self.weight_local = config.get('weight_local', 0.5)
        self.weight_global = config.get('weight_global', 0.5)
        
        self.featuresBlock = nn.ModuleDict()
        for feature, params in self.features.items():
            self.featuresBlock[feature] = FeatureBlock(
                embedding_dim=params['dim'],
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout,
                num_layers=self.num_feature_layers,
                batch_norm=True,
            )

        self.gnnBlock = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            if self.agg_methods == 'gcn':
                self.gnnBlock.append(SubunitGCN(self.hidden_dim*num_features, self.dropout))
            elif self.agg_methods == 'gat':
                self.gnnBlock.append(SubunitGAT(self.hidden_dim*num_features, self.dropout))
            elif self.agg_methods == 'mlp':
                self.gnnBlock.append(SubunitMLP(self.hidden_dim*num_features, self.dropout))
            else:
                raise ValueError('Unsupported aggregation method')

        self.compressBlock = nn.Sequential(
            nn.Linear(self.hidden_dim*num_features, self.hidden_dim//2),
            nn.ELU(),
            nn.Linear(self.hidden_dim//2, self.hidden_dim//4),
            nn.ELU(),
        )
        
        # Process combined features for output
        self.localState = nn.Sequential(
            nn.Linear(self.hidden_dim//4, self.num_labels)
        )

        # Create a MOE for global state prediction
        if self.use_moe:
            self.globalState, self.subunit_to_sto_indices = self._build_MoE_global_state()
        else:
            self.globalState = nn.Sequential(
                nn.Linear(self.hidden_dim//4, len(self.sto2idx))
            )
    
    def forward(
        self, 
        batch: Dict[str, torch.Tensor], 
        return_embedding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch containing features and mask
            return_embedding (bool): Whether to return embeddings instead of predictions
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Local state predictions and global state predictions,
                or embedding if return_embedding=True
        """
        mask = batch['mask']

        # Transform each type of feature independently
        features = {
            feature: self.featuresBlock[feature](batch[feature], mask)
            for feature in self.features
        }
        
        # Concatenate all features
        features = torch.cat([features[feature] for feature in features], dim=-1)
        
        # Apply GCN layers
        for gnn in self.gnnBlock:
            features = gnn(features, mask)
        
        x = self.compressBlock(features)
        if return_embedding:
            return x
        local_state = self.localState(x)  # batch_size, num_subunits, num_labels
        if self.use_moe:
            global_state = self._compute_global_state(x, mask)
        else:
            global_state = self.globalState(x)
        return local_state, global_state

    def _build_MoE_global_state(self):
        """
        Build Mixture of Experts for global state prediction.
        
        Returns:
            Tuple[nn.ModuleDict, Dict]: Global state module and subunit to stoichiometry mapping
        """
        self.max_subunits = max(len(parse_sto(sto)) for sto in self.sto2idx.keys() if sto != 'other')
        self.globalState = nn.ModuleDict()
        self.globalState['other'] = nn.Sequential(
            nn.Linear(self.hidden_dim//4, 1)  # Only predict 'other' vs not 'other'
        )
        
        for num_subunits in range(1, self.max_subunits + 1):
            # Filter stoichiometries that are possible for this number of subunits
            valid_stos = [sto for sto in self.sto2idx.keys() 
                        if sto == 'other' or len(parse_sto(sto)) == num_subunits]
            if valid_stos:  # Only create expert if there are valid stoichiometries
                self.globalState[str(num_subunits)] = nn.Sequential(
                    nn.Linear(self.hidden_dim//4, len(valid_stos))
                )
        
        # Store mapping from subunit count to valid stoichiometry indices
        self.subunit_to_sto_indices = {}
        for num_subunits in range(1, self.max_subunits + 1):
            valid_stos = [sto for sto in self.sto2idx.keys() 
                        if sto == 'other' or len(parse_sto(sto)) == num_subunits]
            if valid_stos:
                self.subunit_to_sto_indices[num_subunits] = [self.sto2idx[sto] for sto in valid_stos]
        return self.globalState, self.subunit_to_sto_indices

    @torch.no_grad()
    def _make_sto_index_tensor(self, sto_idx_list: List[int], device: torch.device) -> torch.Tensor:
        """
        Cache helper to create 1-D LongTensor only once per expert.
        
        Args:
            sto_idx_list (List[int]): List of stoichiometry indices
            device (torch.device): Device to place tensor on
            
        Returns:
            torch.Tensor: Tensor with stoichiometry indices
        """
        key = tuple(sto_idx_list)
        if not hasattr(self, "_sto_idx_cache"):
            self._sto_idx_cache = {}
        if key not in self._sto_idx_cache:
            self._sto_idx_cache[key] = torch.tensor(sto_idx_list, dtype=torch.long,
                                                    device=device)
        return self._sto_idx_cache[key]

    def _compute_global_state(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute global state using Mixture of Experts.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, S_max, H)
            mask (torch.Tensor): Mask for padding positions
            
        Returns:
            torch.Tensor: Global state logits of shape (B, S_max, |𝒯|)
        """
        B, S_max, _ = x.shape
        n_sto = len(self.sto2idx)
        NEG_LARGE = -1e9  # finite -> avoids NaNs

        # (B, S_max, |𝒯|) – start with large negative logits everywhere
        logits = x.new_full((B, S_max, n_sto), NEG_LARGE)

        # 1 "other" expert (always valid) – one shot for the whole batch
        logits[..., self.sto2idx["other"]] = self.globalState["other"](x).squeeze(-1)

        # 2 call experts once per distinct sub-unit count in the batch
        n_sub = mask.sum(1).long()  # (B,)
        unique_k = n_sub.unique()
        for k in unique_k:
            k_int = int(k)
            if k_int not in self.subunit_to_sto_indices:
                continue  # no expert → skip

            idx_in_batch = (n_sub == k_int).nonzero(as_tuple=True)[0]  # (B_k,)
            x_k = x[idx_in_batch]  # (B_k, S_max, H)
            out_k = self.globalState[str(k_int)](x_k)[:, :k_int, :]  # (B_k, k_int, |T_k|)

            # Index tensors with broadcast-compatible shapes
            row_idx = idx_in_batch[:, None, None]  # (B_k, 1, 1)
            sub_idx = torch.arange(k_int, device=x.device)[None, :, None]  # (1, k_int, 1)
            sto_idx = self._make_sto_index_tensor(
                self.subunit_to_sto_indices[k_int], x.device)[None, None, :]  # (1,1,|T_k|)

            logits[row_idx, sub_idx, sto_idx] = out_k  # assign in one go

        # Make sure mask is boolean
        if mask.dtype != torch.bool:
            mask_bool = mask > 0  # (B, S_max)   True = real sub-unit
        else:
            mask_bool = mask
            
        # Mask out padding sub-units so they never affect the loss
        logits = logits.masked_fill(~mask_bool.unsqueeze(-1), NEG_LARGE)
        return logits

    def custom_loss_function(
        self, 
        local_state: torch.Tensor,
        global_state: torch.Tensor,
        gt_local: torch.Tensor,
        gt_global: torch.Tensor, 
        weight_local: float = 0.5, 
        weight_global: float = 0.5, 
        padding_value: int = -100, 
        ignore_global: bool = False
    ) -> torch.Tensor:
        """
        Calculate weighted loss combining local and global predictions.
        
        Args:
            local_state (torch.Tensor): Predicted local state of shape (batch_size, num_subunits, num_labels)
            global_state (torch.Tensor): Predicted global state of shape (batch_size, num_subunits, num_labels)
            gt_local (torch.Tensor): Ground truth local labels of shape (batch_size, num_subunits, num_labels)
            gt_global (torch.Tensor): Ground truth global labels of shape (batch_size, num_subunits, num_labels)
            weight_local (float): Weight for local loss component
            weight_global (float): Weight for global loss component
            padding_value (int): Value used to identify padded positions
            ignore_global (bool): Whether to ignore the global loss
            
        Returns:
            torch.Tensor: Combined weighted loss
        """
        # Convert one-hot encoded ground truth to class indices
        gt_local_labels = torch.argmax(gt_local, dim=-1)  # (batch_size, num_subunits)
        gt_global_labels = torch.argmax(gt_global, dim=-1)  # (batch_size, num_subunits)
        
        # Create mask to ignore padded subunits
        mask = gt_local_labels != padding_value  # (batch_size, num_subunits)
        
        # Reshape predictions and targets for loss computation
        local_state_flat = local_state.view(-1, local_state.size(-1))  # (batch_size * num_subunits, num_labels)
        global_state_flat = global_state.view(-1, global_state.size(-1))  # (batch_size * num_subunits, num_labels)
        gt_local_flat = gt_local_labels.view(-1)  # (batch_size * num_subunits)
        gt_global_flat = gt_global_labels.view(-1)  # (batch_size * num_subunits)
        
        # Apply mask to filter out padded positions
        mask_flat = mask.view(-1)
        valid_local_state = local_state_flat[mask_flat]
        valid_gt_local = gt_local_flat[mask_flat]
        valid_global_state = global_state_flat[mask_flat]
        valid_gt_global = gt_global_flat[mask_flat]

        # Calculate losses using cross-entropy
        local_loss = F.cross_entropy(valid_local_state, valid_gt_local, ignore_index=-100)
        if not self.use_global_state:
            return local_loss
        
        global_loss = F.cross_entropy(valid_global_state, valid_gt_global, ignore_index=-100)

        # Combine losses with weights
        total_loss = (weight_local * local_loss) + (weight_global * global_loss)  # + joint_loss
        return total_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the model.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
            batch_idx (int): Batch index
            
        Returns:
            torch.Tensor: Loss value
        """
        y_hat, y_hat_global = self(batch)
        y = batch['labels']
        y_global = batch['labels_global']
        loss = self.custom_loss_function(y_hat, y_hat_global, y, y_global)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step for the model.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
            batch_idx (int): Batch index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing validation loss
        """
        y_hat, y_hat_global = self(batch)
        y = batch['labels']
        y_global = batch['labels_global']
        loss = self.custom_loss_function(y_hat, y_hat_global, y, y_global)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Optimizer instance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        # Return DataLoader of your training dataset
        pass

    def val_dataloader(self):
        # Return DataLoader of your validation dataset
        pass

    @staticmethod
    def load_from_pkl(filename: str) -> 'StoPredNet':
        """
        Load model from pickle file.
        
        Args:
            filename (str): Path to the pickle file
            
        Returns:
            StoPredNet: Loaded model instance
        """
        model_dict = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
        model = StoPredNet(model_dict['config'])
        model.load_state_dict(model_dict['state_dict'], strict=False)
        return model

    def save_to_pkl(self, filename: str) -> None:
        """
        Save model to pickle file.
        
        Args:
            filename (str): Path to the output pickle file
        """
        model_dict = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(model_dict, filename)

