import torch
from torch import nn
import torch

from model.patchembeddings import PatchEmbedding

from model.multiheadattention import MultiHeadAttention

class ViTMLP(nn.Module):
    def __init__(self,
                 mlp_num_hiddens, 
                 mlp_num_outputs, 
                 dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))

class ViTBlock(nn.Module):
    def __init__(self, 
                 num_hiddens,
                 norm_shape, 
                 mlp_num_hiddens,
                 num_heads, 
                 dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads,dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X_norm = self.ln1(X)
        X = X + self.attention(X_norm, X_norm, X_norm, valid_lens)
        X_norm = self.ln2(X)
        return X + self.mlp(X_norm)
        
class ViT(nn.Module):
    """Vision Transformer."""
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 num_hiddens, 
                 mlp_num_hiddens,
                 num_heads, 
                 num_blks, 
                 emb_dropout,
                 blk_dropout,
                 use_bias=False, 
                 num_classes=10):
        super().__init__()
        #self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(num_hiddens, num_hiddens, mlp_num_hiddens,
                                                  num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))
        
    def compute_embeddins(self,X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        return X

    def forward(self, X):
        X=self.compute_embeddins(X)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
    
   

