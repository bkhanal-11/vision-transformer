import torch
import torch.nn as nn
import time

from utils import FullyConnected, LayerNorm
from multi_head_attention import MultiHeadAttention
from embeddings import ViTEmbedding

class ViTBlock(nn.Module):
    """
    The bert encoder layer is composed of a multi-head self-attention mechanism,
    followed by a simple, position-wise fully connected feed-forward network. 
    This architecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, num_heads, d_model, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(ViTBlock, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      d_model=d_model,
                                      dropout_rate=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=d_model,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm2 = LayerNorm(d_model, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, d_model)
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, d_model)
        """
        # calculate Self-Attention using Multi-Head Attention
        mha_output = self.mha(x, x, x, mask)  # Self attention (batch_size, input_seq_len, d_model)

        # skip connection
        # apply layer normalization on sum of the input and the attention output to get the output of the multi-head attention layer
        skip_x_attention = self.layernorm1(x + mha_output)

        # pass the output of the multi-head attention layer through a ffn
        ffn_output = self.ffn(skip_x_attention)

        # apply dropout layer to ffn output during training
        ffn_output = self.dropout_ffn(ffn_output)

        # apply layer normalization on sum of the output from multi-head attention (skip connection) and ffn output to get the output of the encoder layer
        encoder_layer_out = self.layernorm2(skip_x_attention + ffn_output)

        return encoder_layer_out

class ViT(nn.Module):
    """
    This BERT encoder is composed by a stack of identical layers (EncoderLayers).
    """
    def __init__(self, num_layers, num_heads, d_model, fully_connected_dim, image_size, patch_size, channels,
                 num_classes, dropout_rate=0.1, layernorm_eps=1e-6):
        super(ViT, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.embedding = ViTEmbedding(d_model, image_size, patch_size, channels, dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        
        self.enc_layers = nn.ModuleList([ViTBlock(num_heads, d_model, fully_connected_dim,
                                                       dropout_rate=dropout_rate, layernorm_eps=layernorm_eps)
                                          for _ in range(num_layers)])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        # Initialize the weights
        # self.apply(self._init_weights)
        
    def forward(self, x, mask=None):
        """
        Forward pass for the Encoder
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len)
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_out -- Tensor of shape (batch_size, input_seq_len, embedding_d_model)
        """
        # Add position encoding to the input
        x = self.embedding(x)
        
        # Apply dropout to the input
        x = self.dropout(x)
        
        # Pass the input through each encoder layer
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        
        return self.mlp_head(x[:, 0])
    