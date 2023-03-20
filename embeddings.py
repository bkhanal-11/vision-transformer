import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size=16, channels=3):
        super(PatchEmbedding, self).__init__()        
        # Define the patch embedding layer
        self.patch_embedding = nn.Conv2d(in_channels=channels, out_channels=d_model, 
                                             kernel_size=patch_size,
                                             stride=patch_size)
        
    def forward(self, x):        
        # Apply the patch embedding layer
        x = self.patch_embedding(x)
        
        # Reshape the output to have shape (batch_size, num_patches, embed_dim)
        x = x.flatten(2).transpose(1,2)
        
        return x

class ViTEmbedding(nn.Module):
    def __init__(self, d_model, image_size=224, patch_size=16, channels=3, dropout_rate=0.1):
        super(ViTEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        # Calculate the number of patches
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(d_model, patch_size, channels)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)

        return x
    
