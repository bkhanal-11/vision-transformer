import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.dropout = nn.Dropout(dropout_rate)

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)

        self.W_o = nn.Linear(self.head_dim * self.num_heads, self.d_model)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.size(0)

        # Linear Transformation
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)
        
        # Reshaping Tensors
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim)

        # Calculate dot products between the queries and the keys
        matmul_qk = torch.einsum('bqhd,bkhd->bhqk', [queries, keys])

        dk = keys.size()[-1]
        scaled_attention_logits = matmul_qk / (dk ** 0.5)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask==0, float('-1e20')) 
        
        # Apply softmax function to obtain attention weights
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Apply dropout to the attention weights
        attention_weights_dropout = self.dropout(attention_weights)

        context = torch.einsum('bhqv,bvhd->bqhd', [attention_weights_dropout, values])

        context = context.reshape(batch_size, -1, self.num_heads*self.head_dim)

        # Concatenate heads and apply final linear transformation
        output = self.W_o(context)

        return output

if __name__ == "__main__":
    # Define input tensor
    batch_size = 32
    seq_len = 512
    d_model = 768
    num_heads = 12
    x = torch.randn(batch_size, seq_len, d_model)

    # Create MultiHeadAttention module
    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
    num_params = sum(p.numel() for p in mha.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    # Obtain output tensor
    output = mha(x, x, x)

    # Print output tensor shape
    print(output.shape)

