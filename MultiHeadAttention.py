import torch.nn as nn
import torch 
import numpy as np
import tiktoken

class MultiHeadAttention(nn.modules):

    def __init__(self, d_in, d_out, context_length, dropout, num_head, bias = False):

        self.d_out = d_out 
        self.num_head = num_head
        self.context_length = context_length
        
        self.W_query = nn.Linear(d_in, d_out * num_head, bias=bias)
        self.W_key = nn.Linear(d_in, d_out *  num_head, bias=bias)
        self.W_value = nn.Linear(d_in, d_out * num_head, bias=bias)

        self.out_proj = nn.Linear(d_out*num_head, d_out*num_head)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
    
    def forward(self,x):
        b, num_token, d_in = x.shape


        keys = self.W_key
        queries = self.W_query
        values = self.W_value


        keys = keys.view(b, num_token, self.num_head, self.d_out)
        values = values.view(b, num_token, self.num_head, self.d_out)
        queries = queries.view(b, num_token, self.num_head, self.d_out)

        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)


        attention_score = queries @ keys.tranpose(2,3)
        mask_bool = self.mask.bool()[:num_token,:num_token]

        attention_score.masked_fill(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_score/attention_score.shape[-1]**0.5, dim=1)
        attention_weights = self.dropout(attention_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attention_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, self.num_head, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)





