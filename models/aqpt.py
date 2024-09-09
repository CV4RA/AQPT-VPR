import torch
import torch.nn as nn
from transformers import ViTModel

class MultiChannelAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(MultiChannelAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class AQPT(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_heads=8, dropout=0.1):
        super(AQPT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.attention_layers = nn.ModuleList([MultiChannelAttention(hidden_size, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, 256)  

    def forward(self, x):
        vit_output = self.vit(x).last_hidden_state
        attn_output = vit_output
        for attn_layer in self.attention_layers:
            attn_output = attn_layer(attn_output)
        
        hash_codes = self.fc(attn_output[:, 0])  
        return hash_codes
