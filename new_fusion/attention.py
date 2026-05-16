import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """
    Standard Transformer Cross-Attention.
    Q comes from Modality A.
    K, V come from Modality B.
    """
    def __init__(self, dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, q, kv):
        # q:  [Batch, Seq_Len, Dim]
        # kv: [Batch, Seq_Len, Dim]
        
        # Attention
        attn_out, _ = self.multihead_attn(query=q, key=kv, value=kv)
        
        # Add & Norm
        x = self.norm1(q + attn_out)
        
        # FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class BiDirectionalCrossAttention(nn.Module):
    """
    Executes Audio->Visual and Visual->Audio attention simultaneously.
    """
    def __init__(self, dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.audio_queries_video = CrossAttentionBlock(dim, num_heads, dropout)
        self.video_queries_audio = CrossAttentionBlock(dim, num_heads, dropout)

    def forward(self, audio_seq, video_seq):
        # Audio looks at Video context
        a_out = self.audio_queries_video(q=audio_seq, kv=video_seq)
        
        # Video looks at Audio context
        v_out = self.video_queries_audio(q=video_seq, kv=audio_seq)
        
        return a_out, v_out