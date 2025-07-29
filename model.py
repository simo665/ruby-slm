import torch
import torch.nn as nn

class RubyMini(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, seq_length=128):
        super(RubyMini, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(seq_length, embedding_dim)  
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        x = self.embed(x) + self.pos_embedding(positions)
        
        # No need to permute if batch_first=True
        x = self.transformer(x)
        
        out = self.fc(x)
        return out

