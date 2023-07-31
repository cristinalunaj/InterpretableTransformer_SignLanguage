
import copy
import torch, math
import torch.nn as nn



class PositionalEncodingSinCos(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        posEnc = self.pe[:x.size(0)]
        xAndPosEnc = x + posEnc
        return self.dropout(xAndPosEnc)


class PositionalEncodingRandom(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        posEnc = self.pe[:x.size(0)]
        xAndPosEnc = x + posEnc
        return self.dropout(xAndPosEnc)





class BaselineTransformerClassification(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim=55, n_heads = 9, max_seq_len = 50):
        super().__init__()

        self.output_pos_encoding = PositionalEncodingSinCos(hidden_dim, 0.5)

        self.transformer = nn.Transformer(hidden_dim, n_heads, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)


    def forward(self, inputs):

        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        #Apply positional emb.
        henc = self.output_pos_encoding(h)


        h = self.transformer(henc, henc)
        # Avg. Pooling in timesteps
        pooled = torch.mean(h, dim=0)
        res = self.linear_class(pooled)

        return res

