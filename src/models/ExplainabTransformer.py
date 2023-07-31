
import copy
import torch
import torch.nn as nn
from src.models.spoter_model_original import SPOTERTransformerDecoderLayer, _get_clones



class ExplainabTransformerwQuery(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim=55, n_heads = 9, max_seq_len = 50):
        super().__init__()

        self.wEnc = nn.Parameter(torch.ones(1,1, hidden_dim)) #before it was called self.pos > careful when trying to load them

        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))

        self.transformer = nn.Transformer(hidden_dim, n_heads, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)


        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048,0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)


    def forward(self, inputs):

        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        explainabInputsEnc = h*self.wEnc

        h = self.transformer(explainabInputsEnc, self.class_query.unsqueeze(0))
        res = self.linear_class(h)

        return res


class ExplainabTransformerwSequence(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim=55, n_heads=9, max_seq_len=50):
        super().__init__()

        self.wEnc = nn.Parameter(torch.ones(1, 1, hidden_dim))  # before it was called self.pos > careful when trying to load them

        self.wDec = nn.Parameter(torch.ones(1, 1, hidden_dim))

        self.transformer = nn.Transformer(hidden_dim, n_heads, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)


    def forward(self, inputs):
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        explainabInputsEnc = h * self.wEnc
        explainabInputsDec = h * self.wDec

        h = self.transformer(explainabInputsEnc, explainabInputsDec)

        #Avg Pooling of the timesteps
        pooled = torch.mean(h, dim=0)
        res = self.linear_class(pooled)

        return res


