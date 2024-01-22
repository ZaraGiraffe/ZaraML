import torch 
from torch.nn import MultiheadAttention
from torch import nn as nn
from typing import Tuple


class QKVBlock(nn.Module):
    """
    QKV block of encoder block
    """
    def __init__(self, input_size, *args, **kwargs) -> None:
        """
        Initialize the parameters for attention
        """
        super().__init__(*args, **kwargs)
        self.Wk = nn.Linear(input_size, input_size)
        self.Wq = nn.Linear(input_size, input_size)
        self.Wv = nn.Linear(input_size, input_size)
    
    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs:
            X: the input of the shape (seq_len, batch, features)
        returns:
            tuple of [Queries, Keys, Values] each of the shape (seq_len, batch, num_features)
        """
        K = self.Wk(X)
        Q = self.Wq(X)
        V = self.Wv(X)
        return Q, K, V
    

class AttentionBlock(nn.Module):
    """
        the attention block of encoder block
        this module does not have parameters except layer normalization
    """
    def __init__(self, input_size, nheads, dropout=0.1, *args, **kwargs) -> None:
        """
        initialise some params for forward function
        """
        super().__init__(*args, **kwargs)
        self.nheads = nheads
        assert int(input_size) % int(nheads) == 0, "nheads should divide input_size"
        self.head_size = input_size // nheads
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, Q, K, V, mask=None, eps=1e5) -> torch.Tensor:
        """
        inputs:
            Q: queries
            K: keys
            V: values
            each of these inputs are of shape (seq_len, batch, num_features)
            mask: the mask for attention, has to be torch.Tensor of the shape (seq_size, seq_size)
            1 - means masked, 0 - unmasked
        returns:
            Y: the values after attention of the shape (seq_size, batch, input_size)
        """
        K = torch.reshape(K, (K.shape[0], K.shape[1], self.nheads, self.head_size))
        K = torch.permute(K, (1, 2, 3, 0)) # BHFS

        Q = torch.reshape(Q, (Q.shape[0], Q.shape[1], self.nheads, self.head_size))
        Q = torch.permute(Q, (1, 2, 0, 3)) # BHSF

        qk = torch.matmul(Q, K) # BHSS

        if mask is None:
            qk = torch.softmax(qk, dim=3)
        else:
            assert mask.shape[0] == V.shape[0] and mask.shape[1] == V.shape[0], "mask is of the wrong shape"
            new_mask = mask.repeat(Q.shape[0], Q.shape[1], 1, 1)
            qk = torch.softmax(qk + new_mask * -eps)

        V = torch.reshape(V, (V.shape[0], V.shape[1], self.nheads, self.head_size))
        V = torch.permute(V, (1, 2, 0, 3)) # BHSF

        Y = torch.matmul(qk, V) # BHSF
        Y = torch.permute(Y, (2, 0, 1, 3))
        Y = torch.reshape(Y, (Y.shape[0], Y.shape[1], Y.shape[2] * Y.shape[3])) # SBI
        Y = self.dropout(self.norm(Y))

        return Y


class ForwardBlock(nn.Module):
    """
    forward block of encoder block
    """
    def __init__(self, input_size, hidden_size=None, activation=nn.ReLU(), dropout=0.1, *args, **kwargs) -> None:
        """
        initialise some parameters
        """
        super().__init__(*args, **kwargs)
        if hidden_size is None:
            hidden_size = input_size * 2
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.activation=activation
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, X) -> torch.Tensor:
        """
        inputs:
            X: the input of the shape (seq_len, batch, features)
        returns:
            Y: the input of the shape (seq_len, batch, features)
        """
        H = self.activation(self.linear1(X))
        Y = self.activation(self.linear2(H))
        Y = self.dropout(self.norm(Y))
        return Y


class EncoderLayer(nn.Module):
    """
    complete multihead attention block similar to torch.nn.TransformerEncoderLayer
    """
    def __init__(self, input_size, nheads, hidden_size=None, dropout=0.1, activation=nn.ReLU(), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.qkv = QKVBlock(input_size)
        self.atten = AttentionBlock(input_size, nheads, dropout=dropout)
        self.forw = ForwardBlock(input_size, hidden_size, activation=activation, dropout=dropout)

    def forward(self, X, mask=None) -> torch.Tensor:
        """
        inputs:
            X: the input of the shape (seq_len, batch, features)
            mask: the mask for attention, has to be torch.Tensor of the shape (seq_size, seq_size)
            1 - means masked, 0 - unmasked
        returns:
            Y: the input of the shape (seq_len, batch, features)
        """
        Q, K, V = self.qkv(X)
        Y = self.atten(Q, K, V, mask=mask)
        Y = self.forw(Y)
        return Y

class CrossModalityBlockPart(nn.Module):
    """
    One part of Cross Modality Encoder
    """
    def __init__(self, input_size, nheads, hidden_size=None, dropout=0.1, activation=nn.ReLU(), *args, **kwargs) -> None:
        """
        Initialise params for two encoder blocks
        """
        super().__init__(*args, **kwargs)
        self.qkv1 = QKVBlock(input_size)
        self.atten1 = AttentionBlock(input_size, nheads, dropout=dropout)
        self.forw1 = ForwardBlock(input_size, hidden_size, activation=activation, dropout=dropout)
        self.qkv2 = QKVBlock(input_size)
        self.atten2 = AttentionBlock(input_size, nheads, dropout=dropout)
        self.forw2 = ForwardBlock(input_size, hidden_size, activation=activation, dropout=dropout)
    
    def forward(self, X1, X2, swap="n", mask1=None, mask2=None) -> Tuple[torch.Tensor]:
        """
        inputs:
            X1: the input for the first encoder of the shape (seq_len, batch, features)
            X2: the input for the second encoder of the shape (seq_len, batch, features)
            mask1: attention mask for the first encoder
            mask2: attention mask for the second encoder
            swap: ["q", "k", "v", "n"] this parameter determines what to swap in the two encoders
            q - queris, k - keys, v - values, n - none
        returns:
            Y1: the output of the first encoder of the shape (seq_len, batch, features)
            Y2: the output of the second encoder of the shape (seq_len, batch, features)
        """
        Q1, K1, V1 = self.qkv1(X1)
        Q2, K2, V2 = self.qkv1(X2)

        if swap == "q":
            H1 = self.atten1(Q2, K1, V1, mask=mask1)
            H2 = self.atten1(Q1, K2, V2, mask=mask2)
        elif swap == "k":
            H1 = self.atten1(Q1, K2, V1, mask=mask1)
            H2 = self.atten1(Q2, K1, V2, mask=mask2)
        elif swap == "v":
            H1 = self.atten1(Q1, K1, V2, mask=mask1)
            H2 = self.atten1(Q2, K2, V1, mask=mask2)
        elif swap == "n":
            H1 = self.atten1(Q1, K1, V1, mask=mask1)
            H2 = self.atten1(Q2, K2, V2, mask=mask2)
        else:
            raise Exception("wrong input for swap")
        
        Y1 = self.forw1(H1)
        Y2 = self.forw2(H2)
        return Y1, Y2


class CrossModalityBlock(nn.Module):
    """
    A single layer of Cross Modality Encoder
    """
    def __init__(self, input_size, nheads, hidden_size=None, dropout=0.1, activation=nn.ReLU(), *args, **kwargs) -> None:
        """
        Initialise four blocks for the layer
        """
        super().__init__(*args, **kwargs)
        self.block1 = CrossModalityBlockPart(input_size, nheads, hidden_size=hidden_size, dropout=dropout, activation=activation)
        self.block2 = CrossModalityBlockPart(input_size, nheads, hidden_size=hidden_size, dropout=dropout, activation=activation)
        self.block3 = CrossModalityBlockPart(input_size, nheads, hidden_size=hidden_size, dropout=dropout, activation=activation)
        self.block4 = CrossModalityBlockPart(input_size, nheads, hidden_size=hidden_size, dropout=dropout, activation=activation)
    
    def forward(self, X1, X2, mask1=None, mask2=None):
        """
        inputs:
            X1: the input for the first encoder of the shape (seq_len, batch, features)
            X2: the input for the second encoder of the shape (seq_len, batch, features)
            mask1: attention mask for the first encoder
            mask2: attention mask for the second encoder
        returns:
            Y1: the output of the first encoder of the shape (seq_len, batch, features)
            Y2: the output of the second encoder of the shape (seq_len, batch, features)
        """
        A1, A2 = self.block1(X1, X2, swap="k", mask1=mask1, mask2=mask2)
        B1, B2 = self.block2(A1, A2, swap="n", mask1=mask1, mask2=mask2)
        C1, C2 = self.block1(B1, B2, swap="v", mask1=mask1, mask2=mask2)
        Y1, Y2 = self.block1(C1, C2, swap="n", mask1=mask1, mask2=mask2)
        return Y1, Y2
