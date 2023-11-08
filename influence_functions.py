import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time
import datetime
import numpy as np
import copy
import logging
from transformers import PreTrainedModel

def influence_function(
        model: PreTrainedModel, train_dataloader: DataLoader, test_dataloader: DataLoader
    ):

    return

# https://arxiv.org/pdf/2308.03296.pdf Equation 15

def calc_layer_pseudo_gradient(a_l_m1, s_l, w_l, b_l):
    # Compute the gradient of the log probability with respect to the outputs of layer l
    s_l.requires_grad_(True)
    log_prob = torch.log(torch.softmax(s_l, dim=-1)) # TODO use softmax layer from model?
    pseudo_out_l = torch.autograd.grad(log_prob.sum(), s_l, retain_graph=True)[0]

    # Compute the pseudo-gradient for the weights of layer l
    pseudo_w_l = torch.kron(a_l_m1, pseudo_out_l)

    return pseudo_w_l

# https://github.com/tfjgeorge/nngeometry/blob/master/nngeometry/object/pspace.py#L8

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, src):
        output = self.relu(self.linear1(src))
        output = self.linear2(output)
        return output

small_model = SimpleLinearModel()

train_data = torch.randn((100, 100)) 
train_labels = torch.randint(0, 10, (100,))
train_dataset = TensorDataset(train_data, train_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

from nngeometry.object import PMatEKFAC, PMatDiag, PVector
from nngeometry.metrics import FIM

G = FIM(
    model=small_model,
    loader=train_dataloader,
    representation=PMatDiag,
    n_output=10,
    # variant='classif_logits',
)

# This is FIM approx.
print(G.get_dense_tensor().size())
print(G.get_diag().shape)

v = PVector.from_model(small_model)

# Compute matrix vector product for G and vector v
Fv = G.mv(v)

print(Fv.vector_repr.shape)
