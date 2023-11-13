import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import datetime
import numpy as np
import copy
import logging
from transformers import PreTrainedModel
from torch.autograd.functional import hessian
from torch.autograd import grad

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

from dataset import train_dataset, test_dataset, tokenizer

EMBEDDING_DIMENSION = 512

from model import GPT, GPTConfig

config = GPTConfig(n_layer=1, n_head=1, n_embd=8)

small_model = GPT(config=config)

# train_data = torch.randn((100, 100)) 
# train_labels = torch.randint(0, 10, (100,))
# train_dataset = TensorDataset(train_data, train_labels)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

loss_fn = F.cross_entropy

# hessians = []

# for X, Y in train_dataloader:
#     x_in = X['input_ids'].squeeze(0)
#     y_in = Y['input_ids'].squeeze(0)
#     logits, loss = small_model(x_in, y_in)
#     print(loss)
#     grads = grad(loss, small_model.parameters(), create_graph=True)

#     h = []
#     for g in grads:
#         g = g.sum()  # Convert to scalar by summing all elements of the gradient
#         h.append(grad(g, small_model.parameters(), retain_graph=True))


#     hessians.append(h)

# print(print(hessians[0]))
# # Compute the sum of Hessians element-wise
# sum_hessians = [sum(h) for h in zip(*hessians)]

# # Compute the average Hessian
# average_hessian = [h/len(hessians) for h in sum_hessians]

# H = torch.stack(average_hessian)

# print(H.shape)

from nngeometry.object import PMatEKFAC, PMatDiag, PVector
from nngeometry.metrics import FIM

G = FIM(
    model=small_model,
    loader=train_dataloader,
    representation=PMatEKFAC,
    n_output=10,
)

# This is FIM approx.
print(G.get_dense_tensor().size())

test_point = torch.randn((1, 100)) 
test_label = torch.randint(0, 10, (1,))

small_model.zero_grad()
output = small_model(test_point)
loss = loss_fn(output, test_label)
loss.backward()


grad_test_point = []
for param in small_model.parameters():
    # flatten
    grad_test_point.append(param.grad.view(-1))
grad_test_point = torch.cat(grad_test_point)

print("shape:", grad_test_point.shape)

v_q = G.inverse().get_dense_tensor() @ grad_test_point

print(v_q)

influence_values = []
for data, label in train_dataloader:
    small_model.zero_grad()
    output = small_model(data)
    loss = F.cross_entropy(output, label)
    loss.backward()

    grad_train_point = []
    for param in small_model.parameters():
        # flatten
        grad_train_point.append(param.grad.view(-1))
    grad_train_point = torch.cat(grad_train_point)

    influence_values.append(torch.dot(grad_train_point, v_q).item())

print(len(influence_values))