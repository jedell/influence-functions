import numpy as np
import scipy.sparse.linalg as L
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data as torchdata

from torch_influence.base import BaseInfluenceModule, BaseObjective
from torch_influence import AutogradInfluenceModule

class EKFACInfluenceModule(BaseInfluenceModule):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: torchdata.DataLoader,
            test_loader: torchdata.DataLoader,
            device: torch.device,
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

    def inverse_hvp(self, vec):

        return vec


class MyObjective(BaseObjective):
    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        return F.mse_loss(outputs, batch[1])

    def train_regularization(self, params):
        return 0


    def test_loss(self, model, params, batch):
        return F.mse_loss(model(batch[0]), batch[1])



def create_toy_training():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )
    objective = MyObjective()

    train_data = torch.randn(100, 10)
    train_labels = torch.randn(100, 10)
    train_dataset = torchdata.TensorDataset(train_data, train_labels)
    train_loader = torchdata.DataLoader(train_dataset, batch_size=10)

    test_data = torch.randn(20, 10)
    test_labels = torch.randn(20, 10)
    test_dataset = torchdata.TensorDataset(test_data, test_labels)
    test_loader = torchdata.DataLoader(test_dataset, batch_size=10)

    device = torch.device('cpu')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            model.zero_grad()

            outputs = model(inputs)
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    ekfac_module = AutogradInfluenceModule(
        model=model,
        objective=objective,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        damp=0.001
    )
    return ekfac_module

module = create_toy_training()

scores = module.influences([1,2,3], [0])

print(scores)