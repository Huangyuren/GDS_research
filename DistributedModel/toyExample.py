import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        nvtx.range_push("Toymodel_layer_stack")
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(100, 100).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(100, 50).to('cpu')
        nvtx.range_pop()

    def forward(self, x):
        nvtx.range_push("net1")
        x1 = self.net1(x)
        nvtx.range_pop()
        nvtx.range_push("relu1")
        x2 = self.relu(x1) 
        nvtx.range_pop()
        nvtx.range_push("Copy to cpu")
        x2 = x2.to('cpu')
        nvtx.range_pop()
        nvtx.range_push("net2")
        x3 = self.net2(x2)
        # x = self.relu(self.net1(x))
        nvtx.range_pop()
        # return self.net2(x.to('cpu'))
        return x3

nvtx.range_push("One step pass")
model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
nvtx.range_push("Copy to device")
inputs = torch.randn(200,100).to('cuda:0')
nvtx.range_pop()
outputs = model(inputs)
labels = torch.randn(200, 50).to('cpu')
nvtx.range_push("backward pass")
loss_fn(outputs, labels).backward()
nvtx.range_pop()
optimizer.step()
nvtx.range_pop()
