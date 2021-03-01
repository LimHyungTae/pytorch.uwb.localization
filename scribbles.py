import torch
import torch.nn as nn
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
m = nn.Softmax(dim=1)
input = m(input)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input)
print(target)
print(input.size(), target.size())
output = loss(input, target)
output.backward()