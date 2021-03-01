import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class StackedBiLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, Y_target="end"):
        super(StackedBiLSTM, self).__init__()
        # Input of LSTM: [bs, seq_len, Input_size]
        # Output of LSTM: [bs, seq_len, hidden_size]
        # Output of Hidden size LSTM: [num_layers, bs, hidden size]
        self.Y_target = Y_target

        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        # self.bn = nn.BatchNorm2d(hidden_dim)
        # self.fc1 = torch.nn.Linear(hidden_dim, 128, bias=True)
        # self.fc2 = torch.nn.Linear(128, output_dim, bias=True)

    def forward(self, x):
        # print(x.size())
        x, _status = self.lstm(x)
        # x = self.bn(x)
        # x = self.fc1(x[:, -1])
        # x = self.fc2(x[:, -1])
        print(x.size())
        if self.Y_target == "end":
            x = x[:, -1]
        return x

if __name__ == "__main__":
    bs = 8
    seq_len = 7
    input_size = 3
    hs = 128
    lstm = StackedBiLSTM(input_size, hs, 1)
    inputs = torch.randn(bs, seq_len, input_size)  # make a sequence of length 5
    #
    print(inputs.size())
    out = lstm(inputs)
    print(out.size())
    print(out[:,-1].size())
    print(out.size())
    # lstm = Net(input_size, hs, 1)
    # out, _state = lstm(inputs)

