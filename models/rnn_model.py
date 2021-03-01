import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

OUTPUT_LENGTH = 2 # x and y

class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, Y_target="end", model_type="lstm"):
        super(Model, self).__init__()
        self.Y_target = Y_target

        if model_type == "lstm":
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        elif model_type == "rnn":
            self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True)
        elif model_type == "gru":
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)

        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim * 2, bias=True)
        self.bn = torch.nn.BatchNorm1d(hidden_dim * 2)
        self.relu = torch.nn.ReLU(inplace=False)
        self.fc2 = torch.nn.Linear(hidden_dim * 2, OUTPUT_LENGTH, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        if self.Y_target == "end":
            x = x[:, -1]
            x = self.relu(x)
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.fc2(x)
        elif self.Y_target == "all":
            x = self.relu(x)
            bs, seq, hs = x.size()
            x = x.reshape(bs * seq, hs)
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = x.view(bs, seq, OUTPUT_LENGTH)
        else:
            raise RuntimeError("Not implemented!!")

        return x

if __name__ == "__main__":
    bs = 256
    seq_len = 12
    input_size = 8
    hs = 128
    lstm = Model(input_size, hs, "all")
    inputs = torch.randn(bs, seq_len, input_size)  # make a sequence of length 5
    #
    print(inputs.size())
    out = lstm(inputs)
    print(out.size())

