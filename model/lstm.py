import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional).to(device)
        self.fc = nn.Linear(self.num_directions * hidden_size, num_classes).to(device)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_directions *
                                   self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_directions *
                                   self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0.to(device), c_0.to(device)))
        pred = self.fc(ula)
        pred = pred[:, -1, :]
        # h_out = h_out.view(-1, self.hidden_size)
        # out = self.fc(h_out)
        return pred
