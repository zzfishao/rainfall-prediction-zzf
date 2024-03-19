import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print(device)

class CNN_LSTM_PRO(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, bidirectional):
        super(CNN_LSTM_PRO, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = output_size
        self.num_directions = 2 if bidirectional else 1
        self.relu = nn.ReLU(inplace=True).to(device)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional).to(device)
        self.fc = nn.Linear(self.num_directions * hidden_size, output_size).to(device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
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
