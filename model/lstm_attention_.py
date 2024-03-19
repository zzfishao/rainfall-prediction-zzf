import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print(device)

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state, encoder_outputs):
        hidden_state = hidden_state.repeat(encoder_outputs.size(0), 1, 1)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden_state))
        attention = self.V(energy)
        attention = attention.squeeze(2)
        attention = torch.softmax(attention, dim=1)
        attention = attention.unsqueeze(2)
        weighted = torch.mul(encoder_outputs, attention)
        context = weighted.sum(dim=0)
        return context


# class LSTMAttention(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMAttention, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.lstm = nn.LSTM(input_size, hidden_size)
#         self.attention = Attention(hidden_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, input, hidden_state, cell_state, mask):
#         input = pack_padded_sequence(input, mask, batch_first=True)
#         output, (hidden_state, cell_state) = self.lstm(input, (hidden_state, cell_state))
#         output, _ = pad_packed_sequence(output, batch_first=True)
#         context = self.attention(hidden_state, output)
#         output = self.fc(context)
#         # return output, hidden_state, cell_state
#         return output


class LSTMAttention(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMAttention, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True ).to(device)
        self.attention = Attention(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0.to(device), c_0.to(device)))
        # output, _ = pad_packed_sequence(ula, batch_first=True)
        context = self.attention(h_out, ula)
        pred = self.fc(context)
        pred = pred[:, -1, :]
        # h_out = h_out.view(-1, self.hidden_size)
        # out = self.fc(h_out)
        return pred
