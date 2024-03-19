import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print(device)


class LSTM_Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional, sequence_length):
        super(LSTM_Attention, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional).to(device)

        self.embed_dim = input_size
        self.sequence_length = sequence_length
        self.attention_size = 10
        self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_layers, self.attention_size)).to(device)
        # （30）
        self.u_omega = Variable(torch.zeros(self.attention_size)).to(device)

        self.fc = nn.Linear(self.num_directions * hidden_size, num_classes).to(device)

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*num_layers)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.num_layers])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*num_layers)
        # tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        # 张量相乘
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output
        # print(state.size()) = (batch_size, squence_length, hidden_size*num_layers)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*num_layers)

        return attn_output

    def forward(self, input):
        # input = self.lookup_table(input_sentences)
        # input = input.permute(1, 0, 2)
        # print('input.size():',input.size())
        s, b, f = input.size()
        h_0 = Variable(torch.zeros(self.num_layers, s, self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers, s, self.hidden_size)).to(device)
        # print('input.size(),h_0.size(),c_0.size()', input.size(), h_0.size(), c_0.size())
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)
        logits = self.fc(attn_output)
        return logits
