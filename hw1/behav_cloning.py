import pickle
import torch
from torch import autograd
import torch.nn as nn
import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace
import random


class Data:
    def __init__(self, env, sequence_length, batch, layers):
        file = 'expert_data/' + env + '.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.inputs = data['observations']
        self.outputs = data['actions']
        self.sequence_length = sequence_length
        self.batch_size = batch
        self.input_size = data['observations'].shape[-1]
        self.hidden_size = data['actions'].shape[-1]
        self.num_layers = layers

    def format_sequences(self):
        x = []
        y = []
        for i in range(data.inputs.shape[0] - self.sequence_length - 1):
            _x = data.inputs[i:(i + self.sequence_length)]
            _y = data.outputs[i:(i + self.sequence_length), 0]
            x.append(_x)
            y.append(_y)
        self.inputs = torch.Tensor(x)
        self.outputs = torch.FloatTensor(y)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(data.input_size, data.input_size * 2)
        self.lin2 = nn.Linear(data.input_size * 2, data.input_size * 4)
        self.lin3 = nn.Linear(data.input_size * 4, data.input_size * 2)
        self.lin4 = nn.Linear(data.input_size * 2, data.input_size)
        self.lstm = nn.LSTM(input_size=data.input_size, hidden_size=data.hidden_size, num_layers=data.num_layers,
                            batch_first=True)

    def forward(self, x, hidden):
        x = x.view(data.batch_size, data.sequence_length, data.input_size)
        out = self.lin1(x)
        out = self.lin2(out)
        out = self.lin3(out)
        out = self.lin4(out)
        out, h = self.lstm(out, hidden)
        # out = out.view(-1, data.hidden_size)
        return h, out

    def init_hidden(self):
        return (torch.zeros(data.num_layers, data.batch_size, data.hidden_size),
                torch.zeros(data.num_layers, data.batch_size, data.hidden_size))


if __name__ == '__main__':
    scenario = 'Ant-v2'
    data = Data(scenario, 30, 32, 1)
    data.format_sequences()
    print(data.inputs.shape)
    model = Model()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []
    model.train()
    number_of_batches = 10
    for epoch in range(50):
        for i in range(number_of_batches):
            optimizer.zero_grad()
            hidden = model.init_hidden()
            batch_idx = random.sample(range(0, data.inputs.shape[0] - 1), data.batch_size)
            batch = data.inputs[batch_idx]
            hidden, output = model(batch, hidden)
            loss = (output[-1][-1] - data.outputs[batch_idx]).pow(2).sum()
            loss_list.append(loss.item())
            print(epoch, i, loss.item() / data.hidden_size / data.sequence_length)
            loss.backward()
            optimizer.step()

    plt.plot(loss_list)
    plt.show()
    torch.save(model, scenario + '.pt')
