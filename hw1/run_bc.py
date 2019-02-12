import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import torch
import pickle
from torch import autograd
import torch.nn as nn

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
        self.lin1 = nn.Linear(data.input_size, int(data.input_size))
        self.lin9 = nn.Linear(data.hidden_size, int(data.hidden_size))
        # self.lin2 = nn.Linear(data.input_size * 2, data.input_size * 4)
        # self.lin3 = nn.Linear(data.input_size * 4, data.input_size * 2)
        # self.lin4 = nn.Linear(data.input_size * 2, data.input_size)
        self.lstm = nn.LSTM(input_size=int(data.input_size), hidden_size=data.hidden_size, num_layers=data.num_layers,
                            batch_first=True)

    def forward(self, x, hidden):
        x = x.view(data.batch_size, data.sequence_length, data.input_size)
        out = self.lin1(x)
        # out = self.lin2(out)
        # out = self.lin3(out)
        # out = self.lin4(out)
        out, h = self.lstm(out, hidden)
        out = self.lin9(out)
        # out = out.view(-1, data.hidden_size)
        return h, out

    def init_hidden(self):
        return (torch.zeros(data.num_layers, data.batch_size, data.hidden_size),
                torch.zeros(data.num_layers, data.batch_size, data.hidden_size))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    model = torch.load(args.envname + ".pt")
    model.eval()
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            for s in range(data.sequence_length):
                _list = []
                for p in range(data.input_size):
                    _list.append(0)
                observations.append(_list)
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            hidden = model.init_hidden()
            while not done:
                hidden, action = model(torch.Tensor(observations[-data.sequence_length:]), hidden)
                observations.append(obs)
                actions.append(action.detach().numpy()[-1][-1])
                obs, r, done, _ = env.step(action.detach().numpy()[-1][-1])
                print(done)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    scenario = 'Reacher-v2'
    data = Data(scenario, 10, 1, 1)
    main()
