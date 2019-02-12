import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

import load_policy
import tf_util


class Data:
    def __init__(self, env, batch):
        file = 'expert_data/' + env + '.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        x = data['observations']

        y = data['actions']
        n = len(data['observations'])
        train_percentage = .2
        self.hidden_size = data['actions'].shape[-1]
        self.input_size = data['observations'].shape[-1]
        self.xtrain = x[:int(n * train_percentage)]
        self.ytrain = y[:int(n * train_percentage), 0]
        self.xtest = x[int(n * train_percentage):]
        self.ytest = y[int(n * train_percentage):, 0]
        self.batch = batch
        self.train = None
        self.test = None
        self.train_set = None
        self.test_set = None
        self.format_data()

    def shuffle(self):
        self.train_set = data_utils.DataLoader(self.train, batch_size=self.batch, shuffle=True)
        self.test_set = data_utils.DataLoader(self.test, batch_size=self.batch, shuffle=True)

    def add_data(self, obs, acs):
        print(self.ytrain.shape, acs.shape)
        self.xtrain = np.append(self.xtrain, obs, axis=0)
        print(self.xtrain.shape)
        self.ytrain = np.append(self.ytrain, acs[:,0], axis=0)
        self.format_data()

    def format_data(self):
        self.train = data_utils.TensorDataset(torch.Tensor(self.xtrain), torch.Tensor(self.ytrain))
        self.test = data_utils.TensorDataset(torch.Tensor(self.xtest), torch.Tensor(self.ytest))
        self.train_set = data_utils.DataLoader(self.train, batch_size=self.batch, shuffle=True)
        self.test_set = data_utils.DataLoader(self.test, batch_size=1, shuffle=True)


class Expert:
    def __init__(self, policy_file):
        print('loading and building expert policy')
        self.policy = load_policy.load_policy(policy_file)
        print('loaded and built')

    def act(self, obs):
        with tf.Session():
            tf_util.initialize()
            return self.policy(obs[None, :])


class Model(nn.Sequential):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(data.input_size, 64)
        self.lin2 = nn.Linear(64, data.hidden_size)
        self.loss_fn = nn.MSELoss(size_average=False)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def learn(self, epochs):
        loss_list = []
        model.train()
        for epoch in range(epochs):
            data.shuffle()
            for i, (X, y) in enumerate(data.train_set):
                output = model(X)
                loss = self.loss_fn(output, y)
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print(epoch, i + 1)
        return loss_list

    def evaluate(self, envname, max_timesteps, num_rollouts, render=False):
        model.eval()
        with tf.Session():
            tf_util.initialize()
            env = gym.make(envname)
            max_steps = max_timesteps or env.spec.timestep_limit
            actions = []
            observations = []
            reward = []
            for i in range(num_rollouts):
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0

                while not done:
                    action = model(torch.Tensor(obs))
                    action = action.detach().numpy()
                    actions.append(np.squeeze(action))
                    obs, r, done, _ = env.step(action)
                    observations.append(obs)
                    totalr += r
                    steps += 1

                    if render:
                        env.render()
                    if steps >= max_steps:
                        break

                reward.append(totalr)
        print(np.mean(reward), np.std(reward))
        return actions, observations, np.mean(reward)


if __name__ == '__main__':
    scenario = 'Walker2d-v2'
    expert = Expert("experts/" + scenario + ".pkl")
    data = Data(scenario, 64)
    figure = plt.figure()
    # Run model to get new obs
    rewards = []
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for dag in range(1000):
        loss = model.learn(1)
        actions, observations, reward = model.evaluate(scenario, 1000, 20, render=False)
        rewards.append(reward)
        policy_actions = []
        for i, obs in enumerate(observations):
            policy_act = expert.act(obs)
            policy_actions.append(policy_act)
            if (i + 1) % 100 == 0:
                print(dag, i + 1, len(observations))
        data.add_data(np.asarray(observations), np.asarray(policy_actions))

        plt.clf()
        plt.plot(rewards)
        plt.pause(0.05)
