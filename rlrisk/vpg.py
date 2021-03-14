import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from env import vec_to_territory_matrix

torch.manual_seed(0)

class Batch:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.weights = []
        self.returns = []
        self.lengths = []
        self.start_episode()

    def record(self, obs, action):
        self.obs.append(obs.copy())
        self.actions.append(action[1])
        self.ep_length += 1

    def finish_episode(self, reward):
        self.ep_returns = reward
        self.returns.append(self.ep_returns)
        self.lengths.append(self.ep_length)
        self.weights += [self.ep_returns] * self.ep_length
        self.start_episode()

    def start_episode(self):
        self.ep_length = 0
        self.ep_returns = 0


class VPGPlayer:
    def __init__(self, env):
        self.state_dim = env.get_state_dim()
        self.n_actions = env.get_action_dim()
        self.lr = 0.01

        def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
            # Build a feedforward neural network.
            layers = []
            for j in range(len(sizes) - 1):
                act = activation if j < len(sizes) - 2 else output_activation
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
            return nn.Sequential(*layers)

        self.logits_net = mlp(sizes=[self.state_dim] + [32] + [self.n_actions])
        self.optimizer = Adam(self.logits_net.parameters(), lr=self.lr)

    def get_policy(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def act(self, game, state_vec):
        owner_col = vec_to_territory_matrix(game, state_vec)[:, game.player_idx + 1]
        attack_from = (owner_col == 1).argmax()
        attack_to = self.get_action(torch.as_tensor(state_vec, dtype=torch.float32))
        return attack_from, attack_to

    def learn(self, obs, actions, weights):
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(
            obs=torch.as_tensor(obs, dtype=torch.float32),
            act=torch.as_tensor(actions, dtype=torch.int32),
            weights=torch.as_tensor(weights, dtype=torch.float32),
        )
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss


def train_one_epoch(env, players, batch_size):
    batches = [Batch() for p in players]
    obs = env.reset()

    go = True
    while go:
        # print(env.game.turn_idx, env.game.player_idx)
        action = players[env.game.player_idx].act(env.game, obs)
        batches[env.game.player_idx].record(obs, action)
        obs, reward, done = env.step(action)
        if done:
            for i in range(len(players)):
                player_reward = reward if i == env.game.player_idx else 0
                b = batches[i]
                b.finish_episode(player_reward)
                if len(b.obs) > batch_size:
                    go = False
            obs = env.reset()

    loss = []
    for i in range(len(players)):
        if len(batches[i].obs) == 0:
            loss.append(0)
        else:
            loss.append(
                players[i].learn(batches[i].obs, batches[i].actions, batches[i].weights)
            )
    return loss, [b.returns for b in batches], [b.lengths for b in batches]


def train(env, players, n_batches, batch_size, print_players):
    for i in range(n_batches):
        loss, rets, lens = train_one_epoch(env, players, batch_size)
        for j in print_players:
            win_percentage = np.mean(rets[j])
            game_length = np.mean(lens[j])
            print(
                f"epoch: {i}, player: {j}, loss: {loss[j]:.3f} return: {win_percentage:.3f} ep_len: {game_length:.3f}"
            )
