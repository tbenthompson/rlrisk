import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import env

torch.manual_seed(0)


class VPGPlayer:
    def __init__(self, lr=0.01):
        self.state_dim = env.state_dim
        self.n_actions = env.n_max_territories
        self.lr = lr

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
        return self.get_policy(obs).sample().numpy()

    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def act(self, player_idx, states):
        if states.shape[0] == 0:
            return np.array([]).reshape((0, 2))
        state_matrix = env.states_to_territory_matrix(states)
        attack_from = (state_matrix[:, :, player_idx + 1] == 1).argmax(axis=1)

        states_torch = torch.as_tensor(states, dtype=torch.float32)
        attack_to = self.get_action(states_torch)

        return np.array([attack_from, attack_to]).T

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


class Batch:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.weights = []
        self.returns = []
        self.lengths = []

    def record_game(self, obs, action, rew):
        for i in range(obs.shape[0]):
            self.obs.append(obs[i])
            self.actions.append(action[i, 1])
            self.weights.append(rew)
        self.returns.append(rew)
        self.lengths.append(obs.shape[0])


def train_one_epoch(spec, players, batch_size):
    batches = [Batch() for p in players]
    winners, final_state, state_history = env.play_games(
        spec, players, np.random.randint(1e8, size=batch_size), record=True
    )

    rewards = np.array([(winners == i).astype(np.float32) for i in range(len(players))])
    player_idxs = np.array([data[1] for data in state_history])
    obs = np.array([data[2] for data in state_history])
    acts = np.array([data[3] for data in state_history])

    for i in range(len(players)):
        for j in range(batch_size):
            batches[i].record_game(
                obs[player_idxs[:, j] == i, j],
                acts[player_idxs[:, j] == i, j],
                rewards[i, j],
            )

    loss = []
    for i in range(len(players)):
        if len(batches[i].obs) == 0:
            loss.append(0)
        else:
            loss.append(
                players[i].learn(batches[i].obs, batches[i].actions, batches[i].weights)
            )
    return loss, [b.returns for b in batches], [b.lengths for b in batches], batches


def train(spec, players, n_batches, batch_size, print_players):
    for i in range(n_batches):
        loss, rets, lens, _ = train_one_epoch(spec, players, batch_size)
        for j in print_players:
            win_percentage = np.mean(rets[j])
            game_length = np.mean(lens[j])
            print(
                f"epoch: {i}, player: {j}, loss: {loss[j]:.3f}"
                f" return: {win_percentage:.3f} ep_len: {game_length:.3f} "
                f" n_games: {len(rets[j])}"
            )
