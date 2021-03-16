import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import env

torch.manual_seed(0)


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class SimplePolicyPlayer:
    def __init__(self, pi_lr=0.01):
        self.state_dim = env.state_dim
        self.n_actions = [env.n_max_territories] * 2
        self.pi_lr = pi_lr

        self.pi_net = mlp(sizes=[self.state_dim] + [64, 64] + [sum(self.n_actions)])
        self.pi_optimizer = Adam(self.pi_net.parameters(), lr=self.pi_lr)

    def get_policy(self, obs):
        logits = self.pi_net(obs)
        act1 = Categorical(logits=logits[:, : self.n_actions[0]])
        act2 = Categorical(logits=logits[:, self.n_actions[0] :])
        return act1, act2

    def get_action(self, obs):
        return torch.vstack([c.sample() for c in self.get_policy(obs)]).T.cpu().numpy()

    def compute_loss(self, obs, act, weights):
        logp = sum([self.get_policy(obs)[i].log_prob(act[:, i]) for i in range(2)])
        return -(logp * weights).mean()

    def act(self, player_idx, states):
        if states.shape[0] == 0:
            return np.array([]).reshape((0, 2))
        # state_matrix = env.states_to_territory_matrix(states)
        # attack_from = (state_matrix[:, :, player_idx + 1] == 1).argmax(axis=1)
        #
        states_torch = torch.as_tensor(states, dtype=torch.float32)
        attack = self.get_action(states_torch)
        return attack

    def learn(self, my_turn, obs, acts, game_not_over, rewards):
        observations = []
        actions = []
        weights = []
        returns = []
        lengths = []

        n_games = obs.shape[1]
        for j in range(n_games):
            this_game_obs = obs[my_turn[:, j] & game_not_over[:, j], j]
            this_game_action = acts[my_turn[:, j] & game_not_over[:, j], j]
            for i in range(this_game_obs.shape[0]):
                observations.append(this_game_obs[i])
                actions.append(this_game_action[i, :])
                weights.append(rewards[j])
            returns.append(rewards[j])
            lengths.append(this_game_obs.shape[0])

        loss = 0
        if len(observations) != 0:
            self.pi_optimizer.zero_grad()
            loss = self.compute_loss(
                obs=torch.as_tensor(observations, dtype=torch.float32),
                act=torch.as_tensor(actions, dtype=torch.int32),
                weights=torch.as_tensor(weights, dtype=torch.float32),
            )
            loss.backward()
            self.pi_optimizer.step()

        win_percentage = np.mean(returns)
        game_length = np.mean(lengths)
        print(
            f" loss: {loss:.3f}"
            f" return: {win_percentage:.3f} ep_len: {game_length:.3f} "
            f" n_games: {len(returns)}"
        )
        return None


class VPGPlayer(SimplePolicyPlayer):
    def __init__(self, pi_lr=0.001, vf_lr=0.001, train_v_iters=80):
        super().__init__(pi_lr)

        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.vf_net = mlp(sizes=[self.state_dim] + [64, 64] + [1])
        self.vf_optimizer = Adam(self.vf_net.parameters(), lr=self.vf_lr)

    def get_value(self, obs):
        return torch.squeeze(self.vf_net(obs), -1)

    # Set up function for computing value loss
    def compute_value_loss(self, obs, returns):
        return ((self.get_value(obs) - returns) ** 2).mean()

    def learn(self, my_turn, obs, acts, game_not_over, did_i_win):
        observations = []
        actions = []
        returns_to_go = []
        lengths = []

        n_games = obs.shape[1]
        for j in range(n_games):
            this_game_obs = obs[my_turn[:, j] & game_not_over[:, j], j]
            this_game_action = acts[my_turn[:, j] & game_not_over[:, j], j]
            for i in range(this_game_obs.shape[0]):
                observations.append(this_game_obs[i])
                actions.append(this_game_action[i, :])
                returns_to_go.append(did_i_win[j])
            lengths.append(this_game_obs.shape[0])

        torch_observations = torch.as_tensor(observations, dtype=torch.float32)
        value_estimates = self.get_value(torch_observations)
        torch_returns_to_go = torch.as_tensor(returns_to_go, dtype=torch.float32)

        loss_pi = 0
        loss_vf = 0
        if len(observations) != 0:
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss(
                obs=torch_observations,
                act=torch.as_tensor(actions, dtype=torch.int32),
                weights=torch_returns_to_go - value_estimates
            )
            loss_pi.backward()
            self.pi_optimizer.step()

            for i in range(self.train_v_iters):
                self.vf_optimizer.zero_grad()
                loss_vf = self.compute_value_loss(
                    torch_observations, torch_returns_to_go
                )
                loss_vf.backward()
                self.vf_optimizer.step()

        win_percentage = np.mean(did_i_win)
        game_length = np.mean(lengths)
        print(
            f" loss_pi: {loss_pi:.3f} loss_vf: {loss_vf:.3f}"
            f" return: {win_percentage:.3f} ep_len: {game_length:.3f} "
            f" n_games: {len(did_i_win)}"
        )
        return None


next_seed = 0


def train_one_epoch(spec, players, batch_size):
    global next_seed
    seeds = np.arange(next_seed, next_seed + batch_size)
    next_seed += batch_size
    winners, final_state, state_history = env.play_games(
        spec, players, seeds, record=True
    )

    game_not_over = ~np.array([data[0] for data in state_history])
    player_idxs = np.array([data[1] for data in state_history])
    obs = np.array([data[2] for data in state_history])
    acts = np.array([data[3] for data in state_history])

    out = []
    for i, p in enumerate(players):
        rewards = (winners == i).astype(np.float32)
        my_turn = player_idxs[:, :] == i
        print(f"player {i} ", end="")
        out.append(p.learn(my_turn, obs, acts, game_not_over, rewards))
    return out


def train(spec, players, n_batches, batch_size, print_players):
    results = []
    for i in range(n_batches):
        print("epoch", i)
        results.append(train_one_epoch(spec, players, batch_size))
