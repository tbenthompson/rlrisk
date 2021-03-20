import env
import numpy as np
import torch
import torch.nn
import torch.optim
from torch.distributions.categorical import Categorical


def mlp(sizes, activation=torch.nn.Tanh, output_activation=torch.nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)


class SimplePolicyPlayer:
    """
    A simple player with just a policy function that is optimized via policy
    gradient.
    """

    def __init__(self, pi_lr=0.01, hidden_layers=[64, 64]):
        self.state_dim = env.state_dim
        self.n_actions = [env.n_max_territories] * 2
        self.pi_lr = pi_lr

        self.pi_net = mlp(
            sizes=[self.state_dim] + hidden_layers + [sum(self.n_actions)]
        )
        self.pi_optimizer = torch.optim.Adam(self.pi_net.parameters(), lr=self.pi_lr)
        self.reset_recording()

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
        states_torch = torch.as_tensor(states, dtype=torch.float32)
        attack = self.get_action(states_torch)
        return attack

    def reset_recording(self):
        self.observations = []
        self.actions = []
        self.returns_to_go = []
        self.lengths = []
        self.did_i_win = []

    def record_games(self, my_turn, obs, acts, game_not_over, did_i_win):
        n_games = obs.shape[1]
        for j in range(n_games):
            this_game_obs = obs[my_turn[:, j] & game_not_over[:, j], j]
            this_game_action = acts[my_turn[:, j] & game_not_over[:, j], j]
            for i in range(this_game_obs.shape[0]):
                self.observations.append(this_game_obs[i])
                self.actions.append(this_game_action[i, :])
                self.returns_to_go.append(did_i_win[j])
            self.lengths.append(this_game_obs.shape[0])
        self.did_i_win.extend(did_i_win)

    def learn(self):
        if len(self.observations) == 0:
            return None

        self.pi_optimizer.zero_grad()
        loss = self.compute_loss(
            obs=torch.as_tensor(self.observations, dtype=torch.float32),
            act=torch.as_tensor(self.actions, dtype=torch.int32),
            weights=torch.as_tensor(self.returns_to_go, dtype=torch.float32),
        )
        loss.backward()
        self.pi_optimizer.step()

        win_percentage = np.mean(self.did_i_win)
        game_length = np.mean(self.lengths)
        print(
            f" loss: {loss:.3f}"
            f" return: {win_percentage:.3f} ep_len: {game_length:.3f} "
            f" n_games: {len(self.did_i_win)}"
        )
        self.reset_recording()
        return None
