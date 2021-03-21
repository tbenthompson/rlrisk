import numpy as np
import torch
import torch.nn
import torch.optim
from torch.distributions.categorical import Categorical

from . import batch, env


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

    def __init__(self, pi_lr=0.01, hidden_layers=[64, 64], attack_from=True):
        self.batch_type = batch.Batch
        self.attack_from = attack_from

        self.state_dim = env.state_dim
        self.n_actions = [env.n_max_territories] * (1 + attack_from)
        self.pi_lr = pi_lr

        self.pi_net = mlp(
            sizes=[self.state_dim] + hidden_layers + [sum(self.n_actions)]
        )
        self.pi_optimizer = torch.optim.Adam(self.pi_net.parameters(), lr=self.pi_lr)

    def get_policy(self, obs):
        logits = self.pi_net(obs)
        act1 = Categorical(logits=logits[:, : self.n_actions[0]])
        if self.attack_from:
            act2 = Categorical(logits=logits[:, self.n_actions[0] :])
            return act1, act2
        else:
            return (act1,)

    def get_action(self, obs):
        return torch.vstack([c.sample() for c in self.get_policy(obs)]).T.cpu().numpy()

    def compute_loss(self, obs, act, weights):
        logp = sum(
            [
                self.get_policy(obs)[i].log_prob(act[:, i])
                for i in range(len(self.n_actions))
            ]
        )
        return -(logp * weights).mean()

    def act(self, player_idx, states):
        if states.shape[0] == 0:
            return np.array([]).reshape((0, 2))
        states_torch = torch.as_tensor(states, dtype=torch.float32)
        if not self.attack_from:
            state_matrix = env.states_to_territory_matrix(states)
            attack = np.empty((states.shape[0], 2))
            attack[:, 1:] = self.get_action(states_torch)
            attack[:, 0] = (state_matrix[:, :, player_idx + 1] == 1).argmax(axis=1)
            return attack
        else:
            return self.get_action(states_torch)

    def learn(self, batch):
        if len(batch.observations) == 0:
            return None

        actions = torch.as_tensor(batch.actions, dtype=torch.int32)
        if not self.attack_from:
            actions = actions[:, 1:]

        self.pi_optimizer.zero_grad()
        loss = self.compute_loss(
            obs=torch.as_tensor(batch.observations, dtype=torch.float32),
            act=actions,
            weights=torch.as_tensor(batch.returns_to_go, dtype=torch.float32),
        )
        loss.backward()
        self.pi_optimizer.step()

        win_percentage = np.mean(batch.did_i_win)
        game_length = np.mean(batch.lengths)
        report = {
            "Win percentage": win_percentage,
            "Average game length": game_length,
            "Loss": loss,
        }
        return report
