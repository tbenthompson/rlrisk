import os

import cloudpickle
import numpy as np
import torch
import torch.nn
import torch.optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from . import env


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

    def __init__(
        self, pi_lr=0.01, hidden_layers=[64, 64], attack_from=True, no_writer=False
    ):
        self.attack_from = attack_from
        self.epoch = -1
        if not no_writer:
            self.writer = SummaryWriter()

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
        if not self.attack_from:
            state_matrix = env.states_to_territory_matrix(states)
            attack[:, 0] = (state_matrix[:, :, player_idx + 1] == 1).argmax(axis=1)
        return attack

    def reset_recording(self):
        self.epoch_data = dict()
        self.epoch_data["observations"] = []
        self.epoch_data["actions"] = []
        self.epoch_data["returns_to_go"] = []
        self.epoch_data["lengths"] = []
        self.epoch_data["did_i_win"] = []
        self.epoch += 1

    def record_games(self, my_turn, obs, acts, game_not_over, did_i_win):
        n_games = obs.shape[1]
        for j in range(n_games):
            this_game_obs = obs[my_turn[:, j] & game_not_over[:, j], j]
            this_game_action = acts[my_turn[:, j] & game_not_over[:, j], j]
            for i in range(this_game_obs.shape[0]):
                self.epoch_data["observations"].append(this_game_obs[i])
                self.epoch_data["actions"].append(this_game_action[i, :])
                self.epoch_data["returns_to_go"].append(did_i_win[j])
            self.epoch_data["lengths"].append(this_game_obs.shape[0])
        self.epoch_data["did_i_win"].extend(did_i_win)

    def save_model(self, report, name):
        dir = self.writer.log_dir
        save_dict = dict(
            epoch=self.epoch,
            model_state_dict=self.pi_net.state_dict(),
            full_obj=cloudpickle.dumps(
                {
                    k: v
                    for k, v in self.__dict__.items()
                    if k not in ["writer", "epoch_data"]
                }
            ),
        )
        save_dict.update(report)
        torch.save(save_dict, os.path.join(dir, f"{name}.pkl"))

    def finish_epoch(self, report):
        for k, v in report.items():
            self.writer.add_scalar(k, v, self.epoch)
            print(f" {k}: {v:.3f}", end="")
        print("")
        self.save_model(report, self.epoch)
        self.reset_recording()

    def learn(self):
        if len(self.epoch_data["observations"]) == 0:
            return None

        self.pi_optimizer.zero_grad()
        loss = self.compute_loss(
            obs=torch.as_tensor(self.epoch_data["observations"], dtype=torch.float32),
            act=torch.as_tensor(self.epoch_data["actions"], dtype=torch.int32),
            weights=torch.as_tensor(
                self.epoch_data["returns_to_go"], dtype=torch.float32
            ),
        )
        loss.backward()
        self.pi_optimizer.step()

        win_percentage = np.mean(self.epoch_data["did_i_win"])
        game_length = np.mean(self.epoch_data["lengths"])
        report = {
            "Win percentage": win_percentage,
            "Average game length": game_length,
            "Loss": loss,
        }
        self.finish_epoch(report)
        return None
