import numpy as np
import torch
import torch.nn
import torch.optim

import rlrisk.env
import rlrisk.simple_policy

torch.manual_seed(0)


class VPGPlayer(rlrisk.simple_policy.SimplePolicyPlayer):
    """
    Add a value function to the SimplePolicyPlayer -- a VPG implementation.
    Use lambda-GAE for estimating the advantage.
    """

    def __init__(self, pi_lr=0.001, vf_lr=0.001, train_v_iters=80):
        super().__init__(pi_lr)

        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.vf_net = rlrisk.simple_policy.mlp(sizes=[self.state_dim] + [64, 64] + [1])
        self.vf_optimizer = torch.optim.Adam(self.vf_net.parameters(), lr=self.vf_lr)

    def get_value(self, obs):
        return torch.squeeze(self.vf_net(obs), -1)

    def compute_value_loss(self, obs, returns):
        return ((self.get_value(obs) - returns) ** 2).mean()

    def learn(self):
        if len(self.observations) == 0:
            return None

        torch_observations = torch.as_tensor(self.observations, dtype=torch.float32)
        value_estimates = self.get_value(torch_observations)
        torch_returns_to_go = torch.as_tensor(self.returns_to_go, dtype=torch.float32)

        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss(
            obs=torch_observations,
            act=torch.as_tensor(self.actions, dtype=torch.int32),
            weights=torch_returns_to_go - value_estimates,
        )
        loss_pi.backward()
        self.pi_optimizer.step()

        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_vf = self.compute_value_loss(torch_observations, torch_returns_to_go)
            loss_vf.backward()
            self.vf_optimizer.step()

        win_percentage = np.mean(self.did_i_win)
        game_length = np.mean(self.lengths)
        print(
            f" loss_pi: {loss_pi:.3f} loss_vf: {loss_vf:.3f}"
            f" return: {win_percentage:.3f} ep_len: {game_length:.3f} "
            f" n_games: {len(self.did_i_win)}"
        )
        self.reset_recording()
        return None
