import os

import cloudpickle
import numpy as np
import torch

import rlrisk.env

torch.manual_seed(0)

# TODO: Build a "Trainer" module that pairs up varying levels/types of players.
next_seed = 0


class Trainer:
    def __init__(self, seed=0, writer=False):
        self.next_seed = seed

        self.writer = None
        if writer:
            # Importing this creates a bunch of ugly warnings from tensorboard,
            # let's wait until we actually need it.
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter()

        self.epoch = 0

    def get_seeds(self, n_seeds):
        out = np.arange(self.next_seed, self.next_seed + n_seeds)
        self.next_seed += n_seeds
        return out

    def train(self, spec, players, n_batches, batch_size):
        for i in range(n_batches):
            batches = [p.batch_type() for p in players]

            print(f"epoch={i} n_games={batch_size}")
            seeds = self.get_seeds(batch_size)

            winners, final_state, state_history = rlrisk.env.play_games(
                spec, players, seeds, record=True
            )

            game_over, player_idxs, obs, acts = state_history
            game_not_over = ~game_over

            for j, p in enumerate(players):
                if batches[j] is None:
                    continue
                batches[j].record_games(
                    j, player_idxs, obs, acts, game_not_over, winners
                )

            for j, p in enumerate(players):
                print(f"player {j} ", end="")
                report = p.learn(batches[j])
                for k, v in report.items():
                    if self.writer:
                        self.writer.add_scalar(k, v, self.epoch)
                    print(f" {k}: {v:.3f}", end="")

            if self.writer:
                dir = self.writer.log_dir
                torch.save(
                    cloudpickle.dumps(players), os.path.join(dir, f"{self.epoch}.pkl")
                )

            self.epoch += 1
