import numpy as np

import rlrisk.env

# TODO: Build a "Trainer" module that pairs up varying levels/types of players.
next_seed = 0


def train_one_epoch(spec, players, batch_size):
    global next_seed
    seeds = np.arange(next_seed, next_seed + batch_size)
    next_seed += batch_size
    winners, final_state, state_history = rlrisk.env.play_games(
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
        p.record_games(my_turn, obs, acts, game_not_over, rewards)

    for i, p in enumerate(players):
        print(f"player {i} ", end="")
        out.append(p.learn())
    return out


def train(spec, players, n_batches, batch_size, print_players):
    results = []
    for i in range(n_batches):
        print("epoch", i)
        results.append(train_one_epoch(spec, players, batch_size))
