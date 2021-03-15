import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

import risk_ext

n_max_players = risk_ext.n_max_players()
n_max_territories = risk_ext.n_max_territories()
state_dim = risk_ext.state_dim()

def states_to_territory_matrix(states):
    return states[:, n_max_players:].reshape(
        (states.shape[0], n_max_territories, n_max_players + 1)
    )


def start_game(spec, seed):
    return risk_ext.start_game(
        spec["n_players"],
        spec["n_territories"],
        spec["baseline_reinforcements"],
        spec["n_attacks_per_turn"],
        0,
    )


def vis(game):
    v = game.board_state
    tm = vec_to_territory_matrix(game, v)
    armies = tm[:, 0]
    owners = (tm[:, 1:] == 1.0).argmax(axis=1)
    cmap = plt.get_cmap("coolwarm")
    m = matplotlib.cm.ScalarMappable(cmap=cmap)

    plt.bar(
        np.arange(armies.shape[0]),
        armies,
        color=m.to_rgba(owners / np.max(owners), norm=False),
        align="center",
    )
    plt.ylabel("number of armies")
    plt.xlabel("territory idx")
    plt.show()


def play_games(spec, players, seeds):
    n_games = len(seeds)
    game_set = risk_ext.start_game_set(
        spec["n_players"],
        spec["n_territories"],
        spec["baseline_reinforcements"],
        spec["n_attacks_per_turn"],
        seeds,
    )

    player_idxs = np.empty(n_games, dtype=np.int32)
    states = np.empty((n_games, state_dim), dtype=np.float32)
    while True:
        game_set.get_idxs_states(player_idxs, states)

        actions = np.zeros((n_games, 2), dtype=np.int32)
        for i in range(len(players)):
            this_player_idxs = player_idxs == i
            actions[this_player_idxs] = players[i].act(i, states[this_player_idxs])

        if game_set.step(actions):
            break
    game_set.get_idxs_states(player_idxs, states)
    return player_idxs, states


class DumbPlayer:
    def act(self, player_idx, states):
        state_matrix = states_to_territory_matrix(states)
        attack_from = (state_matrix[:, :, player_idx + 1] == 1).argmax(axis=1)
        attack_to = (state_matrix[:, :, player_idx + 1] != 1).argmax(axis=1)
        return np.array([attack_from, attack_to]).T

    def learn(self, obs, actions, weights):
        return 0
