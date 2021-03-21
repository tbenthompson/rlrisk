import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

import rlrisk.risk_ext

n_max_players = rlrisk.risk_ext.n_max_players()
n_max_territories = rlrisk.risk_ext.n_max_territories()
state_dim = rlrisk.risk_ext.state_dim()


def states_to_territory_matrix(states):
    return states[..., n_max_players:].reshape(
        (*states.shape[:-1], n_max_territories, n_max_players + 1)
    )


def start_game_set(spec, seeds):
    return rlrisk.risk_ext.start_game_set(
        spec["n_players"],
        spec["n_territories"],
        spec["baseline_reinforcements"],
        spec["reinforcements_per_territory"],
        spec["n_attacks_per_turn"],
        spec["max_turns"],
        seeds,
    )


def vis(state):
    tm = states_to_territory_matrix(state)
    armies = tm[:, 0]
    owners = (tm[:, 1:] == 1.0).argmax(axis=1)
    cmap = plt.get_cmap("coolwarm")
    m = matplotlib.cm.ScalarMappable(cmap=cmap)

    plt.bar(
        np.arange(armies.shape[0]),
        armies,
        color=m.to_rgba(owners / (np.max(owners) + 1), norm=False),
        align="center",
    )
    plt.ylabel("number of armies")
    plt.xlabel("territory idx")
    plt.show()


def play_games(spec, players, seeds, verbose=False, record=False, should_plot=False):
    n_games = len(seeds)
    game_set = start_game_set(spec, seeds)
    history = []

    game_over = np.empty(n_games, dtype=bool)
    player_idxs = np.empty(n_games, dtype=np.int32)
    states = np.empty((n_games, state_dim), dtype=np.float32)
    while True:
        if should_plot:
            vis(states[0])

        game_set.observe(game_over, player_idxs, states)

        actions = np.zeros((n_games, 2), dtype=np.int32)
        for i in range(len(players)):
            this_player_idxs = player_idxs == i
            actions[this_player_idxs] = players[i].act(i, states[this_player_idxs])

        if verbose:
            print(f"player={player_idxs[0]}, action={actions[0]}, board={states[0]}")

        if record:
            history.append(
                (game_over.copy(), player_idxs.copy(), states.copy(), actions.copy())
            )

        if game_set.step(actions):
            break

    game_set.observe(game_over, player_idxs, states)
    if record:
        game_over = np.array([data[0] for data in history])
        whose_turn = np.array([data[1] for data in history])
        obs = np.array([data[2] for data in history])
        acts = np.array([data[3] for data in history])
        history = (game_over, whose_turn, obs, acts)

    return player_idxs, states, history


class DumbPlayer:
    batch_type = type(None)

    def act(self, player_idx, states):
        state_matrix = states_to_territory_matrix(states)
        attack_from = (state_matrix[:, :, player_idx + 1] == 1).argmax(axis=1)
        attack_to = (state_matrix[:, :, player_idx + 1] != 1).argmax(axis=1)
        return np.array([attack_from, attack_to]).T

    def learn(self, *args):
        return {}
