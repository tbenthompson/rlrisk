import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

import risk_ext


def vec_to_territory_matrix(game, v):
    return v[game.n_max_players :].reshape(
        (game.n_max_territories, 1 + game.n_max_players)
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


class Env:
    def __init__(self, spec):
        self.n_games = 0
        self.spec = spec
        self.reset()

    def reset(self, seed=None):
        self.n_games += 1
        if seed is None:
            seed = self.n_games
        self.game = start_game(self.spec, seed)
        return self.game.board_state

    def step(self, action):
        self.game.step(*action)
        done = self.game.phase == 3
        return self.game.board_state, float(done), done

    def get_state_dim(self):
        return self.game.board_state.shape[0]

    def get_action_dim(self):
        return self.game.n_territories

    def faceoff(self, players, n_games):
        winners = np.empty(n_games, dtype=np.int32)
        for i in range(n_games):
            winners[i] = self.play_game(players)
        winner_counts = np.unique(winners, return_counts=True)
        return winners, winner_counts[0], winner_counts[1] / n_games

    def play_game(env, players, verbose=False, max_turns=None, plot_each_turn=False):
        env.reset()
        next_turn_plot = 0
        while True:
            if plot_each_turn and env.game.turn_idx == next_turn_plot:
                vis(env.game)
                next_turn_plot += 1
            action = players[env.game.player_idx].act(env.game, env.game.board_state)
            if verbose:
                print(
                    f"turn={env.game.turn_idx}, player={env.game.player_idx}, phase={env.game.phase}, action={action}, board={env.game.board_state}"
                )
            _, _, done = env.step(action)
            if done:
                break
            if max_turns is not None:
                if env.game.turn_idx > max_turns:
                    break
        return env.game.player_idx


class DumbPlayer:
    def act(self, game, state_vec):
        owner_col = vec_to_territory_matrix(game, state_vec)[:, game.player_idx + 1]
        attack_from = (owner_col == 1).argmax()
        attack_to = (owner_col != 1).argmax()
        return attack_from, attack_to

    def learn(self, obs, actions, weights):
        return 0
