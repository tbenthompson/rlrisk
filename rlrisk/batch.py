import numpy as np


class Batch:
    def __init__(self):
        # set epoch = -1 because reset_recording will immediately increment to 0
        self.epoch = -1
        self.reset_recording()

    def reset_recording(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.returns_to_go = []
        self.lengths = []
        self.did_i_win = []
        self.epoch += 1

    def record_games(self, player_idx, whose_turn, obs, acts, game_not_over, winners):
        did_i_win = (winners == player_idx).astype(np.float32)
        my_turn = whose_turn[:, :] == player_idx

        n_games = obs.shape[1]
        for j in range(n_games):
            this_game_obs = obs[my_turn[:, j] & game_not_over[:, j], j]
            this_game_action = acts[my_turn[:, j] & game_not_over[:, j], j]
            # game_start_idx = self.rewards.shape[0]
            # game_end_idx = game_start_idx + this_game_obs.shape[0]
            for i in range(this_game_obs.shape[0]):
                self.observations.append(this_game_obs[i])
                self.actions.append(this_game_action[i, :])

                # rewards come from the next turns observations so we skip i == 0
                self.returns_to_go.append(did_i_win[j])
                # if i != 0:
                #     territory_matrix = \
                #       env.states_to_territory_matrix(this_game_obs[i])
                #     total_armies = np.sum(territory_matrix[:, 0])
                #     my_armies = np.sum(
                #         territory_matrix[:, 0] * territory_matrix[:, player_idx + 1]
                #     )
                #     self.rewards.append(
                #         0.5 * my_armies / total_armies / 50
                #     )
            # reward for the final turn is whether the agent won.
            self.rewards.append(did_i_win[j])
            # self.epoch_data['returns_to_go'].append(

            self.lengths.append(this_game_obs.shape[0])
        self.did_i_win.extend(did_i_win)
