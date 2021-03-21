import rlrisk.env as env
from rlrisk.simple_policy import SimplePolicyPlayer
from rlrisk.train import Trainer

game_spec = dict(
    n_players=2,
    n_territories=8,
    baseline_reinforcements=3,
    reinforcements_per_territory=0,
    n_attacks_per_turn=5,
    max_turns=50,
)


def test_territory_matrix():
    _, _, history = env.play_games(
        game_spec, [env.DumbPlayer(), env.DumbPlayer()], [1, 2, 3], record=True
    )
    obs = history[2]
    reshaped_obs = env.states_to_territory_matrix(obs)
    assert reshaped_obs.shape[-2] == env.n_max_territories
    assert reshaped_obs.shape[-1] == env.n_max_players + 1


def test_train():
    # Just check that it runs for now. Maybe later do a mini-training with a
    # fixed set of seeds and do a golden master test
    ps = [SimplePolicyPlayer(), env.DumbPlayer()]
    Trainer().train(game_spec, ps, 2, 10)
