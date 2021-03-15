use risk_ext;
use rand::Rng;
use rstest::rstest;

#[rstest(n_players, n_territories, case(2, 3), case(3, 3), case(3, 4))]
fn test_init_state(n_players: usize, n_territories: usize) {
    let game = risk_ext::start_game(n_players, n_territories, 1, 1, 0);
    assert_eq!(game.turn_idx, 0);
    assert_eq!(game.player_idx, 0);
    assert_eq!(game.phase, risk_ext::Phase::Attack);
}

// TODO: Need a nicer way to "force" an algorithm for a phase
//
// #[rstest(n_armies, case(1), case(2), case(2000))]
// fn test_reinforce_step(mut game: risk_ext::GameState, n_armies: usize) {
//     let n_armies_start = game.get_board_state()[0];
//     game.step(0, n_armies);
//     assert_eq!(n_armies_start + 1.0, game.get_board_state()[0]);
//     assert_eq!(game.phase, risk_ext::Phase::Attack);
// }
//
// #[rstest(n_armies, case(1), case(2), case(2000))]
// fn test_reinforce_no_op_for_enemy_territory(mut game: risk_ext::GameState, n_armies: usize) {
//     let state = game.get_board_state();
//     game.step(1, n_armies);
//     assert_eq!(state, game.get_board_state());
//     assert_eq!(game.phase, risk_ext::Phase::Reinforce);
// }
//
// #[rstest]
// fn test_reinforce_no_op_for_illegal_territory(mut game: risk_ext::GameState) {
//     let state = game.get_board_state();
//     game.step(3, 1);
//     assert_eq!(state, game.get_board_state());
//     assert_eq!(game.phase, risk_ext::Phase::Reinforce);
// }

#[test]
fn test_attack_step() {
    let mut game = risk_ext::start_game(2, 3, 1, 1, 0);
    game.step(0, 1);
    // game.step(0, 1);
    assert_eq!(game.turn_idx, 0);
    assert_eq!(game.player_idx, 1);
    assert_eq!(game.phase, risk_ext::Phase::Attack);
}

#[test]
fn test_two_attacks() {
    let mut game = risk_ext::start_game(2, 3, 1, 2, 0);
    game.step(0, 2);
    assert_eq!(game.player_idx, 0);
    game.step(0, 2);

    assert_eq!(game.turn_idx, 0);
    assert_eq!(game.player_idx, 1);
    assert_eq!(game.phase, risk_ext::Phase::Attack);
}

#[test]
fn test_skip_attack() {
    // five attacks per turn, but skip the turn by specifying an invalid attack
    let mut game = risk_ext::start_game(2, 3, 1, 5, 0);
    game.step(0, 0);
    assert_eq!(game.turn_idx, 0);
    assert_eq!(game.player_idx, 1);
    assert_eq!(game.phase, risk_ext::Phase::Attack);
}

#[test]
fn test_no_changes_after_gameover() {
    // This is a useful test since in a vectorized context, some games will end before others and
    // it's nice if we can just keep performing actions on the finished games
    // without affecting the final state
    let mut game = risk_ext::start_game(2, 2, 100, 100, 0);
    for _i in 0..100 {
        game.step(0, 1);
    }
    assert_eq!(game.phase, risk_ext::Phase::GameOver);
    assert_eq!(game.player_idx, 0);
    let mut rng = rand::thread_rng();
    for _i in 0..100 {
        game.step(
            rng.gen_range(0..game.n_territories()),
            rng.gen_range(0..game.n_territories()),
        );
    }
    assert_eq!(game.phase, risk_ext::Phase::GameOver);
    assert_eq!(game.player_idx, 0);
}
