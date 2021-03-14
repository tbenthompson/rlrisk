use risk_ext;
use rstest::{fixture, rstest};

#[fixture]
fn game() -> risk_ext::GameState {
    return risk_ext::start_game(2, 3, 0);
}

#[rstest]
fn test_init_state(game: risk_ext::GameState) {
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

#[rstest]
fn test_attack_step(mut game: risk_ext::GameState) {
    game.step(0, 1);
    // game.step(0, 1);
    assert_eq!(game.turn_idx, 0);
    assert_eq!(game.player_idx, 1);
    assert_eq!(game.phase, risk_ext::Phase::Attack);
}
