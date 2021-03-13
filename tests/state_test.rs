use risk_ext;
use rstest::{fixture, rstest};

#[fixture]
fn game() -> risk_ext::GameState {
    return risk_ext::start_game();
}

#[rstest]
fn test_init_state(game: risk_ext::GameState) {
    assert_eq!(game.turn_idx, 0);
    assert_eq!(game.player_idx, 0);
    assert_eq!(game.phase, risk_ext::Phase::Reinforce);
}

#[rstest]
fn test_reinforce_step(mut game: risk_ext::GameState) {
    game.step(0, 1);
    assert_eq!(game.phase, risk_ext::Phase::Attack);
}

#[rstest]
fn test_attack_step(mut game: risk_ext::GameState) {
    game.step(0, 1);
    game.step(0, 1);
    assert_eq!(game.turn_idx, 0);
    assert_eq!(game.player_idx, 1);
    assert_eq!(game.phase, risk_ext::Phase::Reinforce);
}

#[test]
fn test_board_setup() {
    let board = risk_ext::setup_board();
    assert!(board.verify_state());
}

#[test]
fn test_attack_mechanics() {

}
