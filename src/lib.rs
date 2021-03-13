/* Plan:
 * - (CHECK) simple Risk engine
 * - encode the game state as a vector
 * - try some kind of reinforcement learning algorithm
 * - vectorized/matrixify the risk engine so that we run many games at once.
 * - GPU-ify the risk engine?
 * - play with fancier methods
 * - cool idea: what if we just convert the Board struct to binary?!
 *
 * Dimensions on which we can simplify Risk rules:
 * - fewer territories (simple == 2 or 3)
 * - territory connectivity (simple == fully connected)
 * - fewer players (simple == 2)
 * - continents (simple == no continents)
 * - number of reinforcements (simple == constant)
 * - number of cards (simple == 0)
 * - number of attacks allowed per turn (simple == 1)
 * - move full stack vs choice in move size (simple == move full stack)
 * - fortification (simple == no fortification)
 *
 * Common Risk variations
 * - fixed vs progressive cards
 * - blizzards
 * - fog of war
 * - auto vs manual starting placement
 * - different maps
 */
mod board;
mod strategy;
pub use board::{Board, setup_board};
use strategy::Strategy;

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[derive(Debug, PartialEq)]
pub enum Phase {
    Reinforce,
    Attack,
    Fortify,
    GameOver
}

#[derive(Debug)]
pub struct GameState {
    board: Board,
    pub turn_idx: usize,
    pub player_idx: usize, 
    pub phase: Phase,

    // Reinforce phase state variables
    n_reinforcements: usize,

    // Attack phase state variables
    n_attacks_left: usize,
    won_an_attack: bool
}

pub fn start_game() -> GameState {
    let board = board::setup_board();

    let mut state = GameState {
        turn_idx: 0,
        board: board,
        player_idx: 0,
        phase: Phase::Reinforce,
        n_reinforcements: 0,
        n_attacks_left: 0,
        won_an_attack: false
    };

    state.begin_reinforce_phase();
    state
}

const N_ATTACKS_PER_TURN: usize = 1;

impl GameState {
    pub fn step(&mut self, arg0: usize, arg1: usize) {
        match self.phase {
            Phase::Reinforce => self.reinforce_step(arg0, arg1),
            Phase::Attack => self.attack_step(arg0, arg1),
            Phase::Fortify => self.fortify_step(),
            Phase::GameOver => {}
        }
    }

    pub fn get_board_state(&self) -> ndarray::Array1<f32> {
        return self.board.to_array();
    }

    fn begin_next_turn(&mut self) {
        self.player_idx += 1; 
        if self.player_idx == self.board.n_players {
            self.turn_idx += 1;
            self.player_idx = 0;
        }
        self.begin_reinforce_phase();
    }

    fn begin_reinforce_phase(&mut self) {
        self.phase = Phase::Reinforce;
        self.n_reinforcements = 1;
    }

    fn reinforce_step(&mut self, territory_idx: usize, requested_count: usize) {
        let count = std::cmp::min(requested_count, self.n_reinforcements);
        self.board.reinforce(territory_idx, count);
        self.n_reinforcements -= count;
        if self.n_reinforcements == 0 {
            self.begin_attack_phase();
        }
    }

    fn begin_attack_phase(&mut self) {
        self.phase = Phase::Attack;
        self.n_attacks_left = N_ATTACKS_PER_TURN;
        self.won_an_attack = false;
    }

    fn attack_step(&mut self, from: usize, to: usize) {
        self.n_attacks_left -= 1;

        if self.board.is_valid_attack(self.player_idx, from, to) {
            let outcome = self.board.attack(from, to);
            // if self.verbose {
            // println!("Attack: {}, {}, {:?}", from, to, outcome);
            // }
            if outcome.2 {
                self.won_an_attack = true;
                self.board
                    .fortify(from, to, self.board.territories[from].army_count - 1);
                if self.board.player_data[self.player_idx].n_controlled
                    == self.board.n_territories
                {
                    // println!("GAME OVER WINNER {}", self.player_idx);
                    self.phase = Phase::GameOver;
                    return;
                }
            }
        }

        if self.n_attacks_left == 0 {
            self.begin_fortify_phase();
        }
    }

    fn begin_fortify_phase(&mut self) {
        self.phase = Phase::Fortify;
        self.end_turn();
    }

    fn fortify_step(&mut self) {
    }

    fn end_turn(&mut self) {
        if self.won_an_attack  {
            self.board.new_card(self.player_idx);
        }
        self.begin_next_turn();
    }
}

#[pyfunction]
fn tester() {
    let mut state = start_game();

    let players = [
        strategy::Dumb { player_idx: 0 },
        strategy::Dumb { player_idx: 1 },
    ];

    loop {
        match state.phase {
            Phase::Reinforce => {
                let (arg0, arg1) = players[state.player_idx].reinforce_step(&state.board, state.n_reinforcements);
                state.step(arg0, arg1)
            }
            Phase::Attack => {
                let (arg0, arg1) = players[state.player_idx].attack_step(&state.board);
                state.step(arg0, arg1)
            }
            Phase::Fortify => {
            }
            Phase::GameOver => {
                break;
            }
        }
    }
}

// #[pyfunction]
// fn run_dumb_game(py: Python, verbose: bool) -> (usize, &numpy::PyArray1<f32>) {
//     let players: Vec<Box<dyn strategy::Strategy>> = vec![
//         Box::new(strategy::Dumb { player_idx: 0 }),
//         Box::new(strategy::Dumb { player_idx: 1 }),
//     ];
//     let mut game = Game {
//         verbose: verbose,
//         board: setup_board(),
//         players: players,
//     };
//     let winner = game.play();
//     println!("Board: {:?}", game.board);
//     let out = game.board.to_array().into_pyarray(py);
//     return (winner, out);
// }
//
// struct PyCallbackStrategy {
//     player_idx: usize,
//     callback: PyObject,
// }
//
// impl strategy::Strategy for PyCallbackStrategy {
//     fn get_player_idx(&self) -> usize {
//         return self.player_idx;
//     }
//
//     fn attack_step(&self, board: &board::Board) -> (usize, usize) {
//         Python::with_gil(|py| {
//             // NOTE: This is only valid when the python player is index 0
//             let state = board.to_array().into_pyarray(py);
//             let result = self.callback.call1(py, (state,));
//             return result.unwrap().extract::<(usize, usize)>(py).unwrap();
//         })
//     }
// }
//
// #[pyfunction]
// fn run_py_vs_dumb_game(callback: PyObject, verbose: bool) -> usize {
//     let players: Vec<Box<dyn strategy::Strategy>> = vec![
//         Box::new(PyCallbackStrategy { player_idx: 0, callback: callback }),
//         Box::new(strategy::Dumb { player_idx: 1 }),
//     ];
//     let mut game = Game {
//         verbose: verbose,
//         board: setup_board(),
//         players: players,
//     };
//     return game.play();
// }

#[pymodule]
fn risk_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(run_dumb_game, m)?)?;
    // m.add_function(wrap_pyfunction!(run_py_vs_dumb_game, m)?)?;
    m.add_function(wrap_pyfunction!(tester, m)?)?;

    Ok(())
}
