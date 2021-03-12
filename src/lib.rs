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
use board::Board;

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

struct Game {
    verbose: bool,
    board: Board,
    players: Vec<Box<dyn strategy::Strategy>>,
}

impl Game {
    fn play(&mut self) -> usize {
        let max_turns = 100;
        for turn_idx in 0..max_turns {
            for player_idx in 0..self.board.n_players {
                let winner = self.turn(turn_idx, player_idx);
                if winner != board::N_MAX_PLAYERS {
                    return winner;
                }
                assert!(self.board.verify_state());
            }
        }
        return board::N_MAX_PLAYERS;
    }

    /* Returns the winner if the game is over. */
    fn turn(&mut self, _turn_idx: usize, player_idx: usize) -> usize {
        if self.verbose {
            println!(
                "Turn ({}, {}), Territories: {:?}",
                _turn_idx, player_idx, self.board.territories
            );
        }
        let p = &self.players[player_idx];

        let mut n_reinforcements = 1;
        while n_reinforcements != 0 {
            let (territory_idx, mut count) =
                p.reinforce_step(&self.board, n_reinforcements);
            count = std::cmp::min(count, n_reinforcements);
            self.board.reinforce(territory_idx, count);
            n_reinforcements -= count;
        }

        // attack and fortify
        let mut won_an_attack = false;
        for _attack_idx in 0..board::N_ATTACKS_PER_TURN {

            let (from, to) = p.attack_step(&self.board);
            if !self.board.is_valid_attack(player_idx, from, to) {
                continue;
            }
            let outcome = self.board.attack(from, to);
            if self.verbose {
                println!("Attack: {}, {}, {:?}", from, to, outcome);
            }
            if outcome.2 {
                won_an_attack = true;
                self.board
                    .fortify(from, to, self.board.territories[from].army_count - 1);
                if self.board.player_data[player_idx].n_controlled
                    == self.board.n_territories
                {
                    // println!("GAME OVER WINNER {}", player_idx);
                    return player_idx;
                }
            }
        }
        // fortify

        // add card
        if won_an_attack {
            self.board.new_card(player_idx);
        }
        return board::N_MAX_PLAYERS;
    }
}

fn setup_board(players: &Vec<Box<dyn strategy::Strategy>>) -> Board {
    let n_starting_armies = 3;
    let n_players = board::N_MAX_PLAYERS;
    let n_territories = board::N_MAX_TERRITORIES;
    let mut board = Board {
        n_players: n_players,
        player_data: [board::PlayerData {
            n_controlled: 0,
            cards: [board::Card {
                territory_idx: board::N_MAX_TERRITORIES,
                color: -1,
            }; board::N_MAX_CARDS],
        }; board::N_MAX_PLAYERS],

        n_territories: n_territories,
        territories: [board::Territory {
            army_count: 0,
            owner: board::N_MAX_PLAYERS,
        }; board::N_MAX_TERRITORIES],

        rng: rand::thread_rng(),
    };

    for _i in 0..n_starting_armies {
        for j in 0..n_players {
            let territory_idx = players[j].starting_step(&board);
            if board.territories[territory_idx].owner != j {
                board.territories[territory_idx].owner = j;
                board.player_data[j].n_controlled += 1;
            }
            board.territories[territory_idx].army_count += 1;
        }
    }

    assert!(board.verify_state());
    board
}

#[pyfunction]
fn run_dumb_game(py: Python, verbose: bool) -> (usize, &numpy::PyArray1<f32>) {
    let players: Vec<Box<dyn strategy::Strategy>> = vec![
        Box::new(strategy::Dumb { player_idx: 0 }),
        Box::new(strategy::Dumb { player_idx: 1 }),
    ];
    let mut game = Game {
        verbose: verbose,
        board: setup_board(&players),
        players: players,
    };
    let winner = game.play();
    println!("Board: {:?}", game.board);
    let out = game.board.to_array().into_pyarray(py);
    return (winner, out);
}

struct PyCallbackStrategy {
    player_idx: usize,
    callback: PyObject,
}

impl strategy::Strategy for PyCallbackStrategy {
    fn get_player_idx(&self) -> usize {
        return self.player_idx;
    }

    fn attack_step(&self, board: &board::Board) -> (usize, usize) {
        Python::with_gil(|py| {
            // NOTE: This is only valid when the python player is index 0
            let state = board.to_array().into_pyarray(py);
            let result = self.callback.call1(py, (state,));
            return result.unwrap().extract::<(usize, usize)>(py).unwrap();
        })
    }
}

#[pyfunction]
fn run_py_vs_dumb_game(callback: PyObject, verbose: bool) -> usize {
    let players: Vec<Box<dyn strategy::Strategy>> = vec![
        Box::new(PyCallbackStrategy { player_idx: 0, callback: callback }),
        Box::new(strategy::Dumb { player_idx: 1 }),
    ];
    let mut game = Game {
        verbose: verbose,
        board: setup_board(&players),
        players: players,
    };
    return game.play();
}

#[pymodule]
fn risk_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_dumb_game, m)?)?;
    m.add_function(wrap_pyfunction!(run_py_vs_dumb_game, m)?)?;

    Ok(())
}
