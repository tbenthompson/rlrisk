mod board;
pub use board::{setup_board, Board};

use ndarray::Array;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Phase {
    Reinforce,
    Attack,
    Fortify,
    GameOver,
}

#[pyclass]
#[derive(Debug)]
pub struct GameState {
    verbose: bool,
    baseline_reinforcements: usize,
    n_attacks_per_turn: usize,

    board: Board,

    #[pyo3(get)]
    pub turn_idx: usize,
    #[pyo3(get)]
    pub player_idx: usize,
    pub phase: Phase,

    // Reinforce phase state variables
    n_reinforcements: usize,

    // Attack phase state variables
    n_attacks_left: usize,
    won_an_attack: bool,
}

#[pyfunction]
pub fn start_game(
    n_players: usize,
    n_territories: usize,
    baseline_reinforcements: usize,
    n_attacks_per_turn: usize,
    seed: u64,
) -> GameState {
    let mut seed_array: [u8; 32] = [0; 32];
    for i in 0..8 {
        seed_array[i] = seed.to_ne_bytes()[i];
    }
    let board = board::setup_board(n_players, n_territories, seed_array);

    let mut state = GameState {
        verbose: false,
        baseline_reinforcements: baseline_reinforcements,
        n_attacks_per_turn: n_attacks_per_turn,
        turn_idx: 0,
        board: board,
        player_idx: 0,
        phase: Phase::Reinforce,
        n_reinforcements: 0,
        n_attacks_left: 0,
        won_an_attack: false,
    };

    state.begin_reinforce_phase();
    state
}

impl GameState {
    fn begin_next_turn(&mut self) {
        self.next_player();
        self.begin_reinforce_phase();
    }

    fn next_player(&mut self) {
        self.player_idx += 1;
        if self.player_idx == self.board.n_players {
            self.turn_idx += 1;
            self.player_idx = 0;
        }
        if self.board.player_data[self.player_idx].n_controlled == 0 {
            self.next_player();
        }
    }

    fn begin_reinforce_phase(&mut self) {
        self.phase = Phase::Reinforce;
        self.n_reinforcements = self.baseline_reinforcements;
        self.dumb_reinforce();
    }

    fn reinforce_step(&mut self, territory_idx: usize, requested_count: usize) {
        if territory_idx >= self.board.n_territories {
            return;
        }
        if self.board.territories[territory_idx].owner != self.player_idx {
            return;
        }
        let count = std::cmp::min(requested_count, self.n_reinforcements);
        self.n_reinforcements -= count;
        self.board.reinforce(territory_idx, count);
        if self.n_reinforcements == 0 {
            self.begin_attack_phase();
        }
    }

    fn dumb_reinforce(&mut self) {
        // Next player's reinforcement step happens automatically assigning troops to the
        // first identified territory.
        if self.phase == Phase::Reinforce {
            for t_i in 0..self.board.n_territories {
                if self.board.territories[t_i].owner == self.player_idx {
                    self.reinforce_step(t_i, self.n_reinforcements);
                }
            }
        }
    }

    fn begin_attack_phase(&mut self) {
        self.phase = Phase::Attack;
        self.n_attacks_left = self.n_attacks_per_turn;
        self.won_an_attack = false;
    }

    fn attack_step(&mut self, from: usize, to: usize) {
        self.n_attacks_left -= 1;

        if self.board.is_valid_attack(self.player_idx, from, to) {
            let outcome = self.board.attack(from, to);
            if outcome.2 {
                self.won_an_attack = true;
                self.board
                    .fortify(from, to, self.board.territories[from].army_count - 1);
                if self.board.player_data[self.player_idx].n_controlled
                    == self.board.n_territories
                {
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

    fn fortify_step(&mut self) {}

    fn end_turn(&mut self) {
        if self.won_an_attack {
            self.board.new_card(self.player_idx);
        }
        self.begin_next_turn();
    }
}

#[pymethods]
impl GameState {
    pub fn step(&mut self, arg0: usize, arg1: usize) {
        match self.phase {
            Phase::Reinforce => self.reinforce_step(arg0, arg1),
            Phase::Attack => self.attack_step(arg0, arg1),
            Phase::Fortify => self.fortify_step(),
            Phase::GameOver => {}
        }
    }
    
    fn board_state_size(&self) -> usize {
        let dof_per_territory = 1 + board::N_MAX_PLAYERS;
        return board::N_MAX_PLAYERS + board::N_MAX_TERRITORIES * dof_per_territory;
    }

    #[getter]
    fn board_state<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray1<f32> {
        // TODO: re-use board state vector.
        let mut out = Array::zeros((self.board_state_size(),));

        let mut next_dof = 0;
        for i in 0..board::N_MAX_PLAYERS {
            out[next_dof] = (self.player_idx == i) as i32 as f32;
            next_dof += 1;
        }

        for i in 0..board::N_MAX_TERRITORIES {
            out[next_dof] = self.board.territories[i].army_count as f32;
            next_dof += 1;
            // one hot encode owner
            for j in 0..board::N_MAX_PLAYERS {
                out[next_dof] = (self.board.territories[i].owner == j) as i32 as f32;
                next_dof += 1;
            }
        }
        return out.into_pyarray(py);
    }

    #[getter]
    fn phase(&self) -> u8 {
        return self.phase as u8;
    }

    #[getter]
    fn n_max_players(&self) -> usize {
        return board::N_MAX_PLAYERS;
    }

    #[getter]
    fn n_max_territories(&self) -> usize {
        return board::N_MAX_TERRITORIES;
    }

    #[getter]
    fn n_players(&self) -> usize {
        return self.board.n_players;
    }

    #[getter]
    fn n_territories(&self) -> usize {
        return self.board.n_territories;
    }
}

#[pymodule]
fn risk_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_function(wrap_pyfunction!(start_game, m)?)?;

    Ok(())
}
