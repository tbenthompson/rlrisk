mod board;
pub use board::{setup_board, Board};

use ndarray;

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
    max_turns: usize,

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
    max_turns: usize,
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
        max_turns: max_turns,
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
        if self.turn_idx >= self.max_turns {
            self.player_idx = board::N_MAX_PLAYERS;
            self.phase = Phase::GameOver;
            return;
        }
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
        } else {
            self.n_attacks_left = 0;
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

    fn board_state_ndarray(
        &self,
        out: &mut ndarray::ArrayViewMut<'_, f32, ndarray::Ix1>,
    ) {
        let mut next_dof = 0;
        // one hot encode which player is currently acting
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
    }
}

#[pyfunction]
fn state_dim() -> usize {
    let dof_per_territory = 1 + board::N_MAX_PLAYERS;
    return board::N_MAX_PLAYERS + board::N_MAX_TERRITORIES * dof_per_territory;
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

    #[getter]
    fn board_state<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray1<f32> {
        let mut out = ndarray::Array::zeros((state_dim(),));
        self.board_state_ndarray(&mut out.view_mut());
        return out.into_pyarray(py);
    }

    fn first_owned_territory(&self, player_idx: usize) -> usize {
        for i in 0..self.board.n_territories {
            if self.board.territories[i].owner == player_idx {
                return i;
            }
        }
        return board::N_MAX_TERRITORIES;
    }

    fn first_unowned_territory(&self, player_idx: usize) -> usize {
        for i in 0..self.board.n_territories {
            if self.board.territories[i].owner != player_idx {
                return i;
            }
        }
        return board::N_MAX_TERRITORIES;
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
    pub fn n_territories(&self) -> usize {
        return self.board.n_territories;
    }
}

#[pyclass]
struct GameSet {
    games: Vec<GameState>,
}

#[pymethods]
impl GameSet {
    fn observe(
        &self,
        py_game_over: &numpy::PyArray1<bool>,
        py_idxs: &numpy::PyArray1<i32>,
        py_states: &numpy::PyArray2<f32>,
    ) {
        let mut game_over = unsafe { py_game_over.as_array_mut() };
        let mut player_idxs = unsafe { py_idxs.as_array_mut() };
        let mut states = unsafe { py_states.as_array_mut() };
        for i in 0..self.games.len() {
            game_over[i] = self.games[i].phase == Phase::GameOver;
            player_idxs[i] = self.games[i].player_idx as i32;
            self.games[i]
                .board_state_ndarray(&mut states.index_axis_mut(ndarray::Axis(0), i));
        }
    }

    fn step(&mut self, py_actions: numpy::PyReadonlyArray2<i32>) -> bool {
        let actions = py_actions.as_array();
        let mut all_done = true;
        for i in 0..self.games.len() {
            if self.games[i].phase == Phase::GameOver {
                continue;
            }
            self.games[i].step(actions[[i, 0]] as usize, actions[[i, 1]] as usize);
            if self.games[i].phase != Phase::GameOver {
                all_done = false;
            }
        }
        return all_done;
    }
}

#[pyfunction]
fn start_game_set(
    n_players: usize,
    n_territories: usize,
    baseline_reinforcements: usize,
    n_attacks_per_turn: usize,
    max_turns: usize,
    seed: Vec<u64>,
) -> GameSet {
    return GameSet {
        games: seed
            .iter()
            .map(|s| {
                start_game(
                    n_players,
                    n_territories,
                    baseline_reinforcements,
                    n_attacks_per_turn,
                    max_turns,
                    *s,
                )
            })
            .collect(),
    };
}

#[pymodule]
fn risk_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_class::<GameSet>()?;
    m.add_function(wrap_pyfunction!(start_game, m)?)?;
    m.add_function(wrap_pyfunction!(start_game_set, m)?)?;

    #[pyfn(m, "n_max_territories")]
    fn n_max_territories() -> usize {
        return board::N_MAX_TERRITORIES;
    }

    #[pyfn(m, "n_max_players")]
    fn n_max_players() -> usize {
        return board::N_MAX_PLAYERS;
    }

    #[pyfn(m, "state_dim")]
    fn py_state_dim() -> usize {
        return state_dim();
    }

    Ok(())
}
