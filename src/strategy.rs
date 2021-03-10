use crate::board;

pub trait Strategy {
    fn get_player_idx(&self) -> usize;

    /* Select territories to place troops at the beginning of the game. This function will be
     * called multiple times and returns the territory index in which to place one troop each time.
     */
    fn starting_step(&self, board: &board::Board) -> usize {
        for t_i in 0..board.n_territories {
            if board.territories[t_i].owner == board::N_MAX_PLAYERS {
                return t_i;
            }
        }
        for t_i in 0..board.n_territories {
            if board.territories[t_i].owner == self.get_player_idx() {
                return t_i;
            }
        }
        return board::N_MAX_TERRITORIES;
    }

    /*
     * A reinforce is specified by the (territory_idx, n_armies) referring to the territory we
     * reinforce and the number to place at that location.
     */
    fn reinforce_step(
        &self,
        board: &board::Board,
        n_reinforcements: usize,
    ) -> (usize, usize) {
        for t_i in 0..board.n_territories {
            if board.territories[t_i].owner == self.get_player_idx() {
                return (t_i, n_reinforcements);
            }
        }
        assert!(false);
        return (0, 0);
    }

    /* An attack is specified by the (from, to) tuple referring to the territory we attack from and
     * the territory under attack.
     * Returning (N_MAX_TERRITORIES, N_MAX_TERRITORIES) implies no attack.
     */
    fn attack_step(&self, _board: &board::Board) -> (usize, usize);

    /* not allowed for now */
    fn fortify_step(&self, _board: &board::Board) {}
}

#[derive(Clone)]
pub struct Dumb {
    pub player_idx: usize,
}

impl Strategy for Dumb {
    fn get_player_idx(&self) -> usize {
        return self.player_idx;
    }

    fn attack_step(&self, board: &board::Board) -> (usize, usize) {
        for t_i in 0..board.n_territories {
            if board.territories[t_i].owner == self.player_idx {
                for t_i2 in 0..board.n_territories {
                    if board.territories[t_i2].owner != self.player_idx {
                        return (t_i, t_i2);
                    }
                }
            }
        }
        return (board::N_MAX_TERRITORIES, board::N_MAX_TERRITORIES);
    }
}
