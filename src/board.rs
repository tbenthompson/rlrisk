use ndarray::Array;
use rand::prelude::*;

const N_MAX_CARDS: usize = 0;
const N_MAX_TERRITORIES: usize = 2;
const N_MAX_PLAYERS: usize = 2;

#[derive(Debug, Clone, Copy)]
pub struct Territory {
    pub army_count: usize,
    pub owner: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Card {
    pub territory_idx: usize,
    pub color: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct PlayerData {
    pub n_controlled: usize,
    pub cards: [Card; N_MAX_CARDS],
}

#[derive(Debug)]
pub struct Board {
    pub n_players: usize,
    pub player_data: [PlayerData; N_MAX_PLAYERS],

    pub n_territories: usize,
    pub territories: [Territory; N_MAX_TERRITORIES],

    pub rng: rand::rngs::StdRng,
    dice_uniform: rand::distributions::Uniform<u8>
}

impl Board {
    pub fn reinforce(&mut self, territory_idx: usize, n_armies: usize) {
        self.territories[territory_idx].army_count += n_armies;
    }

    pub fn attack(&mut self, from: usize, to: usize) -> (usize, usize, bool) {
        let t = &mut self.territories;
        let n_attacker = std::cmp::min(t[from].army_count - 1, 3);
        let n_defender = std::cmp::min(t[to].army_count, 2);
        let mut dice: [u8; 5] = [0; 5];
        for i in 0..5 {
            dice[i] = self.dice_uniform.sample(&mut self.rng);
        }
        let (attacker_deaths, defender_deaths) =
            attack_mechanics(n_attacker, n_defender, dice);
        t[from].army_count -= attacker_deaths;
        t[to].army_count -= defender_deaths;
        let conquer = t[to].army_count == 0;
        if conquer {
            self.player_data[t[to].owner].n_controlled -= 1;
            self.player_data[t[from].owner].n_controlled += 1;
            t[to].owner = t[from].owner;
        }
        (attacker_deaths, defender_deaths, conquer)
    }

    pub fn fortify(&mut self, from: usize, to: usize, n_armies: usize) {
        assert!(n_armies < self.territories[from].army_count);
        self.territories[from].army_count -= n_armies;
        self.territories[to].army_count += n_armies;
    }

    pub fn new_card(&mut self, player_idx: usize) {
        let mut p = &mut self.player_data[player_idx];
        for i in 0..N_MAX_CARDS {
            if p.cards[i].territory_idx == N_MAX_TERRITORIES {
                p.cards[i].territory_idx = self.rng.gen_range(0..self.n_territories);
                p.cards[i].color = self.rng.gen_range(0..3);
                break;
            }
        }
    }

    pub fn verify_state(&self) -> bool {
        for i in 0..self.n_territories {
            if self.territories[i].owner >= N_MAX_PLAYERS {
                return false;
            }
            if self.territories[i].army_count < 1 {
                return false;
            }
        }
        let mut n_total_controlled = 0;
        for i in 0..self.n_players {
            n_total_controlled += self.player_data[i].n_controlled;
        }
        if n_total_controlled != self.n_territories {
            return false;
        }
        return true;
    }

    pub fn to_array(&self) -> ndarray::Array1<f32> {
        let dof_per_territory = 1 + N_MAX_PLAYERS;
        let mut out = Array::zeros((N_MAX_TERRITORIES * dof_per_territory,));

        for i in 0..N_MAX_TERRITORIES {
            out[dof_per_territory * i + 0] = self.territories[i].army_count as f32;
            // one hot encode owner
            for j in 0..N_MAX_PLAYERS {
                out[dof_per_territory * i + 1 + j] =
                    (self.territories[i].owner == j) as i32 as f32;
            }
        }
        return out;
    }

    pub fn is_valid_attack(&self, player_idx: usize, from: usize, to: usize) -> bool {
        if from >= self.n_territories || self.territories[from].owner != player_idx {
            return false;
        }
        if to >= self.n_territories || self.territories[to].owner == player_idx {
            return false;
        }
        return true;
    }
}

fn attack_mechanics(
    n_attacker: usize,
    n_defender: usize,
    dice: [u8; 5]
) -> (usize, usize) {
    assert!(n_attacker <= 3);
    assert!(n_defender == 2 || n_defender == 1);

    if n_attacker == 0 {
        return (0, 0);
    }

    let mut attacker_dice: Vec<u8> = dice[0..n_attacker].to_vec();
    let mut defender_dice: Vec<u8> = dice[3..(3+n_defender)].to_vec();
    attacker_dice.sort();
    defender_dice.sort();

    let mut attacker_deaths = 0;
    let mut defender_deaths = 0;
    if attacker_dice[n_attacker - 1] > defender_dice[n_defender - 1] {
        defender_deaths += 1;
    } else {
        attacker_deaths += 1;
    }
    if n_defender == 2 && n_attacker >= 2 {
        if attacker_dice[n_attacker - 2] > defender_dice[n_defender - 2] {
            defender_deaths += 1;
        } else {
            attacker_deaths += 1;
        }
    }
    return (attacker_deaths, defender_deaths);
}

pub fn setup_board() -> Board {
    let n_starting_armies = N_MAX_TERRITORIES * 2;
    let n_players = N_MAX_PLAYERS;
    let n_territories = N_MAX_TERRITORIES;
    let mut board = Board {
        n_players: n_players,
        player_data: [PlayerData {
            n_controlled: 0,
            cards: [Card {
                territory_idx: N_MAX_TERRITORIES,
                color: -1,
            }; N_MAX_CARDS],
        }; N_MAX_PLAYERS],

        n_territories: n_territories,
        territories: [Territory {
            army_count: 0,
            owner: N_MAX_PLAYERS,
        }; N_MAX_TERRITORIES],

        rng: StdRng::from_seed([0; 32]),
        dice_uniform: rand::distributions::Uniform::from(0..6)
    };

    for _i in 0..n_starting_armies {
        for j in 0..n_players {
            let territory_idx = starting_place(&board, j);
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

fn starting_place(board: &Board, player_idx: usize) -> usize {
    for t_i in 0..board.n_territories {
        if board.territories[t_i].owner == N_MAX_PLAYERS {
            return t_i;
        }
    }
    for t_i in 0..board.n_territories {
        if board.territories[t_i].owner == player_idx {
            return t_i;
        }
    }
    return N_MAX_TERRITORIES;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_attack_mechanics() {
        assert_eq!(attack_mechanics(3, 2, [5,5,5,5,5]), (2, 0));
        assert_eq!(attack_mechanics(3, 2, [5,5,5,4,4]), (0, 2));
        assert_eq!(attack_mechanics(1, 1, [5,0,0,5,0]), (1, 0));
        assert_eq!(attack_mechanics(1, 2, [4,0,0,3,5]), (1, 0));
        assert_eq!(attack_mechanics(0, 2, [4,0,0,3,5]), (0, 0));
    }

    #[rstest(n_a, n_d, dice,
        case(1, 0, [0,0,0,0,0]),
        case(1, 3, [0,0,0,0,0]),
        case(4, 2, [0,0,0,0,0])
    )]
    #[should_panic]
    fn test_bad_mechanics_input(n_a: usize, n_d: usize, dice: [u8; 5]) {
        attack_mechanics(n_a, n_d, dice);
    }
}
