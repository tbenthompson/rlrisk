use rand::prelude::*;

pub const N_MAX_CARDS: usize = 0;
pub const N_MAX_TERRITORIES: usize = 8;
pub const N_MAX_PLAYERS: usize = 3;

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
    dice_uniform: rand::distributions::Uniform<u8>,
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

    pub fn fortify(
        &mut self,
        from: usize,
        to: usize,
        n_requested_armies: usize,
    ) -> usize {
        if self.territories[from].owner != self.territories[to].owner {
            return 0;
        }
        let n_armies =
            std::cmp::min(n_requested_armies, self.territories[from].army_count - 1);
        self.territories[from].army_count -= n_armies;
        self.territories[to].army_count += n_armies;
        return n_armies;
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

    pub fn assert_state(&self) {
        for i in 0..self.n_territories {
            assert!(self.territories[i].owner < N_MAX_PLAYERS);
            assert!(self.territories[i].army_count >= 1);
        }
        let mut n_total_controlled = 0;
        for i in 0..self.n_players {
            n_total_controlled += self.player_data[i].n_controlled;
        }
        assert_eq!(n_total_controlled, self.n_territories);
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
    dice: [u8; 5],
) -> (usize, usize) {
    assert!(n_attacker <= 3);
    assert!(n_defender == 2 || n_defender == 1);

    if n_attacker == 0 {
        return (0, 0);
    }

    let mut attacker_dice: Vec<u8> = dice[0..n_attacker].to_vec();
    let mut defender_dice: Vec<u8> = dice[3..(3 + n_defender)].to_vec();
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

pub fn setup_board(n_players: usize, n_territories: usize, seed: [u8; 32]) -> Board {
    assert!(n_players <= N_MAX_PLAYERS);
    assert!(n_territories <= N_MAX_TERRITORIES);

    let n_starting_armies = n_territories * 2;
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

        rng: StdRng::from_seed(seed),
        dice_uniform: rand::distributions::Uniform::from(0..6),
    };

    let rand_territory = rand::distributions::Uniform::from(0..board.n_territories);
    let mut unclaimed_territories = n_territories;
    for _i in 0..n_starting_armies {
        for j in 0..n_players {
            let ti = loop {
                let t = rand_territory.sample(&mut board.rng);
                if board.territories[t].owner == N_MAX_PLAYERS {
                    unclaimed_territories -= 1;
                    break t
                }
                if unclaimed_territories == 0 && board.territories[t].owner == j {
                    break t
                }
            };
            if board.territories[ti].owner != j {
                board.territories[ti].owner = j;
                board.player_data[j].n_controlled += 1;
            }
            board.territories[ti].army_count += 1;
        }
    }

    board.assert_state();
    board
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn load_board(territory_spec: &[(usize, usize)], seed: [u8; 32]) -> Board {
        let mut territories = [Territory {
            army_count: 0,
            owner: N_MAX_PLAYERS,
        }; N_MAX_TERRITORIES];
        let mut n_players = 0;
        let mut n_territories = N_MAX_TERRITORIES;
        let mut player_data = [PlayerData {
            n_controlled: 0,
            cards: [Card {
                territory_idx: N_MAX_TERRITORIES,
                color: -1,
            }; N_MAX_CARDS],
        }; N_MAX_PLAYERS];

        for i in 0..territory_spec.len() {
            if territory_spec[i].0 > 0 {
                territories[i].army_count = territory_spec[i].0;
                territories[i].owner = territory_spec[i].1;
                n_players = std::cmp::max(n_players, territories[i].owner + 1);
                player_data[territories[i].owner].n_controlled += 1;
                n_territories = i + 1;
            } else {
                break;
            }
        }

        let board = Board {
            n_players: n_players,
            player_data: player_data,
            n_territories: n_territories,
            territories: territories,
            rng: StdRng::from_seed(seed),
            dice_uniform: rand::distributions::Uniform::from(0..6),
        };

        println!("{:?}", board);
        board.assert_state();
        board
    }


    #[test]
    fn test_attack_mechanics() {
        assert_eq!(attack_mechanics(3, 2, [5, 5, 5, 5, 5]), (2, 0));
        assert_eq!(attack_mechanics(3, 2, [5, 5, 5, 4, 4]), (0, 2));
        assert_eq!(attack_mechanics(1, 1, [5, 0, 0, 5, 0]), (1, 0));
        assert_eq!(attack_mechanics(1, 2, [4, 0, 0, 3, 5]), (1, 0));
        assert_eq!(attack_mechanics(0, 2, [4, 0, 0, 3, 5]), (0, 0));
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

    #[test]
    fn test_fortify_fails_across_enemy_lines() {
        let mut board = setup_board(2, 3, [0; 32]);
        let old_state = board.territories;
        board.fortify(0, 1, 1);
        for i in 0..N_MAX_TERRITORIES {
            assert_eq!(old_state[i].army_count, board.territories[i].army_count);
            assert_eq!(old_state[i].owner, board.territories[i].owner);
        }
    }

    #[test]
    fn test_fortify_too_many() {
        let mut board = load_board(&[(5, 0), (1, 1), (1, 0)], [0; 32]);
        assert_eq!(board.territories[0].army_count, 5);
        board.fortify(0, 2, 20);
        assert_eq!(board.territories[0].army_count, 1);
        assert_eq!(board.territories[2].army_count, 5);
    }
}
