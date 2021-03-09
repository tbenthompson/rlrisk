/* Plan:
 * - simple Risk engine
 * - try some kind of reinforcement learning algorithm
 * - vectorized/matrixify the risk engine so that we run many games at once.
 * - GPU-ify the risk engine?
 * - play with fancier methods
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
use rand::Rng;

const N_MAX_CARDS: usize = 0;
const N_MAX_TERRITORIES: usize = 3;
const N_MAX_PLAYERS: usize = 2;
const N_ATTACKS_PER_TURN: usize = 2;

#[derive(Debug, Clone, Copy)]
struct Territory {
    army_count: usize,
    owner: usize,
}

#[derive(Debug, Clone, Copy)]
struct Card {
    territory_idx: usize,
    color: i32,
}

#[derive(Debug, Clone, Copy)]
struct Player {
    n_controlled: usize,
    cards: [Card; N_MAX_CARDS],
}

#[derive(Debug)]
struct Board {
    n_players: usize,
    players: [Player; N_MAX_PLAYERS],

    n_territories: usize,
    territories: [Territory; N_MAX_TERRITORIES],

    rng: rand::rngs::ThreadRng,
}

impl Board {
    fn reinforce(&mut self, territory_idx: usize, n_armies: usize) {
        self.territories[territory_idx].army_count += n_armies;
    }

    fn attack(&mut self, from: usize, to: usize) -> (usize, usize, bool) {
        let t = &mut self.territories;
        let n_attacker = std::cmp::min(t[from].army_count - 1, 3);
        let n_defender = std::cmp::min(t[to].army_count, 2);
        let (attacker_deaths, defender_deaths) =
            attack_mechanics(&mut self.rng, n_attacker, n_defender);
        t[from].army_count -= attacker_deaths;
        t[to].army_count -= defender_deaths;
        let conquer = t[to].army_count == 0;
        if conquer {
            self.players[t[to].owner].n_controlled -= 1;
            self.players[t[from].owner].n_controlled += 1;
            t[to].owner = t[from].owner;
        }
        (attacker_deaths, defender_deaths, conquer)
    }

    fn fortify(&mut self, from: usize, to: usize, n_armies: usize) {
        assert!(n_armies < self.territories[from].army_count);
        self.territories[from].army_count -= n_armies;
        self.territories[to].army_count += n_armies;
    }

    fn new_card(&mut self, player_idx: usize) {
        let mut p = &mut self.players[player_idx];
        for i in 0..N_MAX_CARDS {
            if p.cards[i].territory_idx == N_MAX_TERRITORIES {
                p.cards[i].territory_idx = self.rng.gen_range(0..self.n_territories);
                p.cards[i].color = self.rng.gen_range(0..3);
                break;
            }
        }
    }
}

fn attack_mechanics(
    rng: &mut rand::rngs::ThreadRng,
    n_attacker: usize,
    n_defender: usize,
) -> (usize, usize) {
    assert!(n_attacker <= 3);
    assert!(n_defender <= 2);

    if n_attacker == 0 {
        return (0, 0);
    }

    let mut attacker_dice = (0..n_attacker)
        .map(|_| -> i32 { rng.gen_range(0..6) })
        .collect::<Vec<i32>>();
    let mut defender_dice = (0..n_defender)
        .map(|_| -> i32 { rng.gen_range(0..6) })
        .collect::<Vec<i32>>();
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

fn setup_board() -> Board {
    let n_starting_armies = 3;
    let n_players = N_MAX_PLAYERS;
    let n_territories = N_MAX_TERRITORIES;
    let mut board = Board {
        n_players: n_players,
        players: [Player {
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

        rng: rand::thread_rng(),
    };

    for k in 0..n_territories {
    for _i in 0..n_starting_armies {
        for j in 0..n_players {
            for k in 0..n_territories {
                let t = &mut board.territories[k];
                if t.owner == N_MAX_PLAYERS {
                    board.players[j].n_controlled += 1;
                    t.owner = j;
                }
                if t.owner == j {
                    t.army_count += 1;
                    break;
                }
            }
        }
    }

    board
}

trait Strategy {
    /*
     * A reinforce is specified by the (territory_idx, n_armies) referring to the territory we 
     * reinforce and the number to place at that location.
     */
    fn reinforce_step(&self, board: &Board, n_reinforcements: usize) -> (usize, usize);

    /* An attack is specified by the (from, to) tuple referring to the territory we attack from and
     * the territory under attack.
     * Returning (N_MAX_TERRITORIES, N_MAX_TERRITORIES) implies no attack.
     */
    fn attack_step(&self, _board: &Board) -> (usize, usize);

    /* not allowed for now */
    fn fortify_step(&self, _board: &Board) {}
}

#[derive(Clone)]
struct Dumb {
    player_idx: usize,
}

impl Strategy for Dumb {
    fn reinforce_step(&self, board: &Board, n_reinforcements: usize) -> (usize, usize) {
        for t_i in 0..board.n_territories {
            if board.territories[t_i].owner == self.player_idx {
                return (t_i, n_reinforcements);
            }
        }
        assert!(false);
        return (0, 0);
    }

    fn attack_step(&self, board: &Board) -> (usize, usize) {
        for t_i in 0..board.n_territories {
            if board.territories[t_i].owner == self.player_idx {
                return (t_i, 1 - t_i);
            }
        }
        return (N_MAX_TERRITORIES, N_MAX_TERRITORIES);
    }
}

fn game() -> Board {
    let mut board = setup_board();
    let players = vec![Dumb { player_idx: 0 }, Dumb { player_idx: 1 }];
    let max_turns = 10;
    for turn_idx in 0..max_turns {
        for player_idx in 0..board.n_players {
            println!("Turn ({}, {}), Board: {:?}", turn_idx, player_idx, board.territories);
            let p = &players[player_idx];

            let mut n_reinforcements = 1;
            while n_reinforcements != 0 {
                let (territory_idx, mut count) = p.reinforce_step(&board, n_reinforcements);
                count = std::cmp::min(count, n_reinforcements);
                board.reinforce(territory_idx, count);
                n_reinforcements -= count;
            }

            // attack and fortify
            let mut won_an_attack = false;
            for _attack_idx in 0..N_ATTACKS_PER_TURN {
                let (from, to) = p.attack_step(&board);
                let outcome = board.attack(from, to);
                println!("Attack: {}, {}, {:?}", from, to, outcome);
                if outcome.2 {
                    if board.players[player_idx].n_controlled == board.n_territories {
                        println!("GAME OVER");
                        return board;
                    }
                    won_an_attack = true;
                    board.fortify(from, to, board.territories[from].army_count - 1);
                }
            }
            // fortify

            // add card
            if won_an_attack {
                board.new_card(player_idx);
            }
        }
    }
    return board;
}

fn main() {
    let board = game();
    println!("Board: {:?}", board.territories);
    println!("Players: {:?}", board.players);
}
