use crate::{
    board::{
        position::{Position, Move, Color, PieceType, MoveType},
        zobrist::{ZOBRIST},
    },
    eval::{
        evaluate::{evaluate, evaluate_fast, Score, MATE_VALUE, DRAW_VALUE, evaluate_int,nnue_push_move, nnue_pop_move,nnue_refresh},
        material::calculate_phase,
        eval_util,
    },
    movegen::{
        legal_moves::{generate_legal_moves, move_to_uci},
        magic::{all_attacks, get_bishop_attacks, get_rook_attacks, get_queen_attacks,
                get_knight_attacks, get_king_attacks, get_pawn_attacks, all_attacks_for_king},
    },
    search::{
        time_management::TimeManager,
        transposition::{TranspositionTable, TT_BOUND_EXACT, TT_BOUND_LOWER, TT_BOUND_UPPER, TTData},
        pruning::*,
    },
    nnue::{nnue::NNUE},
};

use crate::iron::model::get_best_moves;

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicI32, Ordering};
use std::time::Instant;
use std::cmp::{max, min};
use std::thread;

pub const MAX_PLY: i32 = 128;
const MAX_MOVES: usize = 256;
const INFINITY: i32 = 32000;
const MIN_SEARCH_DEPTH: i32 = 0;
const MATE_SCORE: i32 = MATE_VALUE - MAX_PLY;
const DRAW_SCORE: i32 = 0;

const MAX_QPLY: i32 = 10;


const ASPIRATION_WINDOW_INIT: i32 = 40;
const ASPIRATION_WINDOW_MAX: i32 = 400;


const LMR_BASE: f64 = 0.50;
const LMR_FACTOR: f64 = 2.0;


const HISTORY_MAX: i32 = 16384;
const HISTORY_BONUS_MAX: i32 = 2048;
const HISTORY_MALUS_MAX: i32 = -2048;

const COUNTERMOVE_HISTORY_MAX: i32 = 16384;
const COUNTERMOVE_HISTORY_BONUS: i32 = 1024;


const NULL_MOVE_REDUCTION: i32 = 2;
const NULL_MOVE_DEPTH_REDUCTION: i32 = 4;
const NULL_MOVE_VERIFICATION_DEPTH: i32 = 12;


const HASH_MOVE_SCORE: i32 = 2_000_000;        
const GOOD_CAPTURE_BASE: i32 = 1_500_000;      
const KILLER_MOVE_SCORE_1: i32 = 1_100_000;    
const KILLER_MOVE_SCORE_2: i32 = 1_050_000;    
const COUNTER_MOVE_SCORE: i32 = 1_000_000;     
const POLICY_TOP1_SCORE: i32 = 900_000;        
const POLICY_TOP2_SCORE: i32 = 850_000;        
const POLICY_TOP3_SCORE: i32 = 800_000;        
const QUIET_BASE_SCORE: i32 = 0;               
const BAD_CAPTURE_BASE: i32 = -500_000;        

const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 10000];


const POLICY_PRUNE_THRESHOLD: f32 = 0.002;
const POLICY_PRUNE_MIN_MOVES: usize = 12;
const POLICY_LMR_BONUS_THRESHOLD: f32 = 0.20;
const POLICY_EXTENSION_THRESHOLD: f32 = 0.80;
const POLICY_CONFIDENCE_HIGH: f32 = 0.60;




#[derive(Clone)]
pub struct PVTable {
    table: [[Move; MAX_PLY as usize]; MAX_PLY as usize],
    length: [usize; MAX_PLY as usize],
}

impl PVTable {
    pub fn new() -> Self {
        Self {
            table: [[Move::null(); MAX_PLY as usize]; MAX_PLY as usize],
            length: [0; MAX_PLY as usize],
        }
    }
    
    
    #[inline(always)]
    pub fn init_ply(&mut self, ply: usize) {
        self.length[ply] = 0;
        if ply + 1 < MAX_PLY as usize {
            self.length[ply + 1] = 0;  
        }
    }
    
    
    #[inline(always)]
    pub fn update(&mut self, ply: usize, mv: Move) {
        self.table[ply][0] = mv;
        
        
        let child_len = if ply + 1 < MAX_PLY as usize {
            self.length[ply + 1].min(MAX_PLY as usize - ply - 2)
        } else {
            0
        };
        
        for i in 0..child_len {
            self.table[ply][i + 1] = self.table[ply + 1][i];
        }
        
        self.length[ply] = 1 + child_len;
    }
    
    
    pub fn get_pv(&self) -> Vec<Move> {
        let len = self.length[0].min(MAX_PLY as usize);
        let mut pv = Vec::with_capacity(len);
        for i in 0..len {
            let mv = self.table[0][i];
            if mv == Move::null() {
                break;
            }
            pv.push(mv);
        }
        pv
    }
}

impl Default for PVTable {
    fn default() -> Self {
        Self::new()
    }
}




#[derive(Clone, Debug)]
pub struct PolicyData {
    pub moves: Vec<(Move, f32)>,
    pub top_move_prob: f32,
    pub entropy: f32,
    pub is_tactical: bool,
}

impl PolicyData {
    pub fn new(moves: Vec<(Move, f32)>) -> Self {
        let top_move_prob = moves.first().map(|(_, p)| *p).unwrap_or(0.0);
        
        let entropy = moves.iter()
            .filter(|(_, p)| *p > 0.001)
            .map(|(_, p)| -p * p.ln())
            .sum::<f32>();
        
        let is_tactical = top_move_prob > 0.6 || entropy < 1.0;
        
        Self {
            moves,
            top_move_prob,
            entropy,
            is_tactical,
        }
    }
    
    #[inline(always)]
    pub fn get_prob(&self, mv: &Move) -> Option<f32> {
        self.moves.iter()
            .find(|(m, _)| m == mv)
            .map(|(_, p)| *p)
    }
    
    #[inline(always)]
    pub fn get_rank(&self, mv: &Move) -> Option<usize> {
        self.moves.iter()
            .position(|(m, _)| m == mv)
    }
    
    #[inline(always)]
    pub fn has_singular_policy(&self) -> bool {
        self.top_move_prob >= POLICY_EXTENSION_THRESHOLD
    }
    
    #[inline(always)]
    pub fn should_prune(&self, mv: &Move, moves_searched: usize) -> bool {
        if moves_searched < POLICY_PRUNE_MIN_MOVES {
            return false;
        }
        
        match self.get_prob(mv) {
            Some(p) => p < POLICY_PRUNE_THRESHOLD && self.top_move_prob > 0.3,
            None => false 
        }
    }
}

impl Default for PolicyData {
    fn default() -> Self {
        Self {
            moves: Vec::new(),
            top_move_prob: 0.0,
            entropy: f32::MAX,
            is_tactical: false,
        }
    }
}




#[derive(Default)]
pub struct SearchStats {
    pub nodes: AtomicU64,
    pub qnodes: AtomicU64,
    pub tt_hits: AtomicU64,
    pub tt_cuts: AtomicU64,
    pub null_cuts: AtomicU64,
    pub lmr_reductions: AtomicU64,
    pub pruned_moves: AtomicU64,
    pub policy_prunes: AtomicU64,
    pub policy_extensions: AtomicU64,
    pub history_prunes: AtomicU64,
    pub probcut_prunes: AtomicU64,
}

impl SearchStats {
    fn new() -> Self {
        Self::default()
    }
    
    fn clear(&self) {
        self.nodes.store(0, Ordering::Relaxed);
        self.qnodes.store(0, Ordering::Relaxed);
        self.tt_hits.store(0, Ordering::Relaxed);
        self.tt_cuts.store(0, Ordering::Relaxed);
        self.null_cuts.store(0, Ordering::Relaxed);
        self.lmr_reductions.store(0, Ordering::Relaxed);
        self.pruned_moves.store(0, Ordering::Relaxed);
        self.policy_prunes.store(0, Ordering::Relaxed);
        self.policy_extensions.store(0, Ordering::Relaxed);
        self.history_prunes.store(0, Ordering::Relaxed);
        self.probcut_prunes.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
    pub hashfull: u32,
    pub multi_pv_results: Vec<MultiPVLine>,
}


#[derive(Debug, Clone)]
pub struct MultiPVLine {
    pub pv_index: usize,
    pub score: i32,
    pub pv: Vec<Move>,
}




pub struct SharedHistoryTables {
    quiet_history: Box<[[[AtomicI32; 64]; 64]; 2]>,
    capture_history: Box<[[[[AtomicI32; 7]; 64]; 7]; 2]>,
    countermove_history: Box<[[[[AtomicI32; 64]; 7]; 64]; 7]>,
}

impl SharedHistoryTables {
    pub fn new() -> Self {
        Self {
            quiet_history: Box::new(std::array::from_fn(|_| 
                std::array::from_fn(|_| 
                    std::array::from_fn(|_| AtomicI32::new(0))
                )
            )),
            capture_history: Box::new(std::array::from_fn(|_| 
                std::array::from_fn(|_| 
                    std::array::from_fn(|_| 
                        std::array::from_fn(|_| AtomicI32::new(0))
                    )
                )
            )),
            countermove_history: Box::new(std::array::from_fn(|_| 
                std::array::from_fn(|_| 
                    std::array::from_fn(|_| 
                        std::array::from_fn(|_| AtomicI32::new(0))
                    )
                )
            )),
        }
    }
    
    pub fn clear(&self) {
        for color in 0..2 {
            for from in 0..64 {
                for to in 0..64 {
                    self.quiet_history[color][from][to].store(0, Ordering::Relaxed);
                }
            }
        }
        for color in 0..2 {
            for piece in 0..7 {
                for to in 0..64 {
                    for captured in 0..7 {
                        self.capture_history[color][piece][to][captured].store(0, Ordering::Relaxed);
                    }
                }
            }
        }
        for pp in 0..7 {
            for pt in 0..64 {
                for cp in 0..7 {
                    for ct in 0..64 {
                        self.countermove_history[pp][pt][cp][ct].store(0, Ordering::Relaxed);
                    }
                }
            }
        }
    }
    
    #[inline(always)]
    pub fn update_quiet_history(&self, color: Color, from: u8, to: u8, bonus: i32) {
        let h = &self.quiet_history[color as usize][from as usize][to as usize];
        let old = h.load(Ordering::Relaxed);
        let new = old + bonus - (old * bonus.abs() / HISTORY_MAX);
        let clamped = new.clamp(HISTORY_MALUS_MAX, HISTORY_BONUS_MAX);
        h.store(clamped, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn update_capture_history(&self, color: Color, piece: PieceType, to: u8, captured: PieceType, bonus: i32) {
        let h = &self.capture_history[color as usize][piece as usize][to as usize][captured as usize];
        let old = h.load(Ordering::Relaxed);
        let new = old + bonus - (old * bonus.abs() / HISTORY_MAX);
        let clamped = new.clamp(HISTORY_MALUS_MAX, HISTORY_BONUS_MAX);
        h.store(clamped, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn get_quiet_history(&self, color: Color, from: u8, to: u8) -> i32 {
        self.quiet_history[color as usize][from as usize][to as usize].load(Ordering::Relaxed)
    }
    
    #[inline(always)]
    pub fn get_capture_history(&self, color: Color, piece: PieceType, to: u8, captured: PieceType) -> i32 {
        self.capture_history[color as usize][piece as usize][to as usize][captured as usize].load(Ordering::Relaxed)
    }
    
    #[inline(always)]
    pub fn update_countermove_history(&self, prev_piece: PieceType, prev_to: u8, curr_piece: PieceType, curr_to: u8, bonus: i32) {
        if prev_piece == PieceType::None || curr_piece == PieceType::None {
            return;
        }
        let h = &self.countermove_history[prev_piece as usize][prev_to as usize][curr_piece as usize][curr_to as usize];
        let old = h.load(Ordering::Relaxed);
        let new = old + bonus - (old * bonus.abs() / COUNTERMOVE_HISTORY_MAX);
        let clamped = new.clamp(-COUNTERMOVE_HISTORY_MAX, COUNTERMOVE_HISTORY_MAX);
        h.store(clamped, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn get_countermove_history(&self, prev_piece: PieceType, prev_to: u8, curr_piece: PieceType, curr_to: u8) -> i32 {
        if prev_piece == PieceType::None || curr_piece == PieceType::None {
            return 0;
        }
        self.countermove_history[prev_piece as usize][prev_to as usize][curr_piece as usize][curr_to as usize].load(Ordering::Relaxed)
    }
}

static SHARED_HISTORY: once_cell::sync::Lazy<SharedHistoryTables> = once_cell::sync::Lazy::new(|| SharedHistoryTables::new());


struct HistoryTables {
    counter_moves: [[[Move; 64]; 7]; 2],
    killers: [[Move; 2]; MAX_PLY as usize],
    
    move_piece: [PieceType; MAX_PLY as usize],
    move_to: [u8; MAX_PLY as usize],
}

impl HistoryTables {
    fn new() -> Self {
        Self {
            counter_moves: [[[Move::null(); 64]; 7]; 2],
            killers: [[Move::null(); 2]; MAX_PLY as usize],
            move_piece: [PieceType::None; MAX_PLY as usize],
            move_to: [0; MAX_PLY as usize],
        }
    }
    
    #[inline(always)]
    fn update_killers(&mut self, ply: usize, mv: Move) {
        if ply < MAX_PLY as usize && self.killers[ply][0] != mv {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = mv;
        }
    }
    
    #[inline(always)]
    fn update_counter_move(&mut self, color: Color, piece: PieceType, to: u8, counter: Move) {
        if piece != PieceType::None {
            self.counter_moves[color as usize][piece as usize][to as usize] = counter;
        }
    }
    
    #[inline(always)]
    fn get_counter_move(&self, color: Color, piece: PieceType, to: u8) -> Move {
        if piece == PieceType::None {
            Move::null()
        } else {
            self.counter_moves[color as usize][piece as usize][to as usize]
        }
    }
    
    #[inline(always)]
    fn save_move_info(&mut self, ply: usize, piece: PieceType, to: u8) {
        if ply < MAX_PLY as usize {
            self.move_piece[ply] = piece;
            self.move_to[ply] = to;
        }
    }
    
    #[inline(always)]
    fn get_prev_move_info(&self, ply: usize) -> (PieceType, u8) {
        if ply > 0 && ply <= MAX_PLY as usize {
            (self.move_piece[ply - 1], self.move_to[ply - 1])
        } else {
            (PieceType::None, 0)
        }
    }
}

use once_cell::sync::Lazy;

static LMR_TABLE: Lazy<[[i32; 64]; 64]> = Lazy::new(|| {
    let mut table = [[0i32; 64]; 64];
    for d in 1..64 {
        for m in 1..64 {
            table[d][m] = ((d as f64).sqrt() + (m as f64).sqrt() - 1.0) as i32;
        }
    }
    table
});


#[inline(always)]
fn is_loss_score(score: i32) -> bool {
    score < -MATE_SCORE + MAX_PLY
}


#[inline(always)]
fn is_win_score(score: i32) -> bool {
    score > MATE_SCORE - MAX_PLY
}


#[inline(always)]
fn is_mate_score(score: i32) -> bool {
    score.abs() > MATE_SCORE - MAX_PLY
}

#[inline(always)]
fn lmr_reduction(
    depth: i32, 
    moves_searched: usize, 
    improving: bool, 
    is_pv: bool,
    _is_cut_node: bool,
    _is_tt_move: bool,
    history_score: i32,
    _gives_check: bool,
) -> i32 {
    if depth < 3 || moves_searched < 2 {
        return 0;
    }
    
    let mut r = LMR_TABLE[depth.min(63) as usize][moves_searched.min(63)];
    
    if is_pv {
        r -= 1;
    }
    
    if improving {
        r -= 1;
    }
    
    r -= (history_score / 5000).clamp(-2, 2);
    
    r.max(0).min(depth - 2)
}



#[inline(always)]
pub fn see(pos: &Position, mv: Move) -> i32 {
    let from = mv.from();
    let to = mv.to();
    let (moving_piece, moving_color) = pos.piece_at(from);
    let (captured_piece, _) = pos.piece_at(to);
    
    let captured_value = if mv.move_type() == MoveType::EnPassant {
        PIECE_VALUES[PieceType::Pawn as usize]
    } else {
        PIECE_VALUES[captured_piece as usize]
    };
    
    
    if captured_value > PIECE_VALUES[moving_piece as usize] + 50 {
        return captured_value - PIECE_VALUES[moving_piece as usize];
    }
    
    let promotion_bonus = if mv.is_promotion() {
        PIECE_VALUES[mv.promotion() as usize] - PIECE_VALUES[PieceType::Pawn as usize]
    } else {
        0
    };
    
    if captured_piece == PieceType::None && promotion_bonus == 0 && mv.move_type() != MoveType::EnPassant {
        return 0;
    }
    
    let mut gain_stack = [0i32; 32];
    let mut depth = 0;
    let mut occupied = pos.all_pieces();
    let mut color = moving_color;
    
    gain_stack[depth] = captured_value + promotion_bonus;
    occupied ^= 1u64 << from;
    
    loop {
        depth += 1;
        if depth >= 32 {
            break;
        }
        
        color = color.opposite();
        
        let attackers = get_attackers_to_square(pos, to, occupied) & pos.pieces(color) & occupied;
        
        if attackers == 0 {
            break;
        }
        
        let (attacker_square, attacker_value) = find_least_attacker(pos, attackers, color);
        
        if attacker_square == 64 {
            break;
        }
        
        gain_stack[depth] = attacker_value - gain_stack[depth - 1];
        
        
        if attacker_value >= PIECE_VALUES[PieceType::King as usize] {
            if pos.pieces(color.opposite()) & (1u64 << to) != 0 {
                
            }
            break;
        }
        
        occupied ^= 1u64 << attacker_square;
    }
    
    
    while depth > 0 {
        depth -= 1;
        gain_stack[depth] = -(-gain_stack[depth]).max(gain_stack[depth + 1]);
    }
    
    gain_stack[0]
}

#[inline(always)]
fn get_attackers_to_square(pos: &Position, square: u8, occupied: u64) -> u64 {
    let mut attackers = 0u64;
    
    attackers |= get_pawn_attacks(square, Color::Black) & pos.pieces_colored(PieceType::Pawn, Color::White);
    attackers |= get_pawn_attacks(square, Color::White) & pos.pieces_colored(PieceType::Pawn, Color::Black);
    attackers |= get_knight_attacks(square) & pos.pieces_of_type(PieceType::Knight);
    
    let bishop_attacks = get_bishop_attacks(square, occupied);
    attackers |= bishop_attacks & (pos.pieces_of_type(PieceType::Bishop) | pos.pieces_of_type(PieceType::Queen));
    
    let rook_attacks = get_rook_attacks(square, occupied);
    attackers |= rook_attacks & (pos.pieces_of_type(PieceType::Rook) | pos.pieces_of_type(PieceType::Queen));
    
    attackers |= get_king_attacks(square) & pos.pieces_of_type(PieceType::King);
    
    attackers
}

#[inline(always)]
fn find_least_attacker(pos: &Position, attackers: u64, color: Color) -> (u8, i32) {
    const PIECE_ORDER: [PieceType; 6] = [
        PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
        PieceType::Rook, PieceType::Queen, PieceType::King
    ];
    
    for piece_type in PIECE_ORDER {
        let piece_attackers = attackers & pos.pieces_colored(piece_type, color);
        if piece_attackers != 0 {
            let square = piece_attackers.trailing_zeros() as u8;
            return (square, PIECE_VALUES[piece_type as usize]);
        }
    }
    (64, 0)
}

struct ScoredMove {
    mv: Move,
    score: i32,
    see_value: Option<i32>,
    policy_prob: Option<f32>,
}

struct MoveOrderer {
    moves: Vec<ScoredMove>,
    current: usize,
}

impl MoveOrderer {
    fn new(
        pos: &Position,
        tt_move: Move,
        ply: usize,
        history: &HistoryTables,
        policy_data: &PolicyData,
    ) -> Self {
        let legal_moves = generate_legal_moves(pos);
        let mut moves = Vec::with_capacity(legal_moves.len());
        
        let (prev_piece, prev_to) = history.get_prev_move_info(ply);
        let counter_move = if prev_piece != PieceType::None {
            history.get_counter_move(pos.side_to_move, prev_piece, prev_to)
        } else {
            Move::null()
        };
        
        for mv in legal_moves {
            let score = Self::score_move(
                pos, mv, tt_move, ply, history, 
                prev_piece, prev_to, counter_move, policy_data
            );
            
            let policy_prob = policy_data.get_prob(&mv);
            
            moves.push(ScoredMove {
                mv,
                score,
                see_value: None,
                policy_prob,
            });
        }
        
        Self { moves, current: 0 }
    }
    
    fn score_move(
        pos: &Position,
        mv: Move,
        tt_move: Move,
        ply: usize,
        history: &HistoryTables,
        prev_piece: PieceType,
        prev_to: u8,
        counter_move: Move,
        policy_data: &PolicyData,
    ) -> i32 {
        
        if mv == tt_move && tt_move != Move::null() {
            return HASH_MOVE_SCORE;
        }
        
        let from = mv.from();
        let to = mv.to();
        let (piece, color) = pos.piece_at(from);
        let (captured, _) = pos.piece_at(to);
        let is_capture = captured != PieceType::None || mv.move_type() == MoveType::EnPassant;
        
        
        if is_capture {
            let see_value = see(pos, mv);
            
            if see_value >= 0 {
                
                let captured_val = if mv.move_type() == MoveType::EnPassant {
                    PIECE_VALUES[PieceType::Pawn as usize]
                } else {
                    PIECE_VALUES[captured as usize]
                };
                return GOOD_CAPTURE_BASE + captured_val * 10 - PIECE_VALUES[piece as usize] + see_value;
            } else {
                
                return BAD_CAPTURE_BASE + see_value;
            }
        }
        
        
        if mv.is_promotion() {
            return GOOD_CAPTURE_BASE + PIECE_VALUES[mv.promotion() as usize];
        }
        
        
        if ply < MAX_PLY as usize {
            if mv == history.killers[ply][0] {
                return KILLER_MOVE_SCORE_1;
            }
            if mv == history.killers[ply][1] {
                return KILLER_MOVE_SCORE_2;
            }
        }
        
        
        if mv == counter_move && counter_move != Move::null() {
            return COUNTER_MOVE_SCORE;
        }
        
        
        if let Some(rank) = policy_data.get_rank(&mv) {
            let prob = policy_data.moves[rank].1;
            match rank {
                0 => return POLICY_TOP1_SCORE + (prob * 10000.0) as i32,
                1 => return POLICY_TOP2_SCORE + (prob * 10000.0) as i32,
                2 => return POLICY_TOP3_SCORE + (prob * 10000.0) as i32,
                _ => {
                    
                    let policy_bonus = (prob * 100000.0) as i32;
                    let history_score = SHARED_HISTORY.get_quiet_history(color, from, to);
                    return QUIET_BASE_SCORE + policy_bonus + history_score;
                }
            }
        }
        
        
        let history_score = SHARED_HISTORY.get_quiet_history(color, from, to);
        
        
        let cm_bonus = if prev_piece != PieceType::None {
            SHARED_HISTORY.get_countermove_history(prev_piece, prev_to, piece, to) / 2
        } else {
            0
        };
        
        QUIET_BASE_SCORE + history_score + cm_bonus
    }
    
    
    fn next(&mut self) -> Option<&ScoredMove> {
        if self.current >= self.moves.len() {
            return None;
        }
        
        
        let mut best_idx = self.current;
        let mut best_score = self.moves[self.current].score;
        
        for i in (self.current + 1)..self.moves.len() {
            if self.moves[i].score > best_score {
                best_score = self.moves[i].score;
                best_idx = i;
            }
        }
        
        
        if best_idx != self.current {
            self.moves.swap(self.current, best_idx);
        }
        
        let result = &self.moves[self.current];
        self.current += 1;
        Some(result)
    }
    
    fn len(&self) -> usize {
        self.moves.len()
    }
    
    fn remaining(&self) -> usize {
        self.moves.len() - self.current
    }
}




pub struct SearchContext {
    tt: Arc<TranspositionTable>,
    history: Arc<Mutex<HistoryTables>>,
    stats: Arc<SearchStats>,
    stop_flag: Arc<AtomicBool>,
    root_position: Position,
    start_time: Instant,
    time_manager: TimeManager,
    thread_id: usize,
    pruning_stats: PruningStats,

    static_eval_stack: [i32; MAX_PLY as usize],
    move_stack: [Move; MAX_PLY as usize],
    piece_stack: [PieceType; MAX_PLY as usize],
    exclude_move: [Move; MAX_PLY as usize],

    pv_table: PVTable,
    
    
    root_policy: PolicyData,
    
    
    hash_history: Vec<u64>,
    search_hash_stack: [u64; MAX_PLY as usize],
     
    pub excluded_root_moves: Vec<Move>,
}

impl SearchContext {
    fn new(
        tt: Arc<TranspositionTable>,
        stop_flag: Arc<AtomicBool>,
        root_position: Position,
        time_manager: TimeManager,
        thread_id: usize,
        root_policy: PolicyData,
        hash_history: Vec<u64>,
    ) -> Self {
        Self {
            tt,
            history: Arc::new(Mutex::new(HistoryTables::new())),
            stats: Arc::new(SearchStats::new()),
            stop_flag,
            root_position,
            start_time: Instant::now(),
            time_manager,
            thread_id,
            pruning_stats: PruningStats::new(),
            static_eval_stack: [0; MAX_PLY as usize],
            move_stack: [Move::null(); MAX_PLY as usize],
            piece_stack: [PieceType::None; MAX_PLY as usize],
            exclude_move: [Move::null(); MAX_PLY as usize],
            pv_table: PVTable::new(),
            root_policy,
            hash_history,
            search_hash_stack: [0; MAX_PLY as usize],
            excluded_root_moves: Vec::new(),
        }
    }
    #[inline(always)]
    fn should_stop(&self, current_depth: u32) -> bool {
        if self.stop_flag.load(Ordering::Relaxed) {
            return true;
        }
        
        let nodes = self.stats.nodes.load(Ordering::Relaxed);
        if nodes & 2047 == 0 {
            if self.time_manager.should_stop(self.start_time, current_depth) {
                self.stop_flag.store(true, Ordering::SeqCst);
                return true;
            }
        }
        false
    }
    
    fn is_repetition(&self, hash: u64, ply: i32) -> bool {
        
        let mut i = ply - 4;
        while i >= 0 {
            if self.search_hash_stack[i as usize] == hash {
                return true;
            }
            i -= 2; 
        }
        
        
        for &h in self.hash_history.iter().rev() {
            if h == hash {
                return true;
            }
        }
        
        false
    }
    
    #[inline(always)]
    fn get_policy(&self, ply: i32) -> &PolicyData {
        if ply == 0 {
            &self.root_policy
        } else {
            
            static EMPTY_POLICY: once_cell::sync::Lazy<PolicyData> = 
                once_cell::sync::Lazy::new(|| PolicyData::default());
            &EMPTY_POLICY
        }
    }
    

    
    fn alpha_beta(
    &mut self,
    pos: &mut Position,
    mut alpha: i32,
    mut beta: i32,
    mut depth: i32,
    ply: i32,
    is_pv: bool,
    skip_null: bool,
    cut_node: bool,
) -> i32 {
        
        
        if is_pv {
            self.pv_table.init_ply(ply as usize);
        }
        
        
        if ply >= MAX_PLY - 1 {
            return evaluate_fast(pos);
        }
        
        let root_node = ply == 0;
        let alpha_orig = alpha;
        
        self.stats.nodes.fetch_add(1, Ordering::Relaxed);
        
        
        if !root_node && (ply & 7) == 0 && self.should_stop(depth as u32) {
            return alpha;
        }    
        
        self.search_hash_stack[ply as usize] = pos.hash;
        
        if !root_node && self.is_repetition(pos.hash, ply) {
            return DRAW_SCORE;
        }
        
        
        if pos.halfmove_clock >= 100 {
            return DRAW_SCORE;
        }

        
        let mating_score = MATE_VALUE - ply;
        if mating_score < beta {
            beta = mating_score;
            if alpha >= mating_score {
                return mating_score;
            }
        }
        let mated_score = -MATE_VALUE + ply;
        if mated_score > alpha {
            alpha = mated_score;
            if beta <= mated_score {
                return mated_score;
            }
        }
        
        let in_check = pos.is_in_check(pos.side_to_move);
        
        
        if in_check {
            depth += 1;
        }
    
        let tt_entry = self.tt.probe(pos.hash);
        let mut tt_move = Move::null();
        let mut tt_score = 0;
        let mut tt_depth = 0;
        let mut tt_bound = 0;
        let mut tt_static_eval = -INFINITY;
        
        if let Some(ref entry) = tt_entry {
            tt_move = entry.best_move;
            tt_score = self.score_from_tt(entry.score, ply);
            tt_depth = entry.depth;
            tt_bound = entry.bound;
            tt_static_eval = entry.static_eval;
            
            self.stats.tt_hits.fetch_add(1, Ordering::Relaxed);
            
            
            if !is_pv && !root_node && tt_depth >= depth as u8 {
                match tt_bound {
                    TT_BOUND_EXACT => {
                        self.stats.tt_cuts.fetch_add(1, Ordering::Relaxed);
                        return tt_score;
                    }
                    TT_BOUND_LOWER => {
                        if tt_score >= beta {
                            self.stats.tt_cuts.fetch_add(1, Ordering::Relaxed);
                            return tt_score;
                        }
                        
                    }
                    TT_BOUND_UPPER => {
                        if tt_score <= alpha {
                            self.stats.tt_cuts.fetch_add(1, Ordering::Relaxed);
                            return tt_score;
                        }
                    }
                    _ => {}
                }
            }
        }
        
        
        if depth <= 0 {
            return self.quiescence(pos, alpha, beta, ply, 0);
        }

let static_eval = if in_check {
    
    -INFINITY
} else if tt_static_eval != -INFINITY && tt_static_eval.abs() < MATE_SCORE {
    tt_static_eval
} else {
    evaluate(pos).score
};

self.static_eval_stack[ply as usize] = static_eval;


let improving = !in_check 
    && ply >= 2 
    && static_eval > self.static_eval_stack[(ply - 2) as usize];


let side = pos.side_to_move;
let has_npm = has_non_pawn_material(pos, side);


if !is_pv && !in_check && !root_node {
    
    if depth <= 6 && static_eval - 80 * depth >= beta && static_eval.abs() < MATE_SCORE {
        return static_eval;
    }
    
    
    if !skip_null && has_npm && static_eval >= beta && depth >= 3 && beta.abs() < MATE_SCORE {
        let r = (3 + depth / 4).min(depth - 1).max(2);
        
        pos.make_null_move();
        let null_score = -self.alpha_beta(pos, -beta, -beta + 1, depth - r, ply + 1, false, true, false);
        pos.unmake_null_move();
        
        if null_score >= beta && null_score.abs() < MATE_SCORE {
            self.stats.null_cuts.fetch_add(1, Ordering::Relaxed);
            return null_score;
        }
    }
    
    
    if depth <= 2 && static_eval + 300 + 200 * depth <= alpha {
        let razor_score = self.quiescence(pos, alpha, beta, ply, 0);
        if razor_score <= alpha {
            return razor_score;
        }
    }
}


if tt_move == Move::null() && depth >= 6 {
    let iid_depth = depth - depth / 4 - 1;
    self.alpha_beta(pos, alpha, beta, iid_depth, ply, is_pv, skip_null, cut_node);
    
    if let Some(entry) = self.tt.probe(pos.hash) {
        tt_move = entry.best_move;
    }
}


let mut singular_extension = 0;

if !root_node 
    && depth >= 7
    && tt_move != Move::null()
    && tt_depth >= (depth - 3) as u8
    && (tt_bound == TT_BOUND_LOWER || tt_bound == TT_BOUND_EXACT)
    && !is_mate_score(tt_score)
    && self.exclude_move[ply as usize] == Move::null()
{
    let singular_beta = tt_score - 2 * depth;
    let singular_depth = (depth - 1) / 2;
    
    self.exclude_move[ply as usize] = tt_move;
    let singular_score = self.alpha_beta(
        pos, singular_beta - 1, singular_beta, 
        singular_depth, ply, false, true, cut_node
    );
    self.exclude_move[ply as usize] = Move::null();
    
    if singular_score < singular_beta {
        
        singular_extension = 1;
        
        
        if !is_pv && singular_score < singular_beta - depth {
            singular_extension = 2;
        }
    } else if tt_score >= beta {
        
        singular_extension = -1;
    }
}
        
        
        
        
        
        
        let policy_data = self.get_policy(ply); 
        
        let history = self.history.lock().unwrap();
        let mut move_orderer = MoveOrderer::new(pos, tt_move, ply as usize, &history, policy_data);
        drop(history);
        
        if move_orderer.len() == 0 {
            if in_check {
                return -MATE_VALUE + ply; 
            } else {
                return DRAW_SCORE; 
            }
        }
        
        let excluded = self.exclude_move[ply as usize];
        
        let mut best_move = Move::null();
        let mut best_score = -INFINITY;
        let mut moves_searched = 0;
        let mut quiet_moves_tried: Vec<Move> = Vec::with_capacity(32);
        
        while let Some(scored_move) = move_orderer.next() {
            let mv = scored_move.mv;
            let policy_prob = scored_move.policy_prob;
            
            
            if mv == excluded {
                continue;
            }
            
            
            if root_node && self.excluded_root_moves.contains(&mv) {
                continue;
            }
            
            let (moving_piece, moving_color) = pos.piece_at(mv.from());
            let (captured, _) = pos.piece_at(mv.to());
            let is_capture = captured != PieceType::None || mv.move_type() == MoveType::EnPassant;
            let gives_check = pos.gives_check(mv);
            let is_promotion = mv.is_promotion();
            
            
            
            
            


if !root_node && best_score > -MATE_SCORE + MAX_PLY && !in_check {
    
    
    if is_capture && moves_searched > 1 && depth <= 6 {
        let see_val = see(pos, mv);
        if see_val < -100 * depth {
            continue;
        }
    }
    
    
    if !is_capture && !is_promotion && !gives_check {
        
        if !is_pv && depth <= 6 && moves_searched >= (3 + depth * depth) as usize {
            continue;
        }
        
        
        if !is_pv && depth <= 5 {
            let margin = 100 + 100 * depth;
            if static_eval + margin <= alpha {
                continue;
            }
        }
        
        
        if !is_pv && depth >= 4 && moves_searched > 6 {
            let hist = SHARED_HISTORY.get_quiet_history(moving_color, mv.from(), mv.to());
            if hist < -2000 * depth {
                continue;
            }
        }
    }
}
            
            
            {
                let mut history = self.history.lock().unwrap();
                history.save_move_info(ply as usize, moving_piece, mv.to());
            }
            
            
            if !is_capture && !is_promotion {
                quiet_moves_tried.push(mv);
            }
            
            
            nnue_push_move(mv, pos);
            if !pos.make_move(mv) {
                nnue_pop_move();
                continue;
            }
            
            self.move_stack[ply as usize] = mv;
            self.piece_stack[ply as usize] = moving_piece;
            
            
            
            
            let mut score;
            
            
            let mut extension = 0;
if moves_searched == 0 && mv == tt_move && singular_extension != 0 {
    extension = singular_extension.clamp(-1, 2);
}

let new_depth = (depth - 1 + extension).max(0);

            
            if moves_searched == 0 {
    
    score = -self.alpha_beta(pos, -beta, -alpha, new_depth, ply + 1, is_pv, false, false);
} else {
    
    let mut reduction = 0;
    
    if depth >= 3 && !in_check && !is_capture && !is_promotion && !gives_check {
        let hist = SHARED_HISTORY.get_quiet_history(moving_color, mv.from(), mv.to());
        reduction = lmr_reduction(depth, moves_searched, improving, is_pv, cut_node, mv == tt_move, hist, gives_check);
    }
    
    
    let reduced_depth = (new_depth - reduction).max(0);
    score = -self.alpha_beta(pos, -alpha - 1, -alpha, reduced_depth, ply + 1, false, false, true);
    
    
    if score > alpha && score < beta {
        score = -self.alpha_beta(pos, -beta, -alpha, new_depth, ply + 1, is_pv, false, false);
    }
}
            
            pos.unmake_move(mv);
            nnue_pop_move();
            
            moves_searched += 1;
            
            
            
            
            if score > best_score {
                best_score = score;
                best_move = mv;
                
                if score > alpha {
                    alpha = score;
                    
                    
                    
                    
                    if is_pv {
                        self.pv_table.update(ply as usize, mv);
                    }
                    
                    
                    if score >= beta {
                        
                        if !is_capture && !is_promotion {
                            let bonus = (depth * depth + 8 * depth).min(400);
                            
                            let mut history = self.history.lock().unwrap();
                            history.update_killers(ply as usize, mv);
                            
                            let (prev_piece, prev_to) = history.get_prev_move_info(ply as usize);
                            if prev_piece != PieceType::None {
                                history.update_counter_move(moving_color, prev_piece, prev_to, mv);
                            }
                            drop(history);
                            
                            
                            SHARED_HISTORY.update_quiet_history(moving_color, mv.from(), mv.to(), bonus);
                            
                            if prev_piece != PieceType::None {
                                SHARED_HISTORY.update_countermove_history(
                                    prev_piece, prev_to, moving_piece, mv.to(), bonus
                                );
                            }
                            
                            
                            for &quiet_mv in &quiet_moves_tried {
                                if quiet_mv != mv {
                                    SHARED_HISTORY.update_quiet_history(
                                        moving_color, quiet_mv.from(), quiet_mv.to(), -bonus / 2
                                    );
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
        
        
        
        
        let bound = if best_score >= beta {
            TT_BOUND_LOWER
        } else if best_score <= alpha_orig {
            TT_BOUND_UPPER
        } else {
            TT_BOUND_EXACT
        };
        
        if !self.stop_flag.load(Ordering::Relaxed) {
            self.tt.store(
                pos.hash,
                best_move,
                self.score_to_tt(best_score, ply),
                static_eval,
                depth as u8,
                bound,
                ply as u8,
            );
        }
        
        best_score
    }
    
    
    
    
    fn quiescence(
        &mut self,
        pos: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        q_ply: i32,
    ) -> i32 {
        if ply >= MAX_PLY - 1 || q_ply >= MAX_QPLY {
            return evaluate_fast(pos);
        }
        
        self.stats.qnodes.fetch_add(1, Ordering::Relaxed);
        
        if pos.is_draw() {
            return DRAW_SCORE;
        }
        
        let in_check = pos.is_in_check(pos.side_to_move);
        
        
        let mut best_score = if in_check {
            -INFINITY
        } else {
            let stand_pat = evaluate_fast(pos);
            
            if stand_pat >= beta {
                return stand_pat;
            }
            
            
if stand_pat < alpha - 1200 {
    return alpha;
}
            
            if stand_pat > alpha {
                alpha = stand_pat;
            }
            
            stand_pat
        };
        
        let all_moves = generate_legal_moves(pos);
        
        if all_moves.is_empty() {
            if in_check {
                return -MATE_VALUE + ply;
            } else {
                return DRAW_SCORE;
            }
        }
        
        
        let mut scored_moves: Vec<(Move, i32)> = all_moves.into_iter()
            .filter_map(|mv| {
                let (captured, _) = pos.piece_at(mv.to());
                let is_capture = captured != PieceType::None || mv.move_type() == MoveType::EnPassant;
                
                
                if !in_check && !is_capture && !mv.is_promotion() {
                    return None;
                }
                
                let see_val = if is_capture { see(pos, mv) } else { 0 };
                
                
                if !in_check && is_capture && see_val < -80 {

                    return None;
                }
                
                let mut score = see_val;
                if mv.is_promotion() {
                    score += PIECE_VALUES[mv.promotion() as usize];
                }
                if pos.gives_check(mv) {
                    score += 500;
                }
                
                Some((mv, score))
            })
            .collect();
        
        scored_moves.sort_unstable_by_key(|&(_, s)| -s);
        
        let max_moves = if in_check { 32 } else { 16 };
        
        for (mv, _) in scored_moves.into_iter().take(max_moves) {
            nnue_push_move(mv, pos);
            
            if !pos.make_move(mv) {
                nnue_pop_move();
                continue;
            }
            
            let score = -self.quiescence(pos, -beta, -alpha, ply + 1, q_ply + 1);
            
            pos.unmake_move(mv);
            nnue_pop_move();
            
            if score > best_score {
                best_score = score;
                
                if score > alpha {
                    alpha = score;
                    
                    if score >= beta {
                        break;
                    }
                }
            }
        }
        
        best_score
    }
    
    
    #[inline(always)]
    fn score_to_tt(&self, score: i32, ply: i32) -> i32 {
        if score >= MATE_SCORE {
            score + ply
        } else if score <= -MATE_SCORE {
            score - ply
        } else {
            score
        }
    }
    
    #[inline(always)]
    fn score_from_tt(&self, score: i32, ply: i32) -> i32 {
        if score >= MATE_SCORE {
            score - ply
        } else if score <= -MATE_SCORE {
            score + ply
        } else {
            score
        }
    }
    
    pub fn get_pv(&self) -> Vec<Move> {
        self.pv_table.get_pv()
    }
}




struct SearchWorker {
    id: usize,
    position: Position,
    tt: Arc<TranspositionTable>,
    history: Arc<Mutex<HistoryTables>>,
    stop_flag: Arc<AtomicBool>,
    depth_offset: i32,
    stats: Arc<SearchStats>,
    root_policy: PolicyData,
    hash_history: Vec<u64>,
}

impl SearchWorker {
    fn run(
        &self,
        target_depth: i32,
        time_manager: TimeManager,
        result_sender: std::sync::mpsc::Sender<(i32, i32, Vec<Move>)>,
    ) {
        nnue_refresh(&self.position);
        
        let mut local_pos = self.position.clone();
        let mut best_score = -INFINITY;
        
        let start_depth = if self.id == 0 { 1 } else { 1 + (self.id as i32 % 3) };
        
        for depth in start_depth..=target_depth {
            if self.stop_flag.load(Ordering::Relaxed) {
                break;
            }
            
            let mut ctx = SearchContext::new(
                Arc::clone(&self.tt),
                Arc::clone(&self.stop_flag),
                self.position.clone(),
                time_manager.clone(),
                self.id,
                self.root_policy.clone(),
                self.hash_history.clone(),
            );
            ctx.stats = Arc::clone(&self.stats);
            ctx.history = Arc::clone(&self.history);
            
            let adjusted_depth = depth + self.depth_offset;
            
            
            let mut alpha;
            let mut beta;
            let mut delta = ASPIRATION_WINDOW_INIT;
            
            if depth >= 5 && best_score.abs() < MATE_SCORE - 100 {
                alpha = best_score - delta;
                beta = best_score + delta;
            } else {
                alpha = -INFINITY;
                beta = INFINITY;
            }
            
            let mut fail_count = 0;
            
            loop {
                let score = ctx.alpha_beta(&mut local_pos, alpha, beta, adjusted_depth, 0, true, false, false);

                
                if self.stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                
                fail_count += 1;
                
                if fail_count > 8 {
                    
                    let score = ctx.alpha_beta(&mut local_pos, -INFINITY, INFINITY, adjusted_depth, 0, true, false, false);
                    best_score = score;
                    let pv = ctx.get_pv();
                    let _ = result_sender.send((adjusted_depth, score, pv));
                    break;
                }
                
                if score <= alpha {
                    
        beta = alpha;  
                    alpha = (score - delta).max(-INFINITY);
                    delta += delta / 2;
                } else if score >= beta {
                    
                    alpha = (alpha + beta) / 2;
                    beta = (score + delta).min(INFINITY);
                    delta += delta / 2;
                } else {
                    
                    best_score = score;
                    let pv = ctx.get_pv();
                    let _ = result_sender.send((adjusted_depth, score, pv));
                    break;
                }
                
                if delta > ASPIRATION_WINDOW_MAX {
                    alpha = -INFINITY;
                    beta = INFINITY;
                }
            }
        }
    }
}




pub struct ParallelSearch {
    tt: Arc<TranspositionTable>,
    pub stop_flag: Arc<AtomicBool>,
    thread_count: usize,
    global_history: Arc<Mutex<HistoryTables>>,
    global_stats: Arc<SearchStats>,
    hash_history: Vec<u64>,
}

impl ParallelSearch {
    pub fn new(thread_count: usize, tt_size_mb: usize) -> Self {
        Self {
            tt: Arc::new(TranspositionTable::new(tt_size_mb)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            thread_count: thread_count.max(1),
            global_history: Arc::new(Mutex::new(HistoryTables::new())),
            global_stats: Arc::new(SearchStats::new()),
            hash_history: Vec::with_capacity(512),
        }
    }
    
    pub fn push_hash(&mut self, hash: u64) {
        self.hash_history.push(hash);
    }
    
    pub fn clear_hash_history(&mut self) {
        self.hash_history.clear();
    }
    
    pub fn search(&mut self, pos: &Position, max_depth: i32, time_manager: TimeManager) -> SearchResult {
        self.stop_flag.store(false, Ordering::SeqCst);
        self.tt.new_search();
        self.global_stats.clear();
        
        nnue_refresh(pos);
        
        
        let ai_moves = get_best_moves(pos, 15);
        let root_policy = PolicyData::new(ai_moves);
        
        #[cfg(debug_assertions)]
        {
            println!("info string Policy top={:.1}% entropy={:.2} tactical={}", 
                     root_policy.top_move_prob * 100.0,
                     root_policy.entropy,
                     root_policy.is_tactical);
        }
        
        let start_time = Instant::now();
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        
        let mut worker_handles = Vec::new();
        
        for thread_id in 0..self.thread_count {
            let worker = SearchWorker {
                id: thread_id,
                position: pos.clone(),
                tt: Arc::clone(&self.tt),
                history: Arc::clone(&self.global_history),
                stop_flag: Arc::clone(&self.stop_flag),
                depth_offset: if thread_id < 4 { 0 } else { (thread_id as i32 / 4) },
                stats: Arc::clone(&self.global_stats),
                root_policy: root_policy.clone(),
                hash_history: self.hash_history.clone(),
            };
            
            let thread_result_sender = result_sender.clone();
            let thread_time_manager = time_manager.clone();
            let thread_max_depth = max_depth;
            
            let handle = thread::spawn(move || {
                worker.run(thread_max_depth, thread_time_manager, thread_result_sender);
            });
            
            worker_handles.push(handle);
        }
        
        drop(result_sender);
        
        let mut best_result = SearchResult {
            best_move: Move::null(),
            score: 0,
            depth: 0,
            nodes: 0,
            time_ms: 0,
            pv: Vec::new(),
            hashfull: 0,
            multi_pv_results: Vec::new(),
        };
        
        let mut last_info_time = Instant::now();
        let info_interval = std::time::Duration::from_millis(500);
        
        loop {
            if time_manager.should_stop(start_time, best_result.depth as u32) && best_result.depth >= 4 {
                self.stop_flag.store(true, Ordering::SeqCst);
                break;
            }
            
            match result_receiver.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok((depth, score, pv)) => {
                    if depth > best_result.depth || (depth == best_result.depth && score > best_result.score) {
                        let elapsed = start_time.elapsed().as_millis() as u64;
                        let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
                        let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
                        let total_nodes = nodes + qnodes;
                        let nps = if elapsed > 0 { total_nodes * 1000 / elapsed } else { 0 };
                        
                        best_result = SearchResult {
                            best_move: pv.first().copied().unwrap_or(Move::null()),
                            score,
                            depth,
                            nodes: total_nodes,
                            time_ms: elapsed,
                            pv: pv.clone(),
                            hashfull: self.tt.hashfull(),
                            multi_pv_results: Vec::new(),
                        };
                        
                        self.send_uci_info(depth, score, total_nodes, elapsed, nps, &pv, 0, 1);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if last_info_time.elapsed() > info_interval && best_result.depth > 0 {
                        let elapsed = start_time.elapsed().as_millis() as u64;
                        let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
                        let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
                        let total_nodes = nodes + qnodes;
                        let nps = if elapsed > 0 { total_nodes * 1000 / elapsed } else { 0 };
                        
                        self.send_uci_info(
                            best_result.depth, best_result.score, total_nodes, 
                            elapsed, nps, &best_result.pv, 1, 1
                        );
                        
                        last_info_time = Instant::now();
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
            
            if self.stop_flag.load(Ordering::Relaxed) {
                break;
            }
        }
        
        self.stop_flag.store(true, Ordering::SeqCst);
        
        for handle in worker_handles {
            let _ = handle.join();
        }
        
        let elapsed = start_time.elapsed().as_millis() as u64;
        let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
        let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
        let total_nodes = nodes + qnodes;
        
        best_result.nodes = total_nodes;
        best_result.time_ms = elapsed;
        
        #[cfg(debug_assertions)]
        {
            let nps = if elapsed > 0 { total_nodes * 1000 / elapsed } else { 0 };
            println!("info string total nodes={} qnodes={} time={} nps={}", 
                     nodes, qnodes, elapsed, nps);
        }
        
        best_result
    }
    
    fn send_uci_info(&self, depth: i32, score: i32, nodes: u64, time_ms: u64, nps: u64, pv: &[Move], currmovenumber: u32, multipv: usize) {
        let score_str = if score.abs() >= MATE_SCORE {
            let mate_in = (MATE_VALUE - score.abs() + 1) / 2;
            format!("mate {}", if score > 0 { mate_in } else { -mate_in })
        } else {
            format!("cp {}", score)
        };
        
        let pv_str = pv.iter()
            .map(|&mv| move_to_uci(mv))
            .collect::<Vec<_>>()
            .join(" ");
        
        let hashfull = self.tt.hashfull();
        
        if currmovenumber > 0 {
            println!(
                "info depth {} seldepth {} multipv {} score {} nodes {} nps {} time {} hashfull {} currmovenumber {} pv {}",
                depth, depth + 8, multipv, score_str, nodes, nps, time_ms, hashfull, currmovenumber, pv_str
            );
        } else {
            println!(
                "info depth {} seldepth {} multipv {} score {} nodes {} nps {} time {} hashfull {} pv {}",
                depth, depth + 8, multipv, score_str, nodes, nps, time_ms, hashfull, pv_str
            );
        }
        
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
    
    pub fn clear_hash(&mut self) {
        self.tt.clear();
        *self.global_history.lock().unwrap() = HistoryTables::new();
        self.global_stats.clear();
        SHARED_HISTORY.clear();
        self.hash_history.clear();
    }
    
    
    pub fn search_multi_pv(&mut self, pos: &Position, max_depth: i32, time_manager: TimeManager, multi_pv: usize) -> SearchResult {
        
        if multi_pv <= 1 {
            return self.search(pos, max_depth, time_manager);
        }
        
        self.stop_flag.store(false, Ordering::SeqCst);
        self.tt.new_search();
        self.global_stats.clear();
        
        nnue_refresh(pos);
        
        
        let root_moves = generate_legal_moves(pos);
        let actual_multi_pv = multi_pv.min(root_moves.len());
        
        if root_moves.is_empty() {
            return SearchResult {
                best_move: Move::null(),
                score: 0,
                depth: 0,
                nodes: 0,
                time_ms: 0,
                pv: Vec::new(),
                hashfull: 0,
                multi_pv_results: Vec::new(),
            };
        }
        
        
        let ai_moves = get_best_moves(pos, 15);
        let root_policy = PolicyData::new(ai_moves);
        
        let start_time = Instant::now();
        
        let mut best_result = SearchResult {
            best_move: Move::null(),
            score: 0,
            depth: 0,
            nodes: 0,
            time_ms: 0,
            pv: Vec::new(),
            hashfull: 0,
            multi_pv_results: Vec::new(),
        };
        
        
        for depth in 1..=max_depth {
            if self.stop_flag.load(Ordering::Relaxed) {
                break;
            }
            
            if time_manager.should_stop(start_time, depth as u32) && depth >= 4 {
                break;
            }
            
            
            let mut pv_lines: Vec<MultiPVLine> = Vec::with_capacity(actual_multi_pv);
            let mut excluded_moves: Vec<Move> = Vec::with_capacity(actual_multi_pv);
            
            
            for pv_index in 0..actual_multi_pv {
                if self.stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                
                
                let mut local_pos = pos.clone();
                
                let mut ctx = SearchContext::new(
                    Arc::clone(&self.tt),
                    Arc::clone(&self.stop_flag),
                    pos.clone(),
                    time_manager.clone(),
                    0,
                    root_policy.clone(),
                    self.hash_history.clone(),
                );
                ctx.stats = Arc::clone(&self.global_stats);
                ctx.history = Arc::clone(&self.global_history);
                ctx.excluded_root_moves = excluded_moves.clone();
                
                
                let mut alpha = -INFINITY;
                let mut beta = INFINITY;
                let mut delta = ASPIRATION_WINDOW_INIT;
                
                
                if pv_index == 0 && depth >= 5 && best_result.score.abs() < MATE_SCORE - 100 {
                    alpha = best_result.score - delta;
                    beta = best_result.score + delta;
                } else if pv_index > 0 && !pv_lines.is_empty() {
                    
                    let prev_score = pv_lines.last().map(|l| l.score).unwrap_or(0);
                    if prev_score.abs() < MATE_SCORE - 100 {
                        alpha = prev_score - 200 - delta;
                        beta = prev_score + delta;
                    }
                }
                
                let mut fail_count = 0;
                let mut search_score;
                
                loop {
                    search_score = ctx.alpha_beta(&mut local_pos, alpha, beta, depth, 0, true, false, false);

                    
                    if self.stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    
                    fail_count += 1;
                    
                    if fail_count > 6 {
                        
                        search_score = ctx.alpha_beta(&mut local_pos, -INFINITY, INFINITY, depth, 0, true, false, false);

                        break;
                    }
                    
                    if search_score <= alpha {
                        beta = (alpha + beta) / 2;
                        alpha = (search_score - delta).max(-INFINITY);
                        delta += delta / 2;
                    } else if search_score >= beta {
                        alpha = (alpha + beta) / 2;
                        beta = (search_score + delta).min(INFINITY);
                        delta += delta / 2;
                    } else {
                        break;
                    }
                    
                    if delta > ASPIRATION_WINDOW_MAX {
                        alpha = -INFINITY;
                        beta = INFINITY;
                    }
                }
                
                if self.stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                
                let pv = ctx.get_pv();
                if let Some(&first_move) = pv.first() {
                    if first_move != Move::null() {
                        
                        pv_lines.push(MultiPVLine {
                            pv_index: pv_index + 1,
                            score: search_score,
                            pv: pv.clone(),
                        });
                        
                        
                        excluded_moves.push(first_move);
                        
                        
                        let elapsed = start_time.elapsed().as_millis() as u64;
                        let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
                        let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
                        let total_nodes = nodes + qnodes;
                        let nps = if elapsed > 0 { total_nodes * 1000 / elapsed } else { 0 };
                        
                        self.send_uci_info(depth, search_score, total_nodes, elapsed, nps, &pv, 0, pv_index + 1);
                    }
                }
            }
            
            
            if !pv_lines.is_empty() {
                let elapsed = start_time.elapsed().as_millis() as u64;
                let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
                let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
                let total_nodes = nodes + qnodes;
                
                best_result = SearchResult {
                    best_move: pv_lines[0].pv.first().copied().unwrap_or(Move::null()),
                    score: pv_lines[0].score,
                    depth,
                    nodes: total_nodes,
                    time_ms: elapsed,
                    pv: pv_lines[0].pv.clone(),
                    hashfull: self.tt.hashfull(),
                    multi_pv_results: pv_lines,
                };
            }
        }
        
        let elapsed = start_time.elapsed().as_millis() as u64;
        let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
        let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
        let total_nodes = nodes + qnodes;
        
        best_result.nodes = total_nodes;
        best_result.time_ms = elapsed;
        
        best_result
    }
}

pub fn see_ge_threshold(pos: &Position, mv: Move, threshold: i32) -> bool {
    see(pos, mv) >= threshold
}

pub fn alpha_beta_search(
    pos: &mut Position,
    depth: i32,
    time_manager: TimeManager,
    tt: Arc<TranspositionTable>,
    stop_flag: Arc<AtomicBool>,
) -> SearchResult {
    let mut search = ParallelSearch::new(1, 256);
    search.tt = tt;
    search.stop_flag = stop_flag;
    search.search(pos, depth, time_manager)
}