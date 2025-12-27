use crate::board::position::{Position, Move, Color, PieceType};
use crate::eval::material::calculate_phase;
use std::sync::atomic::{AtomicU64, Ordering};

//The Constant Values for Pruning are so effective on this engine. Please if you find a better constant for search contact with me :D

pub const FUTILITY_MARGINS: [i32; 8] = [
    0,    
    100,  
    200,  
    300,  
    400,  
    500,  
    600,  
    700,  
];

pub const REVERSE_FUTILITY_MARGINS: [i32; 8] = [
    0,    
    100,  
    200,  
    300,  
    400,  
    500,  
    600,  
    700,  
];

pub const LMP_COUNTS: [usize; 8] = [
    0,   
    5,   
    10,  
    16,  
    24,  
    32,  
    42,  
    54,  
];

pub const SEE_CAPTURE_THRESHOLD: i32 = -50; 


#[inline(always)]
pub fn futility_margin(depth: i32, improving: bool) -> i32 {
    if depth < 0 || depth >= 8 {
        return i32::MAX; 
    }
    
    let base_margin = FUTILITY_MARGINS[depth as usize];
    
    
    if improving {
        base_margin
    } else {
        base_margin + base_margin / 2
    }
}

#[inline(always)]
pub fn can_futility_prune(
    depth: i32,
    static_eval: i32,
    alpha: i32,
    in_check: bool,
    is_pv: bool,
) -> bool {
    
    if is_pv || in_check || depth >= 8 || depth < 1 {
        return false;
    }
    
    let margin = FUTILITY_MARGINS[depth as usize];
    static_eval + margin <= alpha
}

#[inline(always)]
pub fn should_futility_prune_move(
    static_eval: i32,
    alpha: i32,
    depth: i32,
    improving: bool,
    is_capture: bool,
    gives_check: bool,
    is_promotion: bool,
) -> bool {
    
    if is_capture || gives_check || is_promotion {
        return false;
    }
    
    let margin = futility_margin(depth, improving);
    static_eval + margin <= alpha
}


#[inline(always)]
pub fn reverse_futility_pruning(
    pos: &Position,
    depth: i32,
    beta: i32,
    static_eval: i32,
    improving: bool,
) -> Option<i32> {
    
    if depth >= 7 || depth < 1 || pos.is_in_check(pos.side_to_move) {
        return None;
    }
    
    let depth_idx = depth as usize;
    let mut margin = REVERSE_FUTILITY_MARGINS[depth_idx];
    
    
    if !improving {
        margin += margin / 3;
    }
    
    if static_eval - margin >= beta {
        Some(static_eval - margin)
    } else {
        None
    }
}


#[inline(always)]
pub fn razoring(
    pos: &Position,
    depth: i32,
    alpha: i32,
    static_eval: i32,
) -> Option<i32> {
    if depth > 2 || depth < 1 || pos.is_in_check(pos.side_to_move) {
        return None;
    }
    
    
    let razor_margin = 200 + depth * 100;
    
    if static_eval + razor_margin <= alpha {
        Some(alpha)
    } else {
        None
    }
}


#[inline(always)]
pub fn late_move_pruning(
    depth: i32,
    moves_searched: usize,
    is_pv: bool,
    improving: bool,
) -> bool {
    
    if depth >= 8 || depth < 1 {
        return false;
    }
    
    let depth_idx = depth as usize;
    let mut move_limit = LMP_COUNTS[depth_idx];
    
    
    if is_pv {
        move_limit *= 2;
    }
    
    
    if improving {
        move_limit += move_limit / 3;
    }
    
    moves_searched >= move_limit
}



#[inline(always)]
pub fn should_prune_by_see(
    see_value: i32,
    threshold: i32,
) -> bool {
    see_value < threshold
}


#[inline(always)]
pub fn should_see_prune_capture(
    see_value: i32,
    depth: i32,
) -> bool {
    
    let threshold = SEE_CAPTURE_THRESHOLD - (depth * 10);
    see_value < threshold
}




#[inline(always)]
pub fn history_pruning(
    depth: i32,
    history_score: i32,
    moves_searched: usize,
    is_capture: bool,
) -> bool {
    
    if is_capture {
        return false;
    }
    
    
    if moves_searched < 5 || depth >= 8 || depth < 1 {
        return false;
    }
    
    
    
    
    let threshold = -((depth * depth * 100) as i32);
    
    history_score < threshold
}


#[inline(always)]
pub fn history_reduction_adjustment(history_score: i32) -> i32 {
    
    
    
    (history_score / 4096).clamp(-2, 2)
}




pub fn null_move_reduction(depth: i32, static_eval: i32, beta: i32) -> i32 {
    
    let base_reduction = if depth >= 6 { 4 } else { 3 };
    
    
    let eval_margin = (static_eval - beta).max(0);
    let eval_bonus = (eval_margin / 200).min(2);
    
    
    let depth_bonus = (depth / 5).min(1);
    
    let r = base_reduction + eval_bonus + depth_bonus;
    
    
    r.min(depth - 1).max(2)
}




#[inline(always)]
pub fn enhanced_null_move_conditions(
    pos: &Position,
    depth: i32,
    beta: i32,
    static_eval: i32,
    is_pv: bool,
    has_non_pawn_material: bool,
) -> bool {
    
    if is_pv {
        return false;
    }
    
    
    if depth < 3 {
        return false;
    }
    
    
    if pos.is_in_check(pos.side_to_move) {
        return false;
    }
    
    
    if !has_non_pawn_material {
        return false;
    }
    
    
    static_eval >= beta
}


#[inline(always)]
pub fn needs_null_move_verification(depth: i32, null_score: i32, beta: i32) -> bool {
    
    depth >= 12 && null_score >= beta && null_score < 28000 
}




#[inline(always)]
pub fn delta_pruning(
    alpha: i32,
    stand_pat: i32,
    piece_value: i32,
    promotion_value: i32,
) -> bool {
    const DELTA_MARGIN: i32 = 300; 
    stand_pat + piece_value + promotion_value + DELTA_MARGIN < alpha
}


#[inline(always)]
pub fn delta_margin(phase: i32) -> i32 {
    
    
    if phase > 20 {
        400 
    } else if phase > 10 {
        350 
    } else {
        300 
    }
}




pub struct MultiCutContext {
    pub cutoff_count: u8,
    pub depth_threshold: i32,
    pub required_cutoffs: u8,
}

impl MultiCutContext {
    #[inline(always)]
    pub fn new(depth: i32) -> Self {
        Self {
            cutoff_count: 0,
            
            depth_threshold: 10, 
            required_cutoffs: 3,
        }
    }
    
    #[inline(always)]
    pub fn record_cutoff(&mut self) {
        self.cutoff_count = self.cutoff_count.saturating_add(1);
    }
    
    #[inline(always)]
    pub fn should_multi_cut(&self, depth: i32, moves_searched: usize) -> bool {
        depth >= self.depth_threshold &&
        moves_searched >= 6 && 
        self.cutoff_count >= self.required_cutoffs
    }
}




#[derive(Debug, Clone, Copy)]
pub enum PruningReason {
    Futility,
    ReverseFutility,
    Razoring,
    LateMove,
    SEE,
    History,
    MultiCut,
    Delta,
}

impl PruningReason {
    #[inline(always)]
    pub fn description(&self) -> &'static str {
        match self {
            PruningReason::Futility => "Futility",
            PruningReason::ReverseFutility => "Reverse Futility",
            PruningReason::Razoring => "Razoring",
            PruningReason::LateMove => "Late Move",
            PruningReason::SEE => "SEE",
            PruningReason::History => "History",
            PruningReason::MultiCut => "Multi-Cut",
            PruningReason::Delta => "Delta",
        }
    }
}

#[derive(Default)]
pub struct PruningStats {
    #[cfg(debug_assertions)]
    pub futility_prunes: AtomicU64,
    #[cfg(debug_assertions)]
    pub reverse_futility_prunes: AtomicU64,
    #[cfg(debug_assertions)]
    pub lmp_prunes: AtomicU64,
    #[cfg(debug_assertions)]
    pub see_prunes: AtomicU64,
    #[cfg(debug_assertions)]
    pub history_prunes: AtomicU64,
    #[cfg(debug_assertions)]
    pub razoring_prunes: AtomicU64,
    #[cfg(debug_assertions)]
    pub multi_cut_prunes: AtomicU64,
    #[cfg(debug_assertions)]
    pub delta_prunes: AtomicU64,
}

impl PruningStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    #[inline(always)]
    pub fn record_prune(&self, _reason: PruningReason) {
        #[cfg(debug_assertions)]
        {
            match _reason {
                PruningReason::Futility => self.futility_prunes.fetch_add(1, Ordering::Relaxed),
                PruningReason::ReverseFutility => self.reverse_futility_prunes.fetch_add(1, Ordering::Relaxed),
                PruningReason::LateMove => self.lmp_prunes.fetch_add(1, Ordering::Relaxed),
                PruningReason::SEE => self.see_prunes.fetch_add(1, Ordering::Relaxed),
                PruningReason::History => self.history_prunes.fetch_add(1, Ordering::Relaxed),
                PruningReason::Razoring => self.razoring_prunes.fetch_add(1, Ordering::Relaxed),
                PruningReason::MultiCut => self.multi_cut_prunes.fetch_add(1, Ordering::Relaxed),
                PruningReason::Delta => self.delta_prunes.fetch_add(1, Ordering::Relaxed),
            };
        }
    }
    
    pub fn clear(&self) {
        #[cfg(debug_assertions)]
        {
            self.futility_prunes.store(0, Ordering::Relaxed);
            self.reverse_futility_prunes.store(0, Ordering::Relaxed);
            self.lmp_prunes.store(0, Ordering::Relaxed);
            self.see_prunes.store(0, Ordering::Relaxed);
            self.history_prunes.store(0, Ordering::Relaxed);
            self.razoring_prunes.store(0, Ordering::Relaxed);
            self.multi_cut_prunes.store(0, Ordering::Relaxed);
            self.delta_prunes.store(0, Ordering::Relaxed);
        }
    }
}




#[inline(always)]
pub fn has_non_pawn_material(pos: &Position, side: Color) -> bool {
    pos.pieces_colored(PieceType::Knight, side) != 0 ||
    pos.pieces_colored(PieceType::Bishop, side) != 0 ||
    pos.pieces_colored(PieceType::Rook, side) != 0 ||
    pos.pieces_colored(PieceType::Queen, side) != 0
}






#[inline(always)]
pub fn singular_extension_check(
    depth: i32,
    tt_score: i32,
    tt_depth: u8,
    beta: i32,
    is_pv: bool,
) -> Option<(i32, i32)> {
    
    if depth < 7 {
        return None;
    }
    
    
    if tt_depth < (depth as u8).saturating_sub(3) {
        return None;
    }
    
    
    if tt_score.abs() >= 28000 {
        return None;
    }
    
    
    if tt_score < beta + 30 {
        return None;
    }
    
    
    let singular_margin = if is_pv { depth * 2 } else { depth * 3 };
    let singular_beta = tt_score - singular_margin;
    
    let singular_depth = ((depth - 1) / 2).max(1);
    
    Some((singular_beta, singular_depth))
}


#[inline(always)]
pub fn singular_extension_result(
    singular_score: i32,
    singular_beta: i32,
    tt_score: i32,
    depth: i32,
    is_pv: bool,
) -> i32 {
    if singular_score < singular_beta {
        
        if !is_pv && singular_score < singular_beta - depth && depth < 12 {
            return 2; 
        }
        return 1; 
    }
    
    
    if singular_score >= tt_score {
        return -1; 
    }
    
    0
}





pub const PROBCUT_MARGIN: i32 = 200;
pub const PROBCUT_MIN_DEPTH: i32 = 5;
pub const PROBCUT_REDUCTION: i32 = 4;

#[inline(always)]
pub fn probcut_conditions(
    depth: i32,
    beta: i32,
    is_pv: bool,
    in_check: bool,
    has_non_pawn_material: bool,
) -> bool {
    if is_pv {
        return false;
    }
    
    if depth < PROBCUT_MIN_DEPTH {
        return false;
    }
    
    if in_check {
        return false;
    }
    
    if !has_non_pawn_material {
        return false;
    }
    
    if beta.abs() >= 28000 {
        return false;
    }
    
    true
}

#[inline(always)]
pub fn probcut_beta(beta: i32) -> i32 {
    beta + PROBCUT_MARGIN
}

#[inline(always)]
pub fn probcut_depth(depth: i32) -> i32 {
    (depth - PROBCUT_REDUCTION).max(1)
}

#[inline(always)]
pub fn probcut_move_ok(
    is_capture: bool,
    see_value: i32,
    probcut_beta: i32,
    static_eval: i32,
) -> bool {
    if !is_capture {
        return false;
    }
    
    if see_value < 0 {
        return false;
    }
    
    
    if static_eval + see_value < probcut_beta - 150 {
        return false;
    }
    
    true
}