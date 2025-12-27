use anyhow::{Result, Context, anyhow};
use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::io::Write;
use std::fs;
use std::env;
use std::sync::{OnceLock, Mutex, RwLock};
use std::collections::HashMap;

use crate::board::position::{Position, Move, MoveType, PieceType, Color, algebraic_to_square};

const MODEL_BYTES: &[u8] = include_bytes!("chess_ai_final.onnx");

const MAX_CACHE_SIZE: usize = 100_000;  
const CACHE_CLEANUP_THRESHOLD: usize = 90_000;  

#[derive(Clone)]
struct CacheEntry {
    moves: Vec<(Move, f32)>,
    access_count: u32,
    last_access: u64,
}

static GLOBAL_AI: OnceLock<ChessAI> = OnceLock::new();
static ACCESS_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn init_model() -> Result<()> {
    if GLOBAL_AI.get().is_some() {
        return Ok(());
    }

    let ai = ChessAI::new()?;
    
    GLOBAL_AI.set(ai).map_err(|_| anyhow!("Model global değişkene atanamadı"))?;
    
    Ok(())
}


pub fn get_best_moves(pos: &Position, count: usize) -> Vec<(Move, f32)> {
    let ai = match GLOBAL_AI.get() {
        Some(ai) => ai,
        None => return Vec::new(),
    };

    match ai.internal_get_best_moves(pos, count) {
        Ok(moves) => moves,
        Err(e) => {
            eprintln!("AI Kritik Hata: {}", e);
            Vec::new()
        }
    }
}


#[allow(dead_code)]
pub fn get_cache_stats() -> (usize, u64) {
    let ai = match GLOBAL_AI.get() {
        Some(ai) => ai,
        None => return (0, 0),
    };
    
    let cache = ai.cache.read().unwrap();
    let hits = ai.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
    (cache.len(), hits)
}


#[allow(dead_code)]
pub fn clear_cache() {
    if let Some(ai) = GLOBAL_AI.get() {
        let mut cache = ai.cache.write().unwrap();
        cache.clear();
    }
}

struct ChessAI {
    session: Mutex<Session>,
    cache: RwLock<HashMap<u64, CacheEntry>>,
    cache_hits: std::sync::atomic::AtomicU64,
}

impl ChessAI {
    fn new() -> Result<Self> {
        let temp_dir = env::temp_dir();
        let temp_model_path = temp_dir.join(format!("chess_model_{}.onnx", std::process::id()));
        
        fs::write(&temp_model_path, MODEL_BYTES)
            .context("Model dosyası diske yazılamadı")?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)? 
            .commit_from_file(&temp_model_path)
            .context("ONNX oturumu başlatılamadı")?;

        let _ = fs::remove_file(temp_model_path);

        Ok(Self { 
            session: Mutex::new(session),
            cache: RwLock::new(HashMap::with_capacity(MAX_CACHE_SIZE / 2)),
            cache_hits: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn internal_get_best_moves(&self, pos: &Position, count: usize) -> Result<Vec<(Move, f32)>> {
        let hash = pos.hash;
        let current_access = ACCESS_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        
        
        
        {
            let cache = self.cache.read().unwrap();
            if let Some(entry) = cache.get(&hash) {
                self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                
                
                if entry.moves.len() >= count {
                    return Ok(entry.moves[..count].to_vec());
                }
                
                return Ok(entry.moves.clone());
            }
        }

        
        
        
        let (shape, data) = position_to_tensor_optimized(pos);
        let input_value = Value::from_array((shape, data))?;

        let logits: Vec<f32> = {
            let mut session_guard = self.session.lock().map_err(|_| anyhow!("AI Model Mutex Poisoned"))?;
            let outputs = session_guard.run(ort::inputs![&input_value])?;
            let output = outputs[0].try_extract_tensor::<f32>()?;
            output.1.to_vec()  
        };
        
        
        let (indexed_probs, _temperature_adjusted) = compute_softmax_with_temperature(&logits, pos);

        
        let max_candidates = count * 5 + 30; 
        let mut moves = Vec::with_capacity(count.max(15)); 
        let mut candidates_checked = 0;
        
        for (idx, score) in indexed_probs {
            if moves.len() >= count.max(15) && candidates_checked > max_candidates { 
                break; 
            }
            candidates_checked += 1;

            let uci_string = decode_move_idx(idx);
            
            if let Some(mv) = uci_to_move(pos, &uci_string) {
                moves.push((mv, score));
            }
        }

        
        {
            let mut cache = self.cache.write().unwrap();
            
            
            if cache.len() >= CACHE_CLEANUP_THRESHOLD {
                self.smart_cache_cleanup(&mut cache, current_access);
            }
            
            cache.insert(hash, CacheEntry {
                moves: moves.clone(),
                access_count: 1,
                last_access: current_access,
            });
        }

        Ok(if moves.len() > count { moves[..count].to_vec() } else { moves })
    }
    
    
    fn smart_cache_cleanup(&self, cache: &mut HashMap<u64, CacheEntry>, current_access: u64) {
        
        let target_size = MAX_CACHE_SIZE * 7 / 10;
        
        if cache.len() <= target_size {
            return;
        }
        
        
        let mut entries: Vec<(u64, u64)> = cache.iter()
            .map(|(hash, entry)| {
                let age = current_access.saturating_sub(entry.last_access);
                let score = (entry.access_count as u64) * 1000 / (age + 1);
                (*hash, score)
            })
            .collect();
        
        
        entries.sort_by_key(|(_, score)| *score);
        
        let to_remove = cache.len() - target_size;
        for (hash, _) in entries.into_iter().take(to_remove) {
            cache.remove(&hash);
        }
    }
}


#[inline]
fn position_to_tensor_optimized(pos: &Position) -> ([usize; 4], Vec<f32>) {
    let mut data = vec![0.0f32; 14 * 64]; 
    
    
    for piece_type_idx in 0..6 {
        let piece_type = match piece_type_idx {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            5 => PieceType::King,
            _ => unreachable!(),
        };
        
        
        let white_bb = pos.pieces_colored(piece_type, Color::White);
        let mut bb = white_bb;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            data[piece_type_idx * 64 + sq] = 1.0;
            bb &= bb - 1;
        }
        
        
        let black_bb = pos.pieces_colored(piece_type, Color::Black);
        let mut bb = black_bb;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            data[(piece_type_idx + 6) * 64 + sq] = 1.0;
            bb &= bb - 1;
        }
    }

    
    let turn_val = if pos.side_to_move == Color::White { 1.0 } else { 0.0 };
    let turn_offset = 13 * 64;
    for i in 0..64 {
        data[turn_offset + i] = turn_val;
    }

    ([1, 14, 8, 8], data)
}


#[inline]
fn compute_softmax_with_temperature(logits: &[f32], pos: &Position) -> (Vec<(usize, f32)>, bool) {
    
    
    let in_check = pos.is_in_check(pos.side_to_move);
    let temperature = if in_check { 0.8 } else { 1.0 };
    
    
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    
    let mut sum_exp = 0.0f32;
    let exp_logits: Vec<f32> = logits.iter().map(|&x| {
        let scaled = (x - max_logit) / temperature;
        let e = scaled.exp();
        sum_exp += e;
        e
    }).collect();
    
    
    let mut indexed_probs: Vec<(usize, f32)> = exp_logits.iter()
        .enumerate()
        .map(|(i, &e)| (i, e / sum_exp))
        .collect();
    
    
    indexed_probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    (indexed_probs, temperature != 1.0)
}


fn uci_to_move(pos: &Position, uci: &str) -> Option<Move> {
    if uci.len() < 4 { return None; }
    
    let from_str = &uci[0..2];
    let to_str = &uci[2..4];
    
    let from_sq = algebraic_to_square(from_str)?;
    let to_sq = algebraic_to_square(to_str)?;
    
    
    let (piece_type, piece_color) = pos.piece_at(from_sq);
    
    
    if piece_type == PieceType::None {
        return None;
    }
    
    
    if piece_color != pos.side_to_move {
        return None;
    }
    
    let promotion = if uci.len() == 5 {
        match uci.chars().nth(4).unwrap() {
            'q' => PieceType::Queen,
            'r' => PieceType::Rook,
            'b' => PieceType::Bishop,
            'n' => PieceType::Knight,
            _ => PieceType::None,
        }
    } else {
        PieceType::None
    };

    let mut move_type = MoveType::Normal;

    if promotion != PieceType::None {
        move_type = MoveType::Promotion;
    } else if piece_type == PieceType::King {
        let diff = (from_sq as i8 - to_sq as i8).abs();
        if diff == 2 {
            move_type = MoveType::Castle;
        }
    } else if piece_type == PieceType::Pawn {
        let is_diagonal = (from_sq % 8) != (to_sq % 8);
        let dest_is_empty = pos.piece_at(to_sq).0 == PieceType::None;
        
        if is_diagonal && dest_is_empty {
            move_type = MoveType::EnPassant;
        }
    }

    Some(Move::new(from_sq, to_sq, move_type, promotion))
}


static SQUARE_NAMES: once_cell::sync::Lazy<[String; 64]> = once_cell::sync::Lazy::new(|| {
    let mut names = std::array::from_fn(|_| String::new());
    for sq in 0..64 {
        let rank = sq / 8;
        let file = sq % 8;
        names[sq] = format!("{}{}", (b'a' + file as u8) as char, (b'1' + rank as u8) as char);
    }
    names
});

#[inline]
fn decode_move_idx(idx: usize) -> String {
    let promotions = ['q', 'r', 'b', 'n'];

    if idx >= 4096 {
        
        let offset = idx - 4096;
        let from_idx = offset / 4;
        let promo_idx = offset % 4;
        
        if from_idx >= 64 {
            return String::new();
        }
        
        let from_sq = &SQUARE_NAMES[from_idx];
        let from_rank_char = from_sq.chars().nth(1).unwrap();
        
        let to_rank = if from_rank_char == '7' { '8' } else { '1' };
        let file = from_sq.chars().nth(0).unwrap();
        
        format!("{}{}{}{}", from_sq, file, to_rank, promotions[promo_idx])
    } else {
        
        let from_idx = idx / 64;
        let to_idx = idx % 64;
        
        if from_idx >= 64 || to_idx >= 64 {
            return String::new();
        }
        
        format!("{}{}", &SQUARE_NAMES[from_idx], &SQUARE_NAMES[to_idx])
    }
}


#[allow(dead_code)]
pub fn get_best_moves_batch(positions: &[Position], count: usize) -> Vec<Vec<(Move, f32)>> {
    
    positions.iter()
        .map(|pos| get_best_moves(pos, count))
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decode_move_idx() {
        
        
        
        let uci = decode_move_idx(796);
        assert_eq!(uci, "e2e4");
        
        
        let uci = decode_move_idx(0 * 64 + 8);
        assert_eq!(uci, "a1a2");
    }
    
    #[test]
    fn test_softmax_temperature() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
    }
}