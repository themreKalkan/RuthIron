use crate::board::position::{Position, Color, Move, PieceType, MoveType};
use super::nnue_weights::{NNUEWeights, FT_OUT_DIMS, L1_OUT_DIMS, L2_OUT_DIMS, L1_IN_DIMS, L2_IN_DIMS};
use super::nnue_accumulator::Accumulator;
use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const NNUE_SCALE: i32 = 36;
const L1_SHIFT: i32 = 6;
const L2_SHIFT: i32 = 6;


#[derive(Debug, Clone)]
pub struct NNUEDetail {
    pub eval: i32,
    pub eval_raw: i32,
    pub nnue_eval: i32,
    pub material_bias: i32,
}

pub struct NNUE {
    weights: Arc<NNUEWeights>,
    pub accumulator: Accumulator,
}

impl NNUE {
    pub fn new(weights: Arc<NNUEWeights>) -> Self {
        Self {
            weights,
            accumulator: Accumulator::new(),
        }
    }
    
    
    pub fn load_embedded() -> Result<Self, Box<dyn std::error::Error>> {
        let weights = NNUEWeights::load_embedded()?;
        Ok(Self::new(Arc::new(weights)))
    }
    
    
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let weights = NNUEWeights::load_from_file(path)?;
        Ok(Self::new(Arc::new(weights)))
    }
    
    
    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let weights = NNUEWeights::load_from_bytes(data)?;
        Ok(Self::new(Arc::new(weights)))
    }

    
    #[inline(always)]
    pub fn refresh(&mut self, pos: &Position) {
        self.accumulator.refresh(pos, &self.weights);
    }

    
    #[inline(always)]
    pub fn push(&mut self) {
        self.accumulator.push();
    }
    
    
    #[inline(always)]
    pub fn pop(&mut self) {
        self.accumulator.pop();
    }

    
    #[inline]
    pub fn push_move(&mut self, mv: Move, pos: &Position) {
        self.accumulator.push();
        
        let from = mv.from();
        let to = mv.to();
        let (piece, piece_color) = pos.piece_at(from);
        
        if piece == PieceType::None {
            return;
        }
        
        let w_king = pos.king_square(Color::White);
        let b_king = pos.king_square(Color::Black);
        
        
        if piece == PieceType::King {
            return;
        }
        
        let move_type = mv.move_type();
        
        match move_type {
            MoveType::Normal => {
                
                let old_w_idx = Accumulator::halfkp_index(w_king, from, piece, piece_color, Color::White);
                let old_b_idx = Accumulator::halfkp_index(b_king, from, piece, piece_color, Color::Black);
                
                
                let new_w_idx = Accumulator::halfkp_index(w_king, to, piece, piece_color, Color::White);
                let new_b_idx = Accumulator::halfkp_index(b_king, to, piece, piece_color, Color::Black);
                
                
                let (captured, cap_color) = pos.piece_at(to);
                if captured != PieceType::None && captured != PieceType::King {
                    let cap_w_idx = Accumulator::halfkp_index(w_king, to, captured, cap_color, Color::White);
                    let cap_b_idx = Accumulator::halfkp_index(b_king, to, captured, cap_color, Color::Black);
                    
                    self.accumulator.sub_feature_white(&self.weights.ft_weights, cap_w_idx);
                    self.accumulator.sub_feature_black(&self.weights.ft_weights, cap_b_idx);
                }
                
                
                self.accumulator.add_sub_feature_white(&self.weights.ft_weights, new_w_idx, old_w_idx);
                self.accumulator.add_sub_feature_black(&self.weights.ft_weights, new_b_idx, old_b_idx);
            }
            MoveType::EnPassant => {
                let captured_sq = to ^ 8;
                
                
                let old_w_idx = Accumulator::halfkp_index(w_king, from, PieceType::Pawn, piece_color, Color::White);
                let old_b_idx = Accumulator::halfkp_index(b_king, from, PieceType::Pawn, piece_color, Color::Black);
                let new_w_idx = Accumulator::halfkp_index(w_king, to, PieceType::Pawn, piece_color, Color::White);
                let new_b_idx = Accumulator::halfkp_index(b_king, to, PieceType::Pawn, piece_color, Color::Black);
                
                
                let cap_color = piece_color.opposite();
                let cap_w_idx = Accumulator::halfkp_index(w_king, captured_sq, PieceType::Pawn, cap_color, Color::White);
                let cap_b_idx = Accumulator::halfkp_index(b_king, captured_sq, PieceType::Pawn, cap_color, Color::Black);
                
                self.accumulator.sub_feature_white(&self.weights.ft_weights, cap_w_idx);
                self.accumulator.sub_feature_black(&self.weights.ft_weights, cap_b_idx);
                self.accumulator.add_sub_feature_white(&self.weights.ft_weights, new_w_idx, old_w_idx);
                self.accumulator.add_sub_feature_black(&self.weights.ft_weights, new_b_idx, old_b_idx);
            }
            MoveType::Castle => {
                
                return;
            }
            MoveType::Promotion => {
                let promo_piece = mv.promotion();
                
                
                let old_w_idx = Accumulator::halfkp_index(w_king, from, PieceType::Pawn, piece_color, Color::White);
                let old_b_idx = Accumulator::halfkp_index(b_king, from, PieceType::Pawn, piece_color, Color::Black);
                
                
                let new_w_idx = Accumulator::halfkp_index(w_king, to, promo_piece, piece_color, Color::White);
                let new_b_idx = Accumulator::halfkp_index(b_king, to, promo_piece, piece_color, Color::Black);
                
                
                let (captured, cap_color) = pos.piece_at(to);
                if captured != PieceType::None && captured != PieceType::King {
                    let cap_w_idx = Accumulator::halfkp_index(w_king, to, captured, cap_color, Color::White);
                    let cap_b_idx = Accumulator::halfkp_index(b_king, to, captured, cap_color, Color::Black);
                    
                    self.accumulator.sub_feature_white(&self.weights.ft_weights, cap_w_idx);
                    self.accumulator.sub_feature_black(&self.weights.ft_weights, cap_b_idx);
                }
                
                self.accumulator.add_sub_feature_white(&self.weights.ft_weights, new_w_idx, old_w_idx);
                self.accumulator.add_sub_feature_black(&self.weights.ft_weights, new_b_idx, old_b_idx);
            }
        }
    }
    
    
    #[inline(always)]
    pub fn pop_move(&mut self) {
        self.accumulator.pop();
    }

    
    #[inline]
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let raw_total = self.evaluate_raw(pos);
        raw_total / NNUE_SCALE
    }
    
    
    #[inline]
    pub fn evaluate_current(&self, pos: &Position) -> i32 {
        let raw_total = self.evaluate_raw_current(pos);
        raw_total / NNUE_SCALE
    }

    pub fn evaluate_detailed(&self, pos: &Position) -> NNUEDetail {
        let raw_total = self.evaluate_raw(pos);
        let material = self.weights.output_biases[0];
        let scaled_score = raw_total / NNUE_SCALE;
        NNUEDetail {
            eval: scaled_score,
            eval_raw: raw_total,
            nnue_eval: scaled_score,
            material_bias: material,
        }
    }
    
    
    #[inline]
    fn evaluate_raw(&self, pos: &Position) -> i32 {
        let mut temp_acc = Accumulator::new();
        temp_acc.refresh(pos, &self.weights);
        self.evaluate_raw_with_acc(pos, &temp_acc)
    }
    
    
    #[inline]
    fn evaluate_raw_current(&self, pos: &Position) -> i32 {
        self.evaluate_raw_with_acc(pos, &self.accumulator)
    }
    
    #[inline]
    fn evaluate_raw_with_acc(&self, pos: &Position, acc: &Accumulator) -> i32 {
        let (stm_acc, nstm_acc) = if pos.side_to_move == Color::White {
            (&acc.white, &acc.black)
        } else {
            (&acc.black, &acc.white)
        };

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.evaluate_avx2(stm_acc, nstm_acc) };
            }
        }
        
        
        self.evaluate_scalar(stm_acc, nstm_acc)
    }
    
    
    #[inline]
    fn evaluate_scalar(&self, stm_acc: &[i16; FT_OUT_DIMS], nstm_acc: &[i16; FT_OUT_DIMS]) -> i32 {
        let l1_out = self.layer1_forward_scalar(stm_acc, nstm_acc);
        let l2_out = self.layer2_forward_scalar(&l1_out);
        self.output_forward_scalar(&l2_out) + self.weights.output_biases[0]
    }
    
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn evaluate_avx2(&self, stm_acc: &[i16; FT_OUT_DIMS], nstm_acc: &[i16; FT_OUT_DIMS]) -> i32 {
        
        
        self.evaluate_scalar(stm_acc, nstm_acc)
    }

    
    
    
    
    #[inline]
    fn layer1_forward_scalar(&self, stm_acc: &[i16; FT_OUT_DIMS], nstm_acc: &[i16; FT_OUT_DIMS]) -> [i8; L1_OUT_DIMS] {
        let mut output = [0i8; L1_OUT_DIMS];
        
        for o in 0..L1_OUT_DIMS {
            let mut sum = self.weights.l1_biases[o];
            let weight_base = o * L1_IN_DIMS;
            
            
            for i in (0..FT_OUT_DIMS).step_by(8) {
                unsafe {
                    let w_ptr = self.weights.l1_weights.as_ptr().add(weight_base + i);
                    sum += (*stm_acc.get_unchecked(i)).clamp(0, 127) as i32 * *w_ptr as i32;
                    sum += (*stm_acc.get_unchecked(i+1)).clamp(0, 127) as i32 * *w_ptr.add(1) as i32;
                    sum += (*stm_acc.get_unchecked(i+2)).clamp(0, 127) as i32 * *w_ptr.add(2) as i32;
                    sum += (*stm_acc.get_unchecked(i+3)).clamp(0, 127) as i32 * *w_ptr.add(3) as i32;
                    sum += (*stm_acc.get_unchecked(i+4)).clamp(0, 127) as i32 * *w_ptr.add(4) as i32;
                    sum += (*stm_acc.get_unchecked(i+5)).clamp(0, 127) as i32 * *w_ptr.add(5) as i32;
                    sum += (*stm_acc.get_unchecked(i+6)).clamp(0, 127) as i32 * *w_ptr.add(6) as i32;
                    sum += (*stm_acc.get_unchecked(i+7)).clamp(0, 127) as i32 * *w_ptr.add(7) as i32;
                }
            }
            
            
            for i in (0..FT_OUT_DIMS).step_by(8) {
                unsafe {
                    let w_ptr = self.weights.l1_weights.as_ptr().add(weight_base + FT_OUT_DIMS + i);
                    sum += (*nstm_acc.get_unchecked(i)).clamp(0, 127) as i32 * *w_ptr as i32;
                    sum += (*nstm_acc.get_unchecked(i+1)).clamp(0, 127) as i32 * *w_ptr.add(1) as i32;
                    sum += (*nstm_acc.get_unchecked(i+2)).clamp(0, 127) as i32 * *w_ptr.add(2) as i32;
                    sum += (*nstm_acc.get_unchecked(i+3)).clamp(0, 127) as i32 * *w_ptr.add(3) as i32;
                    sum += (*nstm_acc.get_unchecked(i+4)).clamp(0, 127) as i32 * *w_ptr.add(4) as i32;
                    sum += (*nstm_acc.get_unchecked(i+5)).clamp(0, 127) as i32 * *w_ptr.add(5) as i32;
                    sum += (*nstm_acc.get_unchecked(i+6)).clamp(0, 127) as i32 * *w_ptr.add(6) as i32;
                    sum += (*nstm_acc.get_unchecked(i+7)).clamp(0, 127) as i32 * *w_ptr.add(7) as i32;
                }
            }
            
            let shifted = (sum + (1 << (L1_SHIFT - 1))) >> L1_SHIFT;
            output[o] = shifted.clamp(0, 127) as i8;
        }
        output
    }

    #[inline]
    fn layer2_forward_scalar(&self, l1_out: &[i8; L1_OUT_DIMS]) -> [i8; L2_OUT_DIMS] {
        let mut output = [0i8; L2_OUT_DIMS];
        
        for o in 0..L2_OUT_DIMS {
            let mut sum = self.weights.l2_biases[o];
            let weight_base = o * L2_IN_DIMS;
            
            
            for i in (0..L1_OUT_DIMS).step_by(8) {
                unsafe {
                    let w_ptr = self.weights.l2_weights.as_ptr().add(weight_base + i);
                    sum += *l1_out.get_unchecked(i) as i32 * *w_ptr as i32;
                    sum += *l1_out.get_unchecked(i+1) as i32 * *w_ptr.add(1) as i32;
                    sum += *l1_out.get_unchecked(i+2) as i32 * *w_ptr.add(2) as i32;
                    sum += *l1_out.get_unchecked(i+3) as i32 * *w_ptr.add(3) as i32;
                    sum += *l1_out.get_unchecked(i+4) as i32 * *w_ptr.add(4) as i32;
                    sum += *l1_out.get_unchecked(i+5) as i32 * *w_ptr.add(5) as i32;
                    sum += *l1_out.get_unchecked(i+6) as i32 * *w_ptr.add(6) as i32;
                    sum += *l1_out.get_unchecked(i+7) as i32 * *w_ptr.add(7) as i32;
                }
            }
            
            let shifted = (sum + (1 << (L2_SHIFT - 1))) >> L2_SHIFT;
            output[o] = shifted.clamp(0, 127) as i8;
        }
        output
    }

    #[inline]
    fn output_forward_scalar(&self, l2_out: &[i8; L2_OUT_DIMS]) -> i32 {
        let mut sum = 0i32;
        
        for i in (0..L2_OUT_DIMS).step_by(8) {
            unsafe {
                let w_ptr = self.weights.output_weights.as_ptr().add(i);
                sum += *l2_out.get_unchecked(i) as i32 * *w_ptr as i32;
                sum += *l2_out.get_unchecked(i+1) as i32 * *w_ptr.add(1) as i32;
                sum += *l2_out.get_unchecked(i+2) as i32 * *w_ptr.add(2) as i32;
                sum += *l2_out.get_unchecked(i+3) as i32 * *w_ptr.add(3) as i32;
                sum += *l2_out.get_unchecked(i+4) as i32 * *w_ptr.add(4) as i32;
                sum += *l2_out.get_unchecked(i+5) as i32 * *w_ptr.add(5) as i32;
                sum += *l2_out.get_unchecked(i+6) as i32 * *w_ptr.add(6) as i32;
                sum += *l2_out.get_unchecked(i+7) as i32 * *w_ptr.add(7) as i32;
            }
        }
        sum
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nnue_scale() {
        assert_eq!(NNUE_SCALE, 16);
    }
}