//THIS ENGINE USING NNUE OF STOCKFISH HALFKP
use crate::board::position::{Position, PieceType, Color, Move};
use super::nnue_weights::{NNUEWeights, FT_OUT_DIMS};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;






#[cfg(target_arch = "x86_64")]
const SIMD_WIDTH: usize = 16;

#[cfg(not(target_arch = "x86_64"))]
const SIMD_WIDTH: usize = 1;


const NUM_CHUNKS: usize = FT_OUT_DIMS / SIMD_WIDTH;




#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct AccumulatorState {
    pub white: [i16; FT_OUT_DIMS],
    pub black: [i16; FT_OUT_DIMS],
}

impl Default for AccumulatorState {
    fn default() -> Self {
        Self {
            white: [0; FT_OUT_DIMS],
            black: [0; FT_OUT_DIMS],
        }
    }
}




const MAX_STACK_SIZE: usize = 128;

#[repr(C, align(64))]
pub struct Accumulator {
    pub white: [i16; FT_OUT_DIMS],
    pub black: [i16; FT_OUT_DIMS],
    
    
    stack: Vec<AccumulatorState>,
    stack_size: usize,
    
    
    computed: bool,
}

impl Clone for Accumulator {
    fn clone(&self) -> Self {
        Self {
            white: self.white,
            black: self.black,
            stack: self.stack.clone(),
            stack_size: self.stack_size,
            computed: self.computed,
        }
    }
}

impl std::fmt::Debug for Accumulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Accumulator")
            .field("white[0..5]", &&self.white[0..5])
            .field("black[0..5]", &&self.black[0..5])
            .field("stack_size", &self.stack_size)
            .field("computed", &self.computed)
            .finish()
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Self {
            white: [0; FT_OUT_DIMS],
            black: [0; FT_OUT_DIMS],
            stack: Vec::with_capacity(MAX_STACK_SIZE),
            stack_size: 0,
            computed: false,
        }
    }

    
    #[inline(always)]
    pub fn push(&mut self) {
        if self.stack.len() <= self.stack_size {
            self.stack.push(AccumulatorState::default());
        }
        self.stack[self.stack_size].white = self.white;
        self.stack[self.stack_size].black = self.black;
        self.stack_size += 1;
    }
    
    
    #[inline(always)]
    pub fn pop(&mut self) {
        debug_assert!(self.stack_size > 0);
        self.stack_size -= 1;
        self.white = self.stack[self.stack_size].white;
        self.black = self.stack[self.stack_size].black;
    }

    
    pub fn refresh(&mut self, pos: &Position, weights: &NNUEWeights) {
        let w_king_sq = pos.king_square(Color::White);
        let b_king_sq = pos.king_square(Color::Black);
        
        
        self.copy_biases(weights);

        
        for sq in 0..64 {
            let (piece, color) = pos.piece_at(sq as u8);
            if piece != PieceType::None && piece != PieceType::King {
                let w_idx = Self::halfkp_index(w_king_sq, sq as u8, piece, color, Color::White);
                let b_idx = Self::halfkp_index(b_king_sq, sq as u8, piece, color, Color::Black);
                
                Self::add_feature_to_acc(&mut self.white, &weights.ft_weights, w_idx);
                Self::add_feature_to_acc(&mut self.black, &weights.ft_weights, b_idx);
            }
        }
        
        self.computed = true;
    }
    
    
    #[inline(always)]
    fn copy_biases(&mut self, weights: &NNUEWeights) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.copy_biases_avx2(weights); }
                return;
            }
        }
        
        
        self.white.copy_from_slice(&weights.ft_biases);
        self.black.copy_from_slice(&weights.ft_biases);
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn copy_biases_avx2(&mut self, weights: &NNUEWeights) {
        let src = weights.ft_biases.as_ptr() as *const __m256i;
        let dst_w = self.white.as_mut_ptr() as *mut __m256i;
        let dst_b = self.black.as_mut_ptr() as *mut __m256i;
        
        
        for i in 0..NUM_CHUNKS {
            let val = _mm256_loadu_si256(src.add(i));
            _mm256_storeu_si256(dst_w.add(i), val);
            _mm256_storeu_si256(dst_b.add(i), val);
        }
    }

    
    #[inline(always)]
    fn add_feature_to_acc(acc: &mut [i16; FT_OUT_DIMS], weights: &[i16], feature_idx: usize) {
        let offset = feature_idx * FT_OUT_DIMS;
        if offset + FT_OUT_DIMS > weights.len() {
            return;
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::add_feature_avx2_impl(acc, weights, offset); }
                return;
            }
        }
        
        
        let w_ptr = unsafe { weights.as_ptr().add(offset) };
        
        for i in (0..FT_OUT_DIMS).step_by(8) {
            unsafe {
                *acc.get_unchecked_mut(i) = acc.get_unchecked(i).wrapping_add(*w_ptr.add(i));
                *acc.get_unchecked_mut(i+1) = acc.get_unchecked(i+1).wrapping_add(*w_ptr.add(i+1));
                *acc.get_unchecked_mut(i+2) = acc.get_unchecked(i+2).wrapping_add(*w_ptr.add(i+2));
                *acc.get_unchecked_mut(i+3) = acc.get_unchecked(i+3).wrapping_add(*w_ptr.add(i+3));
                *acc.get_unchecked_mut(i+4) = acc.get_unchecked(i+4).wrapping_add(*w_ptr.add(i+4));
                *acc.get_unchecked_mut(i+5) = acc.get_unchecked(i+5).wrapping_add(*w_ptr.add(i+5));
                *acc.get_unchecked_mut(i+6) = acc.get_unchecked(i+6).wrapping_add(*w_ptr.add(i+6));
                *acc.get_unchecked_mut(i+7) = acc.get_unchecked(i+7).wrapping_add(*w_ptr.add(i+7));
            }
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]  
    unsafe fn add_feature_avx2_impl(acc: &mut [i16; FT_OUT_DIMS], weights: &[i16], offset: usize) {
        let src = weights.as_ptr().add(offset) as *const __m256i;
        let dst = acc.as_mut_ptr() as *mut __m256i;
        
        for i in 0..NUM_CHUNKS {
            let a = _mm256_loadu_si256(dst.add(i));
            let b = _mm256_loadu_si256(src.add(i));
            let sum = _mm256_add_epi16(a, b);
            _mm256_storeu_si256(dst.add(i), sum);
        }
    }

    
    #[inline(always)]
    pub fn add_feature_white(&mut self, weights: &[i16], feature_idx: usize) {
        Self::add_feature_to_acc(&mut self.white, weights, feature_idx);
    }
    
    
    #[inline(always)]
    pub fn add_feature_black(&mut self, weights: &[i16], feature_idx: usize) {
        Self::add_feature_to_acc(&mut self.black, weights, feature_idx);
    }

    
    #[inline(always)]
    fn sub_feature_from_acc(acc: &mut [i16; FT_OUT_DIMS], weights: &[i16], feature_idx: usize) {
        let offset = feature_idx * FT_OUT_DIMS;
        if offset + FT_OUT_DIMS > weights.len() {
            return;
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::sub_feature_avx2_impl(acc, weights, offset); }
                return;
            }
        }
        
        
        let w_ptr = unsafe { weights.as_ptr().add(offset) };
        
        for i in (0..FT_OUT_DIMS).step_by(8) {
            unsafe {
                *acc.get_unchecked_mut(i) = acc.get_unchecked(i).wrapping_sub(*w_ptr.add(i));
                *acc.get_unchecked_mut(i+1) = acc.get_unchecked(i+1).wrapping_sub(*w_ptr.add(i+1));
                *acc.get_unchecked_mut(i+2) = acc.get_unchecked(i+2).wrapping_sub(*w_ptr.add(i+2));
                *acc.get_unchecked_mut(i+3) = acc.get_unchecked(i+3).wrapping_sub(*w_ptr.add(i+3));
                *acc.get_unchecked_mut(i+4) = acc.get_unchecked(i+4).wrapping_sub(*w_ptr.add(i+4));
                *acc.get_unchecked_mut(i+5) = acc.get_unchecked(i+5).wrapping_sub(*w_ptr.add(i+5));
                *acc.get_unchecked_mut(i+6) = acc.get_unchecked(i+6).wrapping_sub(*w_ptr.add(i+6));
                *acc.get_unchecked_mut(i+7) = acc.get_unchecked(i+7).wrapping_sub(*w_ptr.add(i+7));
            }
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub_feature_avx2_impl(acc: &mut [i16; FT_OUT_DIMS], weights: &[i16], offset: usize) {
        let src = weights.as_ptr().add(offset) as *const __m256i;
        let dst = acc.as_mut_ptr() as *mut __m256i;
        
        for i in 0..NUM_CHUNKS {
            let a = _mm256_loadu_si256(dst.add(i));
            let b = _mm256_loadu_si256(src.add(i));
            let diff = _mm256_sub_epi16(a, b);
            _mm256_storeu_si256(dst.add(i), diff);
        }
    }
    
    
    #[inline(always)]
    pub fn sub_feature_white(&mut self, weights: &[i16], feature_idx: usize) {
        Self::sub_feature_from_acc(&mut self.white, weights, feature_idx);
    }
    
    
    #[inline(always)]
    pub fn sub_feature_black(&mut self, weights: &[i16], feature_idx: usize) {
        Self::sub_feature_from_acc(&mut self.black, weights, feature_idx);
    }
    
    
    #[inline(always)]
    fn add_sub_feature_impl(
        acc: &mut [i16; FT_OUT_DIMS],
        weights: &[i16],
        add_idx: usize,
        sub_idx: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::add_sub_feature_avx2_impl(acc, weights, add_idx, sub_idx); }
                return;
            }
        }
        
        
        Self::add_feature_to_acc(acc, weights, add_idx);
        Self::sub_feature_from_acc(acc, weights, sub_idx);
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add_sub_feature_avx2_impl(
        acc: &mut [i16; FT_OUT_DIMS],
        weights: &[i16],
        add_idx: usize,
        sub_idx: usize,
    ) {
        let add_offset = add_idx * FT_OUT_DIMS;
        let sub_offset = sub_idx * FT_OUT_DIMS;
        
        let add_src = weights.as_ptr().add(add_offset) as *const __m256i;
        let sub_src = weights.as_ptr().add(sub_offset) as *const __m256i;
        let dst = acc.as_mut_ptr() as *mut __m256i;
        
        for i in 0..NUM_CHUNKS {
            let a = _mm256_loadu_si256(dst.add(i));
            let add_val = _mm256_loadu_si256(add_src.add(i));
            let sub_val = _mm256_loadu_si256(sub_src.add(i));
            let result = _mm256_sub_epi16(_mm256_add_epi16(a, add_val), sub_val);
            _mm256_storeu_si256(dst.add(i), result);
        }
    }
    
    
    #[inline(always)]
    pub fn add_sub_feature_white(&mut self, weights: &[i16], add_idx: usize, sub_idx: usize) {
        Self::add_sub_feature_impl(&mut self.white, weights, add_idx, sub_idx);
    }
    
    
    #[inline(always)]
    pub fn add_sub_feature_black(&mut self, weights: &[i16], add_idx: usize, sub_idx: usize) {
        Self::add_sub_feature_impl(&mut self.black, weights, add_idx, sub_idx);
    }

    
    #[inline(always)]
    pub fn halfkp_index(
        king_sq: u8,
        piece_sq: u8,
        piece: PieceType,
        piece_color: Color,
        perspective: Color,
    ) -> usize {
        let k_sq = Self::orient(king_sq, perspective) as usize;
        let p_sq = Self::orient(piece_sq, perspective) as usize;
        
        let p_type = match piece {
            PieceType::Pawn => 0,
            PieceType::Knight => 1,
            PieceType::Bishop => 2,
            PieceType::Rook => 3,
            PieceType::Queen => 4,
            _ => 0,
        };
        
        let p_color = (piece_color != perspective) as usize;
        let p_idx = p_type * 2 + p_color;
        
        1 + p_sq + p_idx * 64 + k_sq * 641
    }

    #[inline(always)]
    fn orient(sq: u8, perspective: Color) -> u8 {
        match perspective {
            Color::White => sq,
            Color::Black => sq ^ 63,
        }
    }
}