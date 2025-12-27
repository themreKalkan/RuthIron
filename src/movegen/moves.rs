use crate::board::bitboard::{Bitboard, square_mask, has_bit};
use crate::board::position::{Position, Color, PieceType, Move, MoveType};
use crate::board::zobrist::{CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ};
use crate::movegen::magic::all_attacks_for_king;
use super::magic::{
    get_rook_attacks, get_bishop_attacks, get_queen_attacks,
    get_knight_attacks, get_king_attacks, get_pawn_attacks,
    add_moves_from_bitboard, all_attacks
};

const PROMOTION_PIECES: [PieceType; 4] = [
    PieceType::Queen,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
];


const RANK_1: u64 = 0x00000000000000FF;
const RANK_2: u64 = 0x000000000000FF00;
const RANK_3: u64 = 0x0000000000FF0000;
const RANK_4: u64 = 0x00000000FF000000;
const RANK_5: u64 = 0x000000FF00000000;
const RANK_6: u64 = 0x0000FF0000000000;
const RANK_7: u64 = 0x00FF000000000000;
const RANK_8: u64 = 0xFF00000000000000;


const FILE_A: u64 = 0x0101010101010101;
const FILE_H: u64 = 0x8080808080808080;

#[inline(always)]
pub fn generate_moves(pos: &Position) -> Vec<Move> {
    let mut moves = Vec::with_capacity(256);
    
    generate_pawn_moves(pos, &mut moves);
    generate_knight_moves(pos, &mut moves);
    generate_bishop_moves(pos, &mut moves);
    generate_rook_moves(pos, &mut moves);
    generate_queen_moves(pos, &mut moves);
    generate_king_moves(pos, &mut moves);
    
    moves
}

#[inline(always)]
pub fn generate_moves_ovi(pos: &Position) -> (Vec<Move>, u64) {
    let mut moves = Vec::with_capacity(256);
    
    generate_pawn_moves(pos, &mut moves);
    generate_knight_moves(pos, &mut moves);
    generate_bishop_moves(pos, &mut moves);
    generate_rook_moves(pos, &mut moves);
    generate_queen_moves(pos, &mut moves);
    let attacks = generate_king_moves_ovi(pos, &mut moves);
    
    (moves, attacks)
}

fn generate_pawn_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let pawns = pos.pieces_colored(PieceType::Pawn, color);
    
    if pawns == 0 {
        return;
    }
    
    let occupancy = pos.all_pieces();
    let empty = !occupancy;
    let opponent_pieces = pos.pieces(color.opposite());
    
    match color {
        Color::White => generate_white_pawn_moves(pos, pawns, empty, opponent_pieces, moves),
        Color::Black => generate_black_pawn_moves(pos, pawns, empty, opponent_pieces, moves),
    }
}

fn generate_white_pawn_moves(
    pos: &Position,
    pawns: u64,
    empty: u64,
    opponent_pieces: u64,
    moves: &mut Vec<Move>
) {
    
    let single_push = (pawns << 8) & empty;
    
    
    let double_push_candidates = ((pawns & RANK_2) << 8) & empty;
    let double_push = (double_push_candidates << 8) & empty;
    
    
    let non_promo_push = single_push & !RANK_8;
    add_white_pawn_pushes(non_promo_push, 8, moves);
    
    
    let promo_push = single_push & RANK_8;
    add_white_pawn_promotions(promo_push, 8, moves);
    
    
    add_white_pawn_pushes(double_push, 16, moves);
    
    
    let left_attacks = ((pawns & !FILE_A) << 7) & opponent_pieces;
    let left_non_promo = left_attacks & !RANK_8;
    let left_promo = left_attacks & RANK_8;
    add_white_pawn_pushes(left_non_promo, 7, moves);
    add_white_pawn_promotions(left_promo, 7, moves);
    
    
    let right_attacks = ((pawns & !FILE_H) << 9) & opponent_pieces;
    let right_non_promo = right_attacks & !RANK_8;
    let right_promo = right_attacks & RANK_8;
    add_white_pawn_pushes(right_non_promo, 9, moves);
    add_white_pawn_promotions(right_promo, 9, moves);
    
    
    if pos.en_passant_square < 64 {
        let ep_sq = pos.en_passant_square;
        let ep_bit = 1u64 << ep_sq;
        
        
        if (ep_bit & RANK_6) != 0 {
            let ep_file = ep_sq % 8;
            
            
            if ep_file > 0 && ep_sq >= 9 {
                let attacker = ep_sq - 9;
                if attacker < 64 && (pawns & (1u64 << attacker)) != 0 {
                    moves.push(Move::new(attacker, ep_sq, MoveType::EnPassant, PieceType::None));
                }
            }
            
            
            if ep_file < 7 && ep_sq >= 7 {
                let attacker = ep_sq - 7;
                if attacker < 64 && (pawns & (1u64 << attacker)) != 0 {
                    moves.push(Move::new(attacker, ep_sq, MoveType::EnPassant, PieceType::None));
                }
            }
        }
    }
}

fn generate_black_pawn_moves(
    pos: &Position,
    pawns: u64,
    empty: u64,
    opponent_pieces: u64,
    moves: &mut Vec<Move>
) {
    
    let single_push = (pawns >> 8) & empty;
    
    
    let double_push_candidates = ((pawns & RANK_7) >> 8) & empty;
    let double_push = (double_push_candidates >> 8) & empty;
    
    
    let non_promo_push = single_push & !RANK_1;
    add_black_pawn_pushes(non_promo_push, 8, moves);
    
    
    let promo_push = single_push & RANK_1;
    add_black_pawn_promotions(promo_push, 8, moves);
    
    
    add_black_pawn_pushes(double_push, 16, moves);
    
    
    let left_attacks = ((pawns & !FILE_A) >> 9) & opponent_pieces;
    let left_non_promo = left_attacks & !RANK_1;
    let left_promo = left_attacks & RANK_1;
    add_black_pawn_pushes(left_non_promo, 9, moves);
    add_black_pawn_promotions(left_promo, 9, moves);
    
    
    let right_attacks = ((pawns & !FILE_H) >> 7) & opponent_pieces;
    let right_non_promo = right_attacks & !RANK_1;
    let right_promo = right_attacks & RANK_1;
    add_black_pawn_pushes(right_non_promo, 7, moves);
    add_black_pawn_promotions(right_promo, 7, moves);
    
    
    if pos.en_passant_square < 64 {
        let ep_sq = pos.en_passant_square;
        let ep_bit = 1u64 << ep_sq;
        
        
        if (ep_bit & RANK_3) != 0 {
            let ep_file = ep_sq % 8;
            
            
            if ep_file > 0 {
                let attacker = ep_sq + 7;
                if attacker < 64 && (pawns & (1u64 << attacker)) != 0 {
                    moves.push(Move::new(attacker, ep_sq, MoveType::EnPassant, PieceType::None));
                }
            }
            
            
            if ep_file < 7 {
                let attacker = ep_sq + 9;
                if attacker < 64 && (pawns & (1u64 << attacker)) != 0 {
                    moves.push(Move::new(attacker, ep_sq, MoveType::EnPassant, PieceType::None));
                }
            }
        }
    }
}


#[inline(always)]
fn add_white_pawn_pushes(mut targets: u64, offset: u8, moves: &mut Vec<Move>) {
    while targets != 0 {
        let to = targets.trailing_zeros() as u8;
        targets &= targets - 1;
        
        
        if to >= offset {
            let from = to - offset;
            if from < 64 {
                moves.push(Move::new(from, to, MoveType::Normal, PieceType::None));
            }
        }
    }
}


#[inline(always)]
fn add_white_pawn_promotions(mut targets: u64, offset: u8, moves: &mut Vec<Move>) {
    while targets != 0 {
        let to = targets.trailing_zeros() as u8;
        targets &= targets - 1;
        
        if to >= offset {
            let from = to - offset;
            if from < 64 {
                for &promo in &PROMOTION_PIECES {
                    moves.push(Move::new(from, to, MoveType::Promotion, promo));
                }
            }
        }
    }
}


#[inline(always)]
fn add_black_pawn_pushes(mut targets: u64, offset: u8, moves: &mut Vec<Move>) {
    while targets != 0 {
        let to = targets.trailing_zeros() as u8;
        targets &= targets - 1;
        
        
        let from = to + offset;
        if from < 64 {
            moves.push(Move::new(from, to, MoveType::Normal, PieceType::None));
        }
    }
}


#[inline(always)]
fn add_black_pawn_promotions(mut targets: u64, offset: u8, moves: &mut Vec<Move>) {
    while targets != 0 {
        let to = targets.trailing_zeros() as u8;
        targets &= targets - 1;
        
        let from = to + offset;
        if from < 64 {
            for &promo in &PROMOTION_PIECES {
                moves.push(Move::new(from, to, MoveType::Promotion, promo));
            }
        }
    }
}

fn generate_knight_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let knights = pos.pieces_colored(PieceType::Knight, color);
    let our_pieces = pos.pieces(color);
    
    let mut knights_bb = knights;
    while knights_bb != 0 {
        let from = knights_bb.trailing_zeros() as u8;
        knights_bb &= knights_bb - 1;
        
        if from >= 64 { continue; } 
        
        let attacks = get_knight_attacks(from) & !our_pieces;
        add_moves_from_bitboard(from, attacks, moves);
    }
}

fn generate_bishop_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    let occupancy = pos.all_pieces();
    let our_pieces = pos.pieces(color);
    
    let mut bishops_bb = bishops;
    while bishops_bb != 0 {
        let from = bishops_bb.trailing_zeros() as u8;
        bishops_bb &= bishops_bb - 1;
        
        if from >= 64 { continue; }
        
        let attacks = get_bishop_attacks(from, occupancy) & !our_pieces;
        add_moves_from_bitboard(from, attacks, moves);
    }
}

fn generate_rook_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    let occupancy = pos.all_pieces();
    let our_pieces = pos.pieces(color);
    
    let mut rooks_bb = rooks;
    while rooks_bb != 0 {
        let from = rooks_bb.trailing_zeros() as u8;
        rooks_bb &= rooks_bb - 1;
        
        if from >= 64 { continue; }
        
        let attacks = get_rook_attacks(from, occupancy) & !our_pieces;
        add_moves_from_bitboard(from, attacks, moves);
    }
}

fn generate_queen_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let queens = pos.pieces_colored(PieceType::Queen, color);
    let occupancy = pos.all_pieces();
    let our_pieces = pos.pieces(color);
    
    let mut queens_bb = queens;
    while queens_bb != 0 {
        let from = queens_bb.trailing_zeros() as u8;
        queens_bb &= queens_bb - 1;
        
        if from >= 64 { continue; }
        
        let attacks = get_queen_attacks(from, occupancy) & !our_pieces;
        add_moves_from_bitboard(from, attacks, moves);
    }
}

fn generate_king_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let opponent = color.opposite();
    let from = pos.king_square(color);
    
    
    if from >= 64 {
        return;
    }
    
    let our_pieces = pos.pieces(color);
    let opponent_attacks = all_attacks_for_king(pos, opponent);
    
    let attacks = get_king_attacks(from);
    let targets = attacks & !our_pieces & !opponent_attacks;
    add_moves_from_bitboard(from, targets, moves);
    
    
    if (opponent_attacks & (1u64 << from)) != 0 {
        return;
    }
    
    generate_castling_moves(pos, from, color, opponent_attacks, moves);
}

fn generate_king_moves_ovi(pos: &Position, moves: &mut Vec<Move>) -> u64 {
    let color = pos.side_to_move;
    let opponent = color.opposite();
    let from = pos.king_square(color);
    
    
    if from >= 64 {
        return 0;
    }
    
    let our_pieces = pos.pieces(color);
    let opponent_attacks = all_attacks_for_king(pos, opponent);
    
    let attacks = get_king_attacks(from);
    let targets = attacks & !our_pieces & !opponent_attacks;
    add_moves_from_bitboard(from, targets, moves);
    
    
    if (opponent_attacks & (1u64 << from)) != 0 {
        return opponent_attacks;
    }
    
    generate_castling_moves(pos, from, color, opponent_attacks, moves);
    
    opponent_attacks
}

#[inline(always)]
fn generate_castling_moves(
    pos: &Position,
    king_sq: u8,
    color: Color,
    opponent_attacks: u64,
    moves: &mut Vec<Move>
) {
    
    if king_sq >= 64 {
        return;
    }
    
    let occupancy = pos.all_pieces();
    
    let (kingside_right, queenside_right,
         ks_empty, ks_safe, ks_to,
         qs_empty, qs_safe, qs_to) = match color {
        Color::White => (
            CASTLE_WK, CASTLE_WQ,
            0x60u64, 0x60u64, 6u8,      
            0x0Eu64, 0x0Cu64, 2u8       
        ),
        Color::Black => (
            CASTLE_BK, CASTLE_BQ,
            0x6000000000000000u64, 0x6000000000000000u64, 62u8,
            0x0E00000000000000u64, 0x0C00000000000000u64, 58u8
        ),
    };
    
    
    if (pos.castling_rights & kingside_right) != 0 {
        if (occupancy & ks_empty) == 0 && (opponent_attacks & ks_safe) == 0 {
            moves.push(Move::new(king_sq, ks_to, MoveType::Castle, PieceType::None));
        }
    }
    
    
    if (pos.castling_rights & queenside_right) != 0 {
        if (occupancy & qs_empty) == 0 && (opponent_attacks & qs_safe) == 0 {
            moves.push(Move::new(king_sq, qs_to, MoveType::Castle, PieceType::None));
        }
    }
}

#[inline(always)]
fn is_path_attacked(pos: &Position, path: Bitboard, king_sq: u8, by_color: Color) -> bool {
    let mut squares_bb = path;
    while squares_bb != 0 {
        let sq = squares_bb.trailing_zeros() as u8;
        squares_bb &= squares_bb - 1;
        
        if sq != king_sq && sq < 64 && pos.is_square_attacked(sq, by_color) {
            return true;
        }
    }
    false
}