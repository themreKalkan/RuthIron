use crate::board::position::{Position, Move, Color, PieceType, MoveType};
use crate::movegen::moves::generate_moves_ovi;
use crate::movegen::magic::{
    all_attacks_for_king, get_rook_attacks, get_bishop_attacks,
    get_queen_attacks, get_king_attacks, get_pawn_attacks, get_knight_attacks
};

pub fn generate_legal_moves(pos: &Position) -> Vec<Move> {
    generate_legal_moves_ovi(pos)
}

pub fn generate_legal_moves_ovi(pos: &Position) -> Vec<Move> {
    let our_color = pos.side_to_move;
    let opponent_color = our_color.opposite();
    let king_sq = pos.king_square(our_color);
    
    
    if king_sq >= 64 {
        return Vec::new();
    }
    
    let (pseudo_moves, opponent_attacks) = generate_moves_ovi(pos);
    
    
    let is_in_check = (opponent_attacks & (1u64 << king_sq)) != 0;
    
    let capacity = if is_in_check { 16 } else { pseudo_moves.len().min(64) };
    let mut legal_moves = Vec::with_capacity(capacity);
    
    
    let pinned_pieces = find_pinned_pieces_fast(pos, our_color, king_sq);
    
    
    let checkers = if is_in_check {
        find_checkers(pos, king_sq, opponent_color)
    } else {
        0u64
    };
    
    let checker_count = checkers.count_ones();
    
    for mv in pseudo_moves {
        let from = mv.from();
        let to = mv.to();
        
        
        if from >= 64 || to >= 64 {
            continue;
        }
        
        
        if from == king_sq {
            if is_king_move_safe(pos, &mv, king_sq, opponent_color) {
                legal_moves.push(mv);
            }
            continue;
        }
        
        
        if checker_count > 1 {
            continue;
        }
        
        
        let pin_mask = get_pin_mask(&pinned_pieces, from);
        if pin_mask != !0u64 {
            
            if (pin_mask & (1u64 << to)) == 0 {
                
                continue;
            }
        }
        
        
        if mv.move_type() == MoveType::EnPassant {
            if !is_en_passant_legal(pos, &mv, king_sq, our_color) {
                continue;
            }
        }
        
        
        if is_in_check {
            if !does_move_block_check(&mv, king_sq, checkers, our_color) {
                continue;
            }
        }
        
        legal_moves.push(mv);
    }
    
    legal_moves
}



fn find_pinned_pieces_fast(pos: &Position, color: Color, king_sq: u8) -> [(u8, u64); 8] {
    let mut pinned: [(u8, u64); 8] = [(64, 0); 8]; 
    let mut pin_count = 0;
    
    
    if king_sq >= 64 {
        return pinned;
    }
    
    let opponent = color.opposite();
    let occupancy = pos.all_pieces();
    let our_pieces = pos.pieces(color);
    
    let king_rank = king_sq / 8;
    let king_file = king_sq % 8;
    
    
    let opponent_rooks = pos.pieces_colored(PieceType::Rook, opponent);
    let opponent_bishops = pos.pieces_colored(PieceType::Bishop, opponent);
    let opponent_queens = pos.pieces_colored(PieceType::Queen, opponent);
    
    let rook_queens = opponent_rooks | opponent_queens;
    let bishop_queens = opponent_bishops | opponent_queens;
    
    
    const DIRECTIONS: [(i8, i8); 8] = [
        (0, 1), (0, -1), (1, 0), (-1, 0),  
        (1, 1), (1, -1), (-1, 1), (-1, -1) 
    ];
    
    for (idx, &(dr, df)) in DIRECTIONS.iter().enumerate() {
        let is_diagonal = idx >= 4;
        let attackers = if is_diagonal { bishop_queens } else { rook_queens };
        
        if attackers == 0 {
            continue;
        }
        
        let mut r = king_rank as i8 + dr;
        let mut f = king_file as i8 + df;
        let mut potential_pinned: Option<u8> = None;
        let mut ray_mask = 0u64;
        
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let sq = (r * 8 + f) as u8;
            let sq_bit = 1u64 << sq;
            
            if (occupancy & sq_bit) != 0 {
                if (our_pieces & sq_bit) != 0 {
                    
                    if potential_pinned.is_some() {
                        
                        break;
                    }
                    potential_pinned = Some(sq);
                    
                    ray_mask = sq_bit;
                } else {
                    
                    if (attackers & sq_bit) != 0 && potential_pinned.is_some() {
                        
                        ray_mask |= sq_bit;
                        if pin_count < 8 {
                            pinned[pin_count] = (potential_pinned.unwrap(), ray_mask);
                            pin_count += 1;
                        }
                    }
                    break;
                }
            } else {
                
                if potential_pinned.is_some() {
                    ray_mask |= sq_bit;
                }
            }
            
            r += dr;
            f += df;
        }
    }
    
    pinned
}


#[inline(always)]
fn get_pin_mask(pinned: &[(u8, u64); 8], square: u8) -> u64 {
    for &(pinned_sq, mask) in pinned.iter() {
        if pinned_sq == square {
            return mask;
        }
        if pinned_sq == 64 {
            break; 
        }
    }
    !0u64 
}


fn find_checkers(pos: &Position, king_sq: u8, attacker_color: Color) -> u64 {
    
    if king_sq >= 64 {
        return 0;
    }
    
    let mut checkers = 0u64;
    let occupancy = pos.all_pieces();
    
    
    let pawns = pos.pieces_colored(PieceType::Pawn, attacker_color);
    let pawn_attacks = get_pawn_attacks(king_sq, attacker_color.opposite());
    checkers |= pawns & pawn_attacks;
    
    
    let knights = pos.pieces_colored(PieceType::Knight, attacker_color);
    let knight_attacks = get_knight_attacks(king_sq);
    checkers |= knights & knight_attacks;
    
    
    let bishops = pos.pieces_colored(PieceType::Bishop, attacker_color);
    let bishop_attacks = get_bishop_attacks(king_sq, occupancy);
    checkers |= bishops & bishop_attacks;
    
    
    let rooks = pos.pieces_colored(PieceType::Rook, attacker_color);
    let rook_attacks = get_rook_attacks(king_sq, occupancy);
    checkers |= rooks & rook_attacks;
    
    
    let queens = pos.pieces_colored(PieceType::Queen, attacker_color);
    checkers |= queens & (bishop_attacks | rook_attacks);
    
    checkers
}


fn is_king_move_safe(pos: &Position, mv: &Move, king_sq: u8, opponent: Color) -> bool {
    let to = mv.to();
    let from = mv.from();
    
    
    if to >= 64 || from >= 64 {
        return false;
    }
    
    
    if mv.move_type() == MoveType::Castle {
        return true;
    }
    
    
    let occupancy_without_king = pos.all_pieces() & !(1u64 << from);
    
    
    let opponent_pieces = pos.pieces(opponent);
    let occupancy = if (opponent_pieces & (1u64 << to)) != 0 {
        occupancy_without_king & !(1u64 << to)
    } else {
        occupancy_without_king
    };
    
    let to_bit = 1u64 << to;
    
    
    let pawns = pos.pieces_colored(PieceType::Pawn, opponent);
    let pawn_attacks = get_pawn_attacks(to, opponent.opposite());
    if (pawns & pawn_attacks) != 0 {
        return false;
    }
    
    
    let knights = pos.pieces_colored(PieceType::Knight, opponent);
    let knight_attacks = get_knight_attacks(to);
    if (knights & knight_attacks) != 0 {
        return false;
    }
    
    
    let opp_king_sq = pos.king_square(opponent);
    if opp_king_sq < 64 {
        let king_attacks = get_king_attacks(opp_king_sq);
        if (king_attacks & to_bit) != 0 {
            return false;
        }
    }
    
    
    let bishops = pos.pieces_colored(PieceType::Bishop, opponent);
    let queens = pos.pieces_colored(PieceType::Queen, opponent);
    let bishop_attacks = get_bishop_attacks(to, occupancy);
    if ((bishops | queens) & bishop_attacks) != 0 {
        return false;
    }
    
    
    let rooks = pos.pieces_colored(PieceType::Rook, opponent);
    let rook_attacks = get_rook_attacks(to, occupancy);
    if ((rooks | queens) & rook_attacks) != 0 {
        return false;
    }
    
    true
}


fn is_en_passant_legal(pos: &Position, mv: &Move, king_sq: u8, our_color: Color) -> bool {
    let from = mv.from();
    let to = mv.to();
    
    
    if from >= 64 || to >= 64 || king_sq >= 64 {
        return false;
    }
    
    let opponent = our_color.opposite();
    
    
    let captured_sq = match our_color {
        Color::White => {
            if to < 8 { return false; } 
            to - 8
        },
        Color::Black => {
            if to >= 56 { return false; } 
            to + 8
        },
    };
    
    let king_rank = king_sq / 8;
    let pawn_rank = from / 8;
    
    
    if king_rank != pawn_rank {
        return true;
    }
    
    
    let occupancy = pos.all_pieces() & !(1u64 << from) & !(1u64 << captured_sq);
    
    
    let opponent_rooks = pos.pieces_colored(PieceType::Rook, opponent);
    let opponent_queens = pos.pieces_colored(PieceType::Queen, opponent);
    let rook_queens = opponent_rooks | opponent_queens;
    
    if rook_queens == 0 {
        return true; 
    }
    
    
    let rook_attacks = get_rook_attacks(king_sq, occupancy);
    
    
    let rank_mask = 0xFFu64 << (king_rank * 8);
    if (rook_attacks & rook_queens & rank_mask) != 0 {
        return false; 
    }
    
    true
}


fn does_move_block_check(
    mv: &Move,
    king_sq: u8,
    checkers: u64,
    our_color: Color
) -> bool {
    let to = mv.to();
    
    
    if to >= 64 || king_sq >= 64 {
        return false;
    }
    
    let to_bit = 1u64 << to;
    
    
    if (checkers & to_bit) != 0 {
        return true;
    }
    
    
    if mv.move_type() == MoveType::EnPassant {
        let captured_sq = match our_color {
            Color::White => {
                if to < 8 { return false; }
                to - 8
            },
            Color::Black => {
                if to >= 56 { return false; }
                to + 8
            },
        };
        if (checkers & (1u64 << captured_sq)) != 0 {
            return true;
        }
    }
    
    
    if checkers.count_ones() == 1 {
        let checker_sq = checkers.trailing_zeros() as u8;
        if checker_sq < 64 {
            let block_mask = get_between_mask(king_sq, checker_sq);
            if (block_mask & to_bit) != 0 {
                return true;
            }
        }
    }
    
    false
}


fn get_between_mask(sq1: u8, sq2: u8) -> u64 {
    
    if sq1 >= 64 || sq2 >= 64 || sq1 == sq2 {
        return 0;
    }
    
    let r1 = (sq1 / 8) as i8;
    let f1 = (sq1 % 8) as i8;
    let r2 = (sq2 / 8) as i8;
    let f2 = (sq2 % 8) as i8;
    
    let dr = (r2 - r1).signum();
    let df = (f2 - f1).signum();
    
    
    if dr != 0 && df != 0 {
        let rank_diff = (r2 - r1).abs();
        let file_diff = (f2 - f1).abs();
        if rank_diff != file_diff {
            return 0; 
        }
    }
    
    
    if dr == 0 && df == 0 {
        return 0;
    }
    
    let mut mask = 0u64;
    let mut r = r1 + dr;
    let mut f = f1 + df;
    
    
    let mut safety_counter = 0;
    
    while r >= 0 && r < 8 && f >= 0 && f < 8 && safety_counter < 8 {
        if r == r2 && f == f2 {
            break; 
        }
        mask |= 1u64 << (r * 8 + f);
        r += dr;
        f += df;
        safety_counter += 1;
    }
    
    mask
}


pub fn move_to_uci(mv: Move) -> String {
    let from = mv.from();
    let to = mv.to();
    
    
    if from >= 64 || to >= 64 {
        return String::from("0000");
    }
    
    let mut result = String::with_capacity(5);
    result.push((b'a' + from % 8) as char);
    result.push((b'1' + from / 8) as char);
    result.push((b'a' + to % 8) as char);
    result.push((b'1' + to / 8) as char);
    
    if mv.is_promotion() {
        let promo_char = match mv.promotion() {
            PieceType::Queen => 'q',
            PieceType::Rook => 'r',
            PieceType::Bishop => 'b',
            PieceType::Knight => 'n',
            _ => 'q',
        };
        result.push(promo_char);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_legal_moves_starting_position() {
        let pos = Position::new();
        let moves = generate_legal_moves(&pos);
        assert_eq!(moves.len(), 20);
    }
    
    #[test]
    fn test_between_mask_vertical() {
        
        let mask = get_between_mask(4, 28);
        
        assert_eq!(mask, (1u64 << 12) | (1u64 << 20));
    }
    
    #[test]
    fn test_between_mask_same_square() {
        let mask = get_between_mask(4, 4);
        assert_eq!(mask, 0);
    }
    
    #[test]
    fn test_between_mask_knight_move() {
        
        let mask = get_between_mask(0, 17); 
        assert_eq!(mask, 0);
    }
}