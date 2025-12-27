mod board;
mod eval;
mod movegen;
mod search;
mod uci;
mod nnue;
mod iron;

use crate::board::zobrist;
use crate::movegen::magic;
use crate::eval::evaluate::init_eval;
use crate::eval::evaluate::evaluate;
use RuthChessOVI::board::position::{Position,Move,PieceType,init_attack_tables};
use crate::iron::model;

use RuthChessOVI::nnue::nnue::NNUE;

use std::io::{self, BufRead, BufReader};
use std::sync::Arc;
use std::thread;


use uci::protocol;



fn main() {
    zobrist::init_zobrist();
    magic::init_magics();
    model::init_model();
    init_attack_tables();
    init_eval();

    
    uci::protocol::run_uci();
}
