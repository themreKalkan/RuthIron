use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize};
use RuthChessOVI::board::position::{Position, Move};
use RuthChessOVI::eval::evaluate::{
    evaluate, evaluate_int, evaluate_fast, init_eval, 
    evaluate_nnue, evaluate_nnue_detailed,
    evaluate_material_fast, evaluate_int_classical,
    nnue_refresh, nnue_push_move, nnue_pop_move,
    is_nnue_enabled, get_eval_type
};
use RuthChessOVI::board::zobrist;
use RuthChessOVI::movegen::magic;
use RuthChessOVI::board::position::init_attack_tables;
use RuthChessOVI::movegen::legal_moves::generate_legal_moves;
use std::time::Duration;


fn setup_environment() {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    
    if is_nnue_enabled() {
        println!("NNUE enabled: {}", get_eval_type());
    } else {
        println!("Using classical evaluation");
    }
}





fn get_test_positions() -> Vec<(&'static str, &'static str)> {
    vec![
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("italian_opening", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"),
        ("sicilian_middlegame", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"),
        ("complex_middlegame", "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 1"),
        ("queen_endgame", "8/8/4k3/3q4/3Q4/4K3/8/8 w - - 0 1"),
        ("rook_endgame", "8/8/4k3/8/8/3K4/3R4/8 w - - 0 1"),
        ("pawn_endgame", "8/4k3/4p3/3pPp2/3P1P2/4K3/8/8 w - - 0 1"),
        ("tactical_position", "r1bq1rk1/ppp2ppp/2n5/3p4/2PP4/2N1P3/PP3PPP/R1BQ1RK1 w - - 0 1"),
        ("imbalanced_material", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"),
        ("open_position", "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"),
        ("closed_position", "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1"),
        ("king_safety_test", "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1"),
    ]
}





fn bench_evaluate_full_startpos(c: &mut Criterion) {
    setup_environment();
    let pos = Position::startpos();
    nnue_refresh(&pos);
    
    c.bench_function("eval_full_startpos", |b| {
        b.iter(|| {
            let result = evaluate(black_box(&pos));
            black_box(result);
        });
    });
}

fn bench_evaluate_int_startpos(c: &mut Criterion) {
    setup_environment();
    let pos = Position::startpos();
    nnue_refresh(&pos);
    
    c.bench_function("eval_int_startpos", |b| {
        b.iter(|| {
            let score = evaluate_int(black_box(&pos));
            black_box(score);
        });
    });
}

fn bench_evaluate_fast_startpos(c: &mut Criterion) {
    setup_environment();
    let pos = Position::startpos();
    nnue_refresh(&pos);
    
    c.bench_function("eval_fast_startpos", |b| {
        b.iter(|| {
            let score = evaluate_fast(black_box(&pos));
            black_box(score);
        });
    });
}





fn bench_nnue_evaluate(c: &mut Criterion) {
    setup_environment();
    let pos = Position::startpos();
    nnue_refresh(&pos);
    
    c.bench_function("nnue_evaluate", |b| {
        b.iter(|| {
            let score = evaluate_nnue(black_box(&pos));
            black_box(score);
        });
    });
}



fn bench_nnue_detailed(c: &mut Criterion) {
    setup_environment();
    let pos = Position::startpos();
    nnue_refresh(&pos);
    
    c.bench_function("nnue_detailed", |b| {
        b.iter(|| {
            let detail = evaluate_nnue_detailed(black_box(&pos));
            black_box(detail);
        });
    });
}

fn bench_nnue_refresh(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("nnue_refresh");
    
    for (name, fen) in get_test_positions().iter() {
        let pos = Position::from_fen(fen).unwrap();
        
        group.bench_function(name.to_string(), |b| {
            b.iter(|| {
                nnue_refresh(black_box(&pos));
            });
        });
    }
    
    group.finish();
}

fn bench_material_fast(c: &mut Criterion) {
    setup_environment();
    let pos = Position::startpos();
    
    c.bench_function("material_fast", |b| {
        b.iter(|| {
            let score = evaluate_material_fast(black_box(&pos));
            black_box(score);
        });
    });
}





fn bench_classical_vs_nnue(c: &mut Criterion) {
    setup_environment();
    let pos = Position::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1").unwrap();
    nnue_refresh(&pos);
    
    let mut group = c.benchmark_group("classical_vs_nnue");
    
    group.bench_function("classical", |b| {
        b.iter(|| {
            let score = evaluate_int_classical(black_box(&pos));
            black_box(score);
        });
    });
    
    group.bench_function("nnue", |b| {
        b.iter(|| {
            let score = evaluate_nnue(black_box(&pos));
            black_box(score);
        });
    });
    
    group.finish();
}





fn bench_nnue_incremental_updates(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("nnue_incremental");
    
    
    let pos = Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1").unwrap();
    nnue_refresh(&pos);
    
    let moves = generate_legal_moves(&pos);
    
    if !moves.is_empty() {
        let test_move = moves[0];
        
        group.bench_function("push_move", |b| {
            b.iter(|| {
                nnue_push_move(black_box(test_move), black_box(&pos));
                nnue_pop_move();
            });
        });
        
        group.bench_function("push_pop_sequence", |b| {
            b.iter(|| {
                for _ in 0..10 {
                    nnue_push_move(black_box(test_move), black_box(&pos));
                }
                for _ in 0..10 {
                    nnue_pop_move();
                }
            });
        });
    }
    
    group.finish();
}

fn bench_nnue_move_sequence(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("nnue_move_sequences");
    
    for sequence_length in [1, 5, 10, 20].iter() {
        group.bench_function(format!("depth_{}", sequence_length), |b| {
            b.iter_batched(
                || {
                    let pos = Position::startpos();
                    nnue_refresh(&pos);
                    let moves = generate_legal_moves(&pos);
                    (pos, moves)
                },
                |(mut pos, mut moves)| {
                    for _ in 0..*sequence_length {
                        if moves.is_empty() {
                            break;
                        }
                        let mv = moves[0];
                        nnue_push_move(mv, &pos);
                        pos.make_move(mv);
                        moves = generate_legal_moves(&pos);
                        let _ = evaluate_nnue(&pos);
                    }
                    for _ in 0..*sequence_length {
                        nnue_pop_move();
                    }
                },
                BatchSize::SmallInput
            );
        });
    }
    
    group.finish();
}





fn bench_evaluate_by_phase(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("eval_by_phase");
    
    let phases = vec![
        ("opening", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("early_middlegame", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"),
        ("middlegame", "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 1"),
        ("late_middlegame", "2rq1rk1/pp3ppp/2n5/3p4/3P4/2N2N2/PP3PPP/2RQ1RK1 w - - 0 1"),
        ("early_endgame", "8/4k3/4p3/3pPp2/3P1P2/4K3/8/8 w - - 0 1"),
        ("endgame", "8/8/4k3/8/8/3K4/8/8 w - - 0 1"),
    ];
    
    for (name, fen) in phases {
        let pos = Position::from_fen(fen).unwrap();
        nnue_refresh(&pos);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                let score = evaluate_nnue(black_box(&pos));
                black_box(score);
            });
        });
    }
    
    group.finish();
}





fn bench_evaluate_by_material(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("eval_by_material");
    
    let positions = vec![
        ("equal_material", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("pawn_up", "rnbqkbnr/ppppppp1/8/7p/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("piece_up", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1"),
        ("exchange_up", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"),
        ("queen_vs_pieces", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("minimal_material", "8/8/4k3/8/8/3K4/8/8 w - - 0 1"),
    ];
    
    for (name, fen) in positions {
        let pos = Position::from_fen(fen).unwrap();
        nnue_refresh(&pos);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                let score = evaluate_nnue(black_box(&pos));
                black_box(score);
            });
        });
    }
    
    group.finish();
}





fn bench_evaluate_with_cold_cache(c: &mut Criterion) {
    setup_environment();
    
    let positions: Vec<Position> = get_test_positions()
        .iter()
        .map(|(_, fen)| Position::from_fen(fen).unwrap())
        .collect();
    
    c.bench_function("eval_cold_cache", |b| {
        b.iter(|| {
            for pos in &positions {
                nnue_refresh(pos);
                let score = evaluate_nnue(black_box(pos));
                black_box(score);
            }
        });
    });
}

fn bench_evaluate_with_warm_cache(c: &mut Criterion) {
    setup_environment();
    
    let pos = Position::startpos();
    nnue_refresh(&pos);
    
    
    for _ in 0..1000 {
        let _ = evaluate_int(&pos);
    }
    
    c.bench_function("eval_warm_cache", |b| {
        b.iter(|| {
            let score = evaluate_int(black_box(&pos));
            black_box(score);
        });
    });
}

fn bench_cache_hit_rate(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("cache_performance");
    
    let positions: Vec<Position> = vec![
        Position::startpos(),
        Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1").unwrap(),
    ];
    
    for pos in &positions {
        nnue_refresh(pos);
    }
    
    group.bench_function("repeated_same_position", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let score = evaluate_int(black_box(&positions[0]));
                black_box(score);
            }
        });
    });
    
    group.bench_function("alternating_positions", |b| {
        b.iter(|| {
            for _ in 0..50 {
                let score1 = evaluate_int(black_box(&positions[0]));
                let score2 = evaluate_int(black_box(&positions[1]));
                black_box((score1, score2));
            }
        });
    });
    
    group.finish();
}





fn bench_evaluate_throughput(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(10));
    
    let pos = Position::startpos();
    nnue_refresh(&pos);
    
    for count in [100, 1000, 10000, 100000].iter() {
        group.throughput(criterion::Throughput::Elements(*count as u64));
        
        group.bench_with_input(BenchmarkId::new("nnue", count), count, |b, &count| {
            b.iter(|| {
                for _ in 0..count {
                    let score = evaluate_nnue(black_box(&pos));
                    black_box(score);
                }
            });
        });
        
        
        
        group.bench_with_input(BenchmarkId::new("material_fast", count), count, |b, &count| {
            b.iter(|| {
                for _ in 0..count {
                    let score = evaluate_material_fast(black_box(&pos));
                    black_box(score);
                }
            });
        });
    }
    
    group.finish();
}

fn bench_nodes_per_second(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("nodes_per_second");
    group.measurement_time(Duration::from_secs(5));
    
    let positions: Vec<Position> = get_test_positions()
        .iter()
        .take(5)
        .map(|(_, fen)| Position::from_fen(fen).unwrap())
        .collect();
    
    for pos in &positions {
        nnue_refresh(pos);
    }
    
    group.bench_function("mixed_positions", |b| {
        b.iter(|| {
            for pos in &positions {
                let score = evaluate_nnue(black_box(pos));
                black_box(score);
            }
        });
    });
    
    group.finish();
}





fn bench_evaluate_by_complexity(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("eval_by_complexity");
    
    let positions = vec![
        ("simple", "8/8/4k3/8/8/3K4/8/8 w - - 0 1"),
        ("moderate", "8/4k3/4p3/3pPp2/3P1P2/4K3/8/8 w - - 0 1"),
        ("complex", "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 1"),
        ("very_complex", "r1bq1rk1/pp1n1ppp/2pbpn2/6B1/2pP4/2N1PN2/PP1BQPPP/R4RK1 w - - 0 1"),
    ];
    
    for (name, fen) in positions {
        let pos = Position::from_fen(fen).unwrap();
        nnue_refresh(&pos);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                let score = evaluate_nnue(black_box(&pos));
                black_box(score);
            });
        });
    }
    
    group.finish();
}





fn bench_all_eval_functions(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("all_eval_functions");
    
    let pos = Position::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1").unwrap();
    nnue_refresh(&pos);
    
    group.bench_function("evaluate_full", |b| {
        b.iter(|| {
            let result = evaluate(black_box(&pos));
            black_box(result);
        });
    });
    
    group.bench_function("evaluate_int", |b| {
        b.iter(|| {
            let score = evaluate_int(black_box(&pos));
            black_box(score);
        });
    });
    
    group.bench_function("evaluate_fast", |b| {
        b.iter(|| {
            let score = evaluate_fast(black_box(&pos));
            black_box(score);
        });
    });
    
    group.bench_function("evaluate_nnue", |b| {
        b.iter(|| {
            let score = evaluate_nnue(black_box(&pos));
            black_box(score);
        });
    });
    
    
    
    group.bench_function("evaluate_material_fast", |b| {
        b.iter(|| {
            let score = evaluate_material_fast(black_box(&pos));
            black_box(score);
        });
    });
    
    group.bench_function("evaluate_classical", |b| {
        b.iter(|| {
            let score = evaluate_int_classical(black_box(&pos));
            black_box(score);
        });
    });
    
    if is_nnue_enabled() {
        group.bench_function("evaluate_nnue_detailed", |b| {
            b.iter(|| {
                let detail = evaluate_nnue_detailed(black_box(&pos));
                black_box(detail);
            });
        });
    }
    
    group.finish();
}





fn bench_special_positions(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("special_positions");
    
    let special_positions = vec![
        ("promotion_position", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
        ("castling_position", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        ("en_passant", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1"),
        ("pinned_pieces", "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"),
        ("discovered_attack", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"),
    ];
    
    for (name, fen) in special_positions {
        let pos = Position::from_fen(fen).unwrap();
        nnue_refresh(&pos);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                let score = evaluate_nnue(black_box(&pos));
                black_box(score);
            });
        });
    }
    
    group.finish();
}





fn bench_memory_intensive(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("memory_intensive");
    
    let positions: Vec<Position> = (0..100)
        .map(|i| {
            let fen_idx = i % get_test_positions().len();
            Position::from_fen(get_test_positions()[fen_idx].1).unwrap()
        })
        .collect();
    
    for pos in &positions {
        nnue_refresh(pos);
    }
    
    group.bench_function("eval_100_positions", |b| {
        b.iter(|| {
            for pos in &positions {
                let score = evaluate_nnue(black_box(pos));
                black_box(score);
            }
        });
    });
    
    group.bench_function("refresh_100_positions", |b| {
        b.iter(|| {
            for pos in &positions {
                nnue_refresh(black_box(pos));
            }
        });
    });
    
    group.finish();
}





fn bench_accuracy_vs_speed(c: &mut Criterion) {
    setup_environment();
    
    let mut group = c.benchmark_group("accuracy_vs_speed");
    
    let pos = Position::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1").unwrap();
    nnue_refresh(&pos);
    
    
    group.bench_function("1_material_only", |b| {
        b.iter(|| {
            let score = evaluate_material_fast(black_box(&pos));
            black_box(score);
        });
    });
    
   
    
    
    group.bench_function("3_fast_eval", |b| {
        b.iter(|| {
            let score = evaluate_fast(black_box(&pos));
            black_box(score);
        });
    });
    
    
    group.bench_function("4_nnue_full", |b| {
        b.iter(|| {
            let score = evaluate_nnue(black_box(&pos));
            black_box(score);
        });
    });
    
    
    if is_nnue_enabled() {
        group.bench_function("5_nnue_detailed", |b| {
            b.iter(|| {
                let detail = evaluate_nnue_detailed(black_box(&pos));
                black_box(detail);
            });
        });
    }
    
    group.finish();
}





criterion_group!(
    name = basic_benchmarks;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = 
        bench_evaluate_full_startpos,
        bench_evaluate_int_startpos,
        bench_evaluate_fast_startpos,
);

criterion_group!(
    name = nnue_benchmarks;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = 
        bench_nnue_evaluate,
        bench_nnue_detailed,
        bench_nnue_refresh,
        bench_material_fast,
);

criterion_group!(
    name = comparison_benchmarks;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = 
        bench_classical_vs_nnue,
        bench_all_eval_functions,
        bench_accuracy_vs_speed,
);

criterion_group!(
    name = incremental_benchmarks;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = 
        bench_nnue_incremental_updates,
        bench_nnue_move_sequence,
);

criterion_group!(
    name = phase_benchmarks;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = 
        bench_evaluate_by_phase,
        bench_evaluate_by_material,
        bench_evaluate_by_complexity,
);

criterion_group!(
    name = cache_benchmarks;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = 
        bench_evaluate_with_cold_cache,
        bench_evaluate_with_warm_cache,
        bench_cache_hit_rate,
);

criterion_group!(
    name = throughput_benchmarks;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(10));
    targets = 
        bench_evaluate_throughput,
        bench_nodes_per_second,
);

criterion_group!(
    name = special_benchmarks;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = 
        bench_special_positions,
        bench_memory_intensive,
);

criterion_main!(
    basic_benchmarks,
    nnue_benchmarks,
    comparison_benchmarks,
    incremental_benchmarks,
    phase_benchmarks,
    cache_benchmarks,
    throughput_benchmarks,
    special_benchmarks,
);