use criterion::{black_box, criterion_group, criterion_main, Criterion};


use RuthChessOVI::iron::model::{init_model, get_best_moves}; 
use RuthChessOVI::board::position::Position;

fn onnx_model_benchmark(c: &mut Criterion) {
    
    
    println!("Model başlatılıyor...");
    if let Err(e) = init_model() {
        panic!("Model yüklenemedi: {}", e);
    }
    println!("Model hazır, test başlıyor...");

    let pos = black_box(Position::startpos());

    
    let mut group = c.benchmark_group("Chess AI Model");
    
    
    group.sample_size(20); 
    
    
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("get_best_moves (startpos, count=3)", |b| {
        b.iter(|| {
            
            let moves = get_best_moves(&pos, 3);
            black_box(moves);
        });
    });

    group.finish();
}

criterion_group!(
    name = model_benches;
    config = Criterion::default();
    targets = onnx_model_benchmark
);

criterion_main!(model_benches);