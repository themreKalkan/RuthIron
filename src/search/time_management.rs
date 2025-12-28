use std::time::{Instant, Duration};

const MOVE_OVERHEAD_MS: u64 = 50;

const MIN_DEPTH_LIMIT: u32 = 12;

#[derive(Debug, Clone)]
pub struct TimeManager {
    start_time: Instant,
    
    optimum_time: Duration,
    
    maximum_time: Duration,
    
    move_time: Option<Duration>,
    
    max_nodes: Option<u64>,
    
    infinite: bool,
    
    white_time: Option<u64>,
    
    black_time: Option<u64>,
    
    white_increment: Option<u64>,
    
    black_increment: Option<u64>,
    
    moves_to_go: Option<u32>,
    
    is_white_to_move: bool,
    
    ply: u32,
    
    min_depth: u32,
}

impl TimeManager {
    
    pub fn infinite() -> Self {
        Self {
            start_time: Instant::now(),
            optimum_time: Duration::from_secs(3600 * 24),
            maximum_time: Duration::from_secs(3600 * 24),
            move_time: None,
            max_nodes: None,
            infinite: true,
            white_time: None,
            black_time: None,
            white_increment: None,
            black_increment: None,
            moves_to_go: None,
            is_white_to_move: true,
            ply: 0,
            min_depth: MIN_DEPTH_LIMIT,
        }
    }

    pub fn is_infinite(&self) -> bool {
        self.infinite
    }

    
    pub fn new(
        wtime: Option<u32>,
        btime: Option<u32>,
        movetime: Option<u32>,
        moves_to_go: Option<u32>,
        max_nodes: Option<u64>,
        infinite: bool,
    ) -> Self {
        Self::new_with_increment(
            wtime, btime, movetime,
            None, None,
            moves_to_go, max_nodes, infinite,
            true, 0,
        )
    }

    pub fn new_with_increment(
        wtime: Option<u32>,
        btime: Option<u32>,
        movetime: Option<u32>,
        winc: Option<u32>,
        binc: Option<u32>,
        moves_to_go: Option<u32>,
        max_nodes: Option<u64>,
        infinite: bool,
        is_white_to_move: bool,
        ply: u32,
    ) -> Self {
        if infinite {
            return Self::infinite();
        }

        let start_time = Instant::now();

        
        if let Some(mt) = movetime {
            let move_duration = Duration::from_millis(mt as u64);
            return Self {
                start_time,
                optimum_time: move_duration,
                maximum_time: move_duration,
                move_time: Some(move_duration),
                max_nodes,
                infinite: false,
                white_time: wtime.map(|t| t as u64),
                black_time: btime.map(|t| t as u64),
                white_increment: winc.map(|t| t as u64),
                black_increment: binc.map(|t| t as u64),
                moves_to_go,
                is_white_to_move,
                ply,
                min_depth: MIN_DEPTH_LIMIT,
            };
        }

        
        let (optimum, maximum) = Self::calculate_time(
            wtime.map(|t| t as u64),
            btime.map(|t| t as u64),
            winc.map(|t| t as u64),
            binc.map(|t| t as u64),
            moves_to_go,
            is_white_to_move,
            ply,
        );

        Self {
            start_time,
            optimum_time: optimum,
            maximum_time: maximum,
            move_time: None,
            max_nodes,
            infinite: false,
            white_time: wtime.map(|t| t as u64),
            black_time: btime.map(|t| t as u64),
            white_increment: winc.map(|t| t as u64),
            black_increment: binc.map(|t| t as u64),
            moves_to_go,
            is_white_to_move,
            ply,
            min_depth: MIN_DEPTH_LIMIT,
        }
    }

    
    
    fn calculate_time(
        wtime: Option<u64>,
        btime: Option<u64>,
        winc: Option<u64>,
        binc: Option<u64>,
        moves_to_go: Option<u32>,
        is_white_to_move: bool,
        ply: u32,
    ) -> (Duration, Duration) {
        let my_time = if is_white_to_move { wtime } else { btime };
        let my_inc = if is_white_to_move { winc } else { binc }.unwrap_or(0);

        let time_ms = match my_time {
            Some(t) => t,
            None => return (Duration::from_millis(1000), Duration::from_millis(5000)),
        };

        
        if time_ms < 500 {
            let opt = (time_ms / 4).max(10);
            return (Duration::from_millis(opt), Duration::from_millis(opt * 2));
        }

        
        let safe_time = time_ms.saturating_sub(MOVE_OVERHEAD_MS);

        
        let moves_left = if let Some(mtg) = moves_to_go {
            mtg.max(1) as u64
        } else {
            
            
            let estimated_total_moves = 50u64;
            let moves_played = (ply as u64) / 2;
            estimated_total_moves.saturating_sub(moves_played).max(10)
        };

        let (optimum_ms, maximum_ms) = if moves_to_go.is_some() {
            
            let base_time = safe_time / moves_left;
            let optimum = base_time.min(safe_time / 2);
            let maximum = (optimum * 3).min(safe_time * 3 / 4);
            (optimum, maximum)
        } else {
            
            
            
            let base_time = safe_time / moves_left;
            
            
            let inc_bonus = (my_inc * 3) / 4; 
            
            
            let optimum = (base_time + inc_bonus)
                .min(safe_time / 5)  
                .max(100);           
            
            
            let maximum = (optimum * 4)
                .min(safe_time / 2)  
                .max(optimum);
            
            (optimum, maximum)
        };

        
        
        let ply_factor = if ply < 10 {
            0.8  
        } else if ply < 40 {
            1.0  
        } else {
            1.1  
        };

        let optimum_ms = ((optimum_ms as f64) * ply_factor) as u64;
        let maximum_ms = ((maximum_ms as f64) * ply_factor) as u64;

        
        let optimum_ms = optimum_ms.max(50).min(safe_time / 3);
        let maximum_ms = maximum_ms.max(optimum_ms).min(safe_time / 2);

        (
            Duration::from_millis(optimum_ms),
            Duration::from_millis(maximum_ms),
        )
    }

    
    
    pub fn should_stop(&self, start: Instant, current_depth: u32) -> bool {
        
        if current_depth < self.min_depth {
            return false;
        }

        if self.infinite {
            return false;
        }

        let elapsed = start.elapsed();
        elapsed >= self.optimum_time
    }

    
    
    pub fn must_stop(&self, start: Instant, current_depth: u32) -> bool {
        
        if current_depth < self.min_depth {
            return false;
        }

        if self.infinite {
            return false;
        }

        let elapsed = start.elapsed();
        elapsed >= self.maximum_time
    }

    
    
    pub fn can_extend(&self, start: Instant, extension_factor: f64) -> bool {
        if self.infinite {
            return true;
        }

        let elapsed = start.elapsed();
        let extended_time = Duration::from_millis(
            (self.optimum_time.as_millis() as f64 * extension_factor) as u64
        );
        
        
        let capped_time = extended_time.min(self.maximum_time);
        elapsed < capped_time
    }

    
    
    
    pub fn should_stop_early(&self, start: Instant, best_move_stability: f64, current_depth: u32) -> bool {
        
        if current_depth < self.min_depth {
            return false;
        }

        if self.infinite {
            return false;
        }

        let elapsed = start.elapsed();
        
        
        
        let early_stop_factor = 0.5 + (0.5 * best_move_stability);
        let early_stop_time = Duration::from_millis(
            (self.optimum_time.as_millis() as f64 * early_stop_factor) as u64
        );
        
        elapsed >= early_stop_time
    }

    
    pub fn set_min_depth(&mut self, min_depth: u32) {
        self.min_depth = min_depth;
    }

    
    pub fn get_min_depth(&self) -> u32 {
        self.min_depth
    }

    
    pub fn optimum_ms(&self) -> u64 {
        self.optimum_time.as_millis() as u64
    }

    
    pub fn maximum_ms(&self) -> u64 {
        self.maximum_time.as_millis() as u64
    }

    
    pub fn allocated_ms(&self) -> u64 {
        self.optimum_ms()
    }

    
    pub fn remaining_time_ms(&self) -> Option<u64> {
        if self.is_white_to_move {
            self.white_time
        } else {
            self.black_time
        }
    }

    
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    
    pub fn set_side_to_move(&mut self, is_white: bool) {
        self.is_white_to_move = is_white;
    }

    
    pub fn set_ply(&mut self, ply: u32) {
        self.ply = ply;
    }

    
    pub fn new_move(&mut self, is_white_to_move: bool, ply: u32) {
        if self.infinite || self.move_time.is_some() {
            self.start_time = Instant::now();
            return;
        }

        self.is_white_to_move = is_white_to_move;
        self.ply = ply;
        self.start_time = Instant::now();

        let (optimum, maximum) = Self::calculate_time(
            self.white_time,
            self.black_time,
            self.white_increment,
            self.black_increment,
            self.moves_to_go,
            is_white_to_move,
            ply,
        );

        self.optimum_time = optimum;
        self.maximum_time = maximum;
    }

    
    pub fn update_remaining_time(&mut self, wtime: Option<u64>, btime: Option<u64>) {
        self.white_time = wtime;
        self.black_time = btime;
    }

    
    pub fn debug_info(&self) -> String {
        format!(
            "TimeManager {{ optimum: {}ms, maximum: {}ms, elapsed: {}ms, infinite: {}, min_depth: {} }}",
            self.optimum_ms(),
            self.maximum_ms(),
            self.elapsed_ms(),
            self.infinite,
            self.min_depth
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_5_plus_3_game() {
        
        let tm = TimeManager::new_with_increment(
            Some(300_000), 
            Some(300_000),
            None,          
            Some(3_000),   
            Some(3_000),
            None,          
            None,          
            false,         
            true,          
            0,             
        );

        println!("5+3 başlangıç: {}", tm.debug_info());
        
        
        
        
        assert!(tm.optimum_ms() < 15_000, "İlk hamle için çok fazla zaman: {}ms", tm.optimum_ms());
        assert!(tm.optimum_ms() > 1_000, "İlk hamle için çok az zaman: {}ms", tm.optimum_ms());
    }

    #[test]
    fn test_5_plus_3_mid_game() {
        
        let tm = TimeManager::new_with_increment(
            Some(180_000), 
            Some(180_000),
            None,
            Some(3_000),
            Some(3_000),
            None,
            None,
            false,
            true,
            40, 
        );

        println!("5+3 mid-game: {}", tm.debug_info());
        
        
        assert!(tm.optimum_ms() < 20_000, "Orta oyunda çok fazla zaman: {}ms", tm.optimum_ms());
    }

    #[test]
    fn test_low_time() {
        
        let tm = TimeManager::new_with_increment(
            Some(10_000),
            Some(10_000),
            None,
            Some(3_000),
            Some(3_000),
            None,
            None,
            false,
            true,
            60,
        );

        println!("Düşük zaman: {}", tm.debug_info());
        
        
        assert!(tm.optimum_ms() < 5_000, "Düşük zamanda çok yavaş: {}ms", tm.optimum_ms());
    }

    #[test]
    fn test_very_low_time() {
        
        let tm = TimeManager::new_with_increment(
            Some(1_000),
            Some(1_000),
            None,
            Some(100),
            Some(100),
            None,
            None,
            false,
            true,
            80,
        );

        println!("Çok düşük zaman: {}", tm.debug_info());
        
        
        assert!(tm.optimum_ms() < 500, "Çok düşük zamanda yavaş: {}ms", tm.optimum_ms());
    }

    #[test]
    fn test_40_moves_in_5_minutes() {
        
        let tm = TimeManager::new_with_increment(
            Some(300_000),
            Some(300_000),
            None,
            None,
            None,
            Some(40), 
            None,
            false,
            true,
            0,
        );

        println!("40 hamle/5dk: {}", tm.debug_info());
        
        
        assert!(tm.optimum_ms() < 12_000, "Çok fazla zaman ayrılmış");
        assert!(tm.optimum_ms() > 2_000, "Çok az zaman ayrılmış");
    }

    #[test]
    fn test_depth_limit() {
        
        let tm = TimeManager::new_with_increment(
            Some(300_000),
            Some(300_000),
            None,
            Some(3_000),
            Some(3_000),
            None,
            None,
            false,
            true,
            0,
        );

        let start = Instant::now();
        
        
        assert!(!tm.should_stop(start, 1), "Derinlik 1'de durmamalı");
        assert!(!tm.should_stop(start, 5), "Derinlik 5'te durmamalı");
        assert!(!tm.should_stop(start, 10), "Derinlik 10'da durmamalı");
        assert!(!tm.should_stop(start, 13), "Derinlik 13'te durmamalı");
        
        
        
        
        println!("Derinlik 14'te should_stop: {}", tm.should_stop(start, 14));
        println!("Derinlik 20'de should_stop: {}", tm.should_stop(start, 20));
    }

    #[test]
    fn test_must_stop_depth_limit() {
        let tm = TimeManager::new_with_increment(
            Some(300_000),
            Some(300_000),
            None,
            Some(3_000),
            Some(3_000),
            None,
            None,
            false,
            true,
            0,
        );

        let start = Instant::now();
        
        
        assert!(!tm.must_stop(start, 1), "Derinlik 1'de must_stop false olmalı");
        assert!(!tm.must_stop(start, 13), "Derinlik 13'te must_stop false olmalı");
    }

    #[test]
    fn test_custom_min_depth() {
        let mut tm = TimeManager::new_with_increment(
            Some(300_000),
            Some(300_000),
            None,
            Some(3_000),
            Some(3_000),
            None,
            None,
            false,
            true,
            0,
        );

        
        assert_eq!(tm.get_min_depth(), 14);

        
        tm.set_min_depth(10);
        assert_eq!(tm.get_min_depth(), 10);

        let start = Instant::now();
        
        
        assert!(!tm.should_stop(start, 9), "Derinlik 9'da durmamalı");
        
    }

}
