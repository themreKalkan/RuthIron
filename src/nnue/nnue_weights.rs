//THIS ENGINE USING NNUE OF STOCKFISH HALFKP

use std::fs::File;
use std::io::{Read, BufReader, Cursor};
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt};

#[cfg(feature = "embedded_nnue")]
pub static EMBEDDED_NNUE: &[u8] = include_bytes!("nn-62ef826d1a6d.nnue");

#[cfg(not(feature = "embedded_nnue"))]
pub static EMBEDDED_NNUE: &[u8] = &[];

pub const FT_IN_DIMS: usize = 41024;
pub const FT_OUT_DIMS: usize = 256;
pub const L1_IN_DIMS: usize = 512;
pub const L1_OUT_DIMS: usize = 32;
pub const L2_IN_DIMS: usize = 32;
pub const L2_OUT_DIMS: usize = 32;
pub const OUTPUT_DIMS: usize = 1;

const HALFKP_HASH: u32 = 0x5d69d5b8;


#[repr(C, align(64))]
pub struct NNUEWeights {
    pub ft_biases: Vec<i16>,
    pub ft_weights: Vec<i16>,
    pub l1_biases: Vec<i32>,
    pub l1_weights: Vec<i8>,
    pub l2_biases: Vec<i32>,
    pub l2_weights: Vec<i8>,
    pub output_biases: Vec<i32>,
    pub output_weights: Vec<i8>,
}

impl Clone for NNUEWeights {
    fn clone(&self) -> Self {
        Self {
            ft_biases: self.ft_biases.clone(),
            ft_weights: self.ft_weights.clone(),
            l1_biases: self.l1_biases.clone(),
            l1_weights: self.l1_weights.clone(),
            l2_biases: self.l2_biases.clone(),
            l2_weights: self.l2_weights.clone(),
            output_biases: self.output_biases.clone(),
            output_weights: self.output_weights.clone(),
        }
    }
}

impl NNUEWeights {
    pub fn new() -> Self {
        Self {
            ft_biases: vec![],
            ft_weights: vec![],
            l1_biases: vec![],
            l1_weights: vec![],
            l2_biases: vec![],
            l2_weights: vec![],
            output_biases: vec![],
            output_weights: vec![],
        }
    }

    
    
    pub fn load_embedded() -> Result<Self, Box<dyn std::error::Error>> {
        if EMBEDDED_NNUE.is_empty() {
            return Err("NNUE not embedded. Use --features embedded_nnue or load_from_file()".into());
        }
        Self::load_from_bytes(EMBEDDED_NNUE)
    }
    
    
    
    pub fn load_auto(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        
        if !EMBEDDED_NNUE.is_empty() {
            return Self::load_from_bytes(EMBEDDED_NNUE);
        }
        
        
        Self::load_from_file(file_path)
    }
    
    
    pub fn load_from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = Cursor::new(data);
        Self::load_from_reader(&mut reader)
    }

    
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);
        Self::load_from_reader(&mut reader)
    }
    
    
    fn load_from_reader<R: Read>(reader: &mut R) -> Result<Self, Box<dyn std::error::Error>> {
        
        let version = reader.read_u32::<LittleEndian>()?;
        
        if version != 0x7AF32F16 && version != 0x7AF32F17 {
            return Err(format!("Unknown NNUE version: 0x{:08X}", version).into());
        }
        
        let _hash = reader.read_u32::<LittleEndian>()?;
        let desc_len = reader.read_u32::<LittleEndian>()? as usize;
        
        
        let mut desc = vec![0u8; desc_len];
        reader.read_exact(&mut desc)?;

        
        let ft_hash = reader.read_u32::<LittleEndian>()?;
        
        if ft_hash != HALFKP_HASH {
            eprintln!("WARNING: FT hash doesn't match HalfKP!");
        }

        
        let mut ft_biases = vec![0i16; FT_OUT_DIMS];
        reader.read_i16_into::<LittleEndian>(&mut ft_biases)?;

        
        let mut ft_weights = vec![0i16; FT_IN_DIMS * FT_OUT_DIMS];
        reader.read_i16_into::<LittleEndian>(&mut ft_weights)?;

        
        let _net_hash = reader.read_u32::<LittleEndian>()?;

        
        let mut l1_biases = vec![0i32; L1_OUT_DIMS];
        reader.read_i32_into::<LittleEndian>(&mut l1_biases)?;

        
        let mut l1_weights = vec![0i8; L1_OUT_DIMS * L1_IN_DIMS];
        reader.read_exact(unsafe {
            std::slice::from_raw_parts_mut(l1_weights.as_mut_ptr() as *mut u8, l1_weights.len())
        })?;

        
        let mut l2_biases = vec![0i32; L2_OUT_DIMS];
        reader.read_i32_into::<LittleEndian>(&mut l2_biases)?;

        
        let mut l2_weights = vec![0i8; L2_OUT_DIMS * L2_IN_DIMS];
        reader.read_exact(unsafe {
            std::slice::from_raw_parts_mut(l2_weights.as_mut_ptr() as *mut u8, l2_weights.len())
        })?;

        
        let mut output_biases = vec![0i32; OUTPUT_DIMS];
        reader.read_i32_into::<LittleEndian>(&mut output_biases)?;

        
        let mut output_weights = vec![0i8; L2_OUT_DIMS * OUTPUT_DIMS];
        reader.read_exact(unsafe {
            std::slice::from_raw_parts_mut(output_weights.as_mut_ptr() as *mut u8, output_weights.len())
        })?;

        Ok(Self {
            ft_biases,
            ft_weights,
            l1_biases,
            l1_weights,
            l2_biases,
            l2_weights,
            output_biases,
            output_weights,
        })
    }
    
    
    #[inline(always)]
    pub fn is_embedded() -> bool {
        !EMBEDDED_NNUE.is_empty()
    }
}