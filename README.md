# RuthIron  
**ML-Supported UCI Chess Engine (Rust + Python)**

## Overview
**RuthIron** is a UCI-compatible chess engine written primarily in **Rust**, enhanced with a **machine-learning–assisted evaluation pipeline**.  
It is the advanced successor of **RuthChess**, designed to be significantly **faster, stronger, and more efficient** than earlier versions.

The engine combines a classical alpha-beta search with **NNUE-based evaluation** and a custom-trained ML model to improve positional understanding and move quality.


## Architecture

### Core Engine
- Language: **Rust**
- Protocol: **UCI**
- Search: Alpha-Beta with modern pruning techniques
- Fallback: Classical handcrafted evaluation when NNUE is unavailable

### NNUE Evaluation
- Architecture: **Stockfish NNUE (HalfKP)**
- Model: Stockfish-compatible NNUE network
- Purpose: High-performance incremental evaluation
- Status: Fully integrated and optional at runtime

If the NNUE file is missing, the engine automatically switches to classical evaluation.

## Machine Learning Component

### Training
- Framework: **Python / PyTorch**
- Environment: **Google Colab**
- Hardware: **NVIDIA A100**
- Dataset: **180M+ unique chess positions**
- Objective: Improve positional evaluation and move selection quality

### Model Deployment
- Initial format: `.pt`
- Runtime format: **ONNX**
- Precision: **FP16**
- INT8 quantization was tested but showed no measurable performance improvement

The ONNX model is optimized for inference and integrated into the Rust engine.

## Usage

1. Download **`RuthChessOVI.exe`**
2. Place the engine executable and the **NNUE file** in the same directory
3. Load the engine in any UCI-compatible GUI:
   - CuteChess
   - ChessBase
   - Arena
   - etc.

If the NNUE file is not detected, the engine will run using classical evaluation.


## Lichess Bot

Lichess profile:  
https://lichess.org/@/RuthIron

- Primarily used for **debugging, testing, and experimentation**
- The bot is relatively new and still under active development
- Match requests are accepted when the bot is online

Planned:
- Deployment on **Raspberry Pi 5**
- 24/7 availability for public challenges

## Notes
- RuthIron is under continuous development
- Search quality, pruning stability, and evaluation accuracy are actively being improved
- Contributions, testing, and technical feedback are welcome
