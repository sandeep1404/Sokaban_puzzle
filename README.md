# Track 1 — Tree-Based Planning with LLMs (Sokoban)

A tree-based planning system that uses an LLM as a one-step action predictor to solve Sokoban puzzles. The LLM (Qwen2.5-7B-Instruct via vLLM) guides search by predicting the best next action at each board state. Four search algorithms are implemented: BFS (baseline), Beam Search, A*, and MCTS.

---

## Requirements

- Python 3.10+
- AMD GPU with ROCm 6.x (tested on MI300X, 192 GB VRAM) — or any CUDA GPU with ≥16 GB VRAM
- PreInstalled vLLM Image from amd digital ocean([Link](https://amd.digitalocean.com/))

<img width="1917" height="1001" alt="image" src="https://github.com/user-attachments/assets/fdda4414-3611-4d5e-a6d2-3212145015d8" />


---

## Installation

```bash
pip install -r requirements.txt
```

---

## Starting the vLLM Server

All LLM-guided solvers require the vLLM inference server to be running first.

```bash
bash start_vllm.sh
```

This starts `Qwen/Qwen2.5-7B-Instruct` on port 8000 with ROCm backend. Wait until you see:
```
INFO:     Application startup complete.
```

To verify the server is running:
```bash
curl http://localhost:8000/v1/models
```

> **Note:** BFS (`--solver bfs`) does not require the server.

---

## File Structure

```
track_1/
├── parser.py                   # Load and parse Microban.txt puzzle files
├── environment.py              # Sokoban game logic, state, actions, rendering
├── representation.py           # Three LLM state representations (ASCII / Structured / Annotated)
├── llm_predictor.py            # LLM interface — single-action & batch prediction via vLLM
├── search.py                   # BFS, Beam Search, A*, MCTS solvers
├── evaluation.py               # Batch evaluation framework + plots
├── solve_puzzle.py             # Step-by-step single puzzle visualiser
├── compare_representations.py  # Compare ASCII vs Structured vs Annotated on A*
├── compare_solvers.py          # Compare BFS vs Beam Search vs A* vs MCTS side-by-side
├── debug_llm.py                # Inspect LLM predictions step-by-step
├── start_vllm.sh               # Launch vLLM server
├── data/
│   └── Microban.txt            # David Skinner's 155 Sokoban puzzles
├── outputs/                    # Generated plots saved here
└── requirements.txt
```

---

## Running Experiments

### 1. Full evaluation (BFS vs A* vs MCTS)
```bash
python3 evaluation.py
```
Runs all solvers on 10 varied-difficulty puzzles and prints a comparison table + saves plots to `outputs/`.

### 2. Compare search strategies (BFS vs Beam vs A* vs MCTS)
```bash
python3 compare_solvers.py
# With custom budget:
python3 compare_solvers.py --max-states 20000 --max-llm-calls 20000
# With custom MCTS iterations:
python3 compare_solvers.py --mcts-iterations 1000
```

### 3. Compare state representations (ASCII vs Structured vs Annotated)
```bash
python3 compare_representations.py
# Faster test run:
python3 compare_representations.py --puzzles 0 1 5 10 --max-states 5000
```

### 4. Solve a single puzzle step-by-step
```bash
python3 solve_puzzle.py --puzzle 0 --solver bfs
python3 solve_puzzle.py --puzzle 3 --solver astar
python3 solve_puzzle.py --puzzle 7 --solver beam
python3 solve_puzzle.py --puzzle 0 --solver mcts
```
Prints every board state from start to solution.

### 5. Debug LLM predictions
```bash
python3 debug_llm.py --n 5 --puzzles 0 1 2
```
Shows raw LLM output, parsed action, and whether it matches a valid move — step by step.

---

## Puzzle Selection

Puzzles are 0-indexed from Microban.txt (155 total). Default evaluation uses 10 puzzles spanning varying difficulty:

| Index | Difficulty |
|-------|-----------|
| 0, 1  | Easy (2 boxes) |
| 5, 10 | Easy-Medium |
| 20, 30 | Medium |
| 50, 70 | Medium-Hard |
| 90, 110 | Hard |

---

## Model

- **Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Backend:** vLLM (OpenAI-compatible API at `http://localhost:8000/v1`)
- **Inference:** Continuous batching — all states in a batch are sent concurrently
- **GPU:** AMD Instinct MI300X (192 GB HBM3, ROCm 7.2)

---

## Key Design Decisions

- **LLM as one-step predictor only:** The LLM receives the current board state and outputs a single action. It is never used for multi-step lookahead.
- **A* priority:** `f(n) = g(n) − h(n) − llm_boost(n)` where `h` is the sum of Manhattan distances from each unplaced box to its nearest target.
- **Batched inference:** A* pops `batch_size=16` states at wall-clock once and fires all LLM requests concurrently, reducing inferencetime significantly.
- **State caching:** Identical board positions reuse the cached LLM prediction — no duplicate calls.
- **Deadlock detection:** Corner-deadlock pruning eliminates states where a box is struck.
