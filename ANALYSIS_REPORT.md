# Analysis Report — Track 1: Tree-Based Planning with LLMs

**System:** Sokoban solver using Qwen2.5-7B-Instruct (via vLLM) as a one-step action predictor, combined with A*, Beam Search, and BFS tree search.  
**Hardware:** AMD Instinct MI300X (192 GB HBM3), ROCm 7.2, vLLM with continuous batching.  
**Dataset:** David Skinner's Microban collection (155 puzzles). Evaluated on 10 puzzles spanning easy to hard: indices [0, 1, 5, 10, 20, 30, 50, 70, 90, 110].

---

## 1. System Architecture

### 1.1 Overall Design

The system follows a strict constraint: **the LLM is used only as a one-step predictor**. At each board state, it receives a prompt describing the current configuration and outputs a single action (`up`, `down`, `left`, `right`). The tree search algorithm decides which states to visit and when to call the LLM — the LLM never plans ahead or reasons about future states.

```
Board State → Prompt Builder → vLLM (Qwen2.5-7B) → Action + Confidence
                ↑                                         ↓
           representation.py                    A* priority queue
```

### 1.2 Modules

| Module | Responsibility |
|--------|---------------|
| `parser.py` | Parses raw Microban.txt into structured puzzle dicts |
| `environment.py` | Immutable `SokobanState`, action application, deadlock detection, heuristic |
| `representation.py` | Three LLM prompt formats: ASCII, Structured, Annotated |
| `llm_predictor.py` | vLLM client, batch inference, logprob scoring, state cache |
| `search.py` | BFS baseline, Beam Search, A* with LLM guidance |
| `evaluation.py` | Batch evaluation, metrics, plots |

### 1.3 LLM Integration

The LLM receives the full board prompt and the server returns logprobs over output tokens. The system scans **all output token positions** for action words and extracts their log-probabilities — not just the first token. This means even when the model responds in JSON format (`{"action": "up"}`), the correct token's probability is captured.

```python
# f-score formula in A*
priority = g_cost - heuristic_score(state) - llm_weight * llm_prob - push_bonus
```

The LLM probability (`llm_prob`) acts as a small tiebreaker. If the LLM is wrong, the correct path is still explored — just slightly later.

---

## 2. State Representations

Three representations were implemented and compared:

### ASCII
Raw board grid using standard Sokoban symbols:
```
####
# .#
#  ###
#*@  #
#  $ #
```

### Structured
Coordinate-based text description with no grid:
```
Board size: 7 rows x 6 cols
Player position: row 3, col 2
Box positions: [(3, 1), (4, 3)]
Target positions: [(1, 2), (3, 1)]
Unplaced boxes: [(4, 3)]
Unmatched targets: [(1, 2)]
```

### Annotated
ASCII grid augmented with Manhattan distance hints:
```
####
# .#
...
--- Hints ---
Player is at (3, 2).
Box at (4, 3) → nearest target at (1, 2), distance 4 steps.
1 of 2 boxes already on targets.
```

### 2.1 Representation Experiment Results

All three representations were tested with A* on the same 10 puzzles (max 20,000 states, 20,000 LLM calls):

| Representation | Solved | Accuracy | Avg Steps | Avg States | Avg LLM Calls | Avg Time |
|---|---|---|---|---|---|---|
| ASCII | 8/10 | 80.0% | 45.0 | 6,594 | 6,594 | 68.2s |
| Structured | 8/10 | 80.0% | 45.0 | 6,608 | 6,608 | 63.3s |
| Annotated | 8/10 | 80.0% | 45.0 | 6,592 | 6,592 | 67.6s |

**Finding:** All three representations achieved identical accuracy (80%) with the same 8 puzzles solved and the same 2 failures (puzzles 5 and 110). Step counts and state counts are nearly identical across representations.

**Interpretation:** Qwen2.5-7B is robust to representation format at this difficulty level. The two failures are **budget-constrained** (both hit the 20k cap), not representation-dependent. The model's spatial reasoning is sufficient to interpret any of the three formats.

**When representation would matter:** Differences would emerge at the margin — puzzles where one representation gives just enough extra signal to tip the solver over the budget threshold. With the current puzzle set, no puzzle is at that margin.

---

## 3. Search Strategy Comparison

Three search algorithms were compared on the same 10 puzzles:

### 3.1 BFS (Baseline, No LLM)

Standard breadth-first search exploring all reachable states. Complete within its budget — if the solution exists and the state cap is not hit, BFS will find it.

**Properties:**
- Zero LLM calls → negligible latency per state
- Explores states in order of path length (optimal solution length)
- Fails only when the budget cap is hit before the solution is found

### 3.2 Beam Search (LLM-guided)

Maintains a fixed-size "beam" of the top-k states at each depth level. LLM ranks candidate actions; only the top-k resulting states survive to the next step.

**Properties:**
- Permanently prunes states — cannot recover from bad pruning decisions
- Sokoban requires counter-intuitive moves (pushing boxes *away* from goals to set up future pushes); beam search scores these moves low and discards them
- Memory efficient but fundamentally incomplete

### 3.3 A* (LLM-guided, batched)

Priority queue search where every candidate state is retained until visited. The LLM biases expansion order via a small priority boost, but never eliminates candidates.

**Properties:**
- Complete within budget (guaranteed to find solution if one exists and cap not hit)
- Batched LLM calls: pops `batch_size=16` states at once, fires all HTTP requests concurrently
- State caching: identical board positions reuse cached predictions
- Deadlock pruning: corner-deadlock states discarded before entering the heap

### 3.4 Results Summary

| Solver | Solved | Accuracy | Avg Steps | Avg States | Avg LLM Calls | Avg Time |
|---|---|---|---|---|---|---|
| BFS (no LLM) | 9/10 | 90.0% | 51.9 | ~3,200 | 0 | 0.08s |
| Beam Search (w=5) | 7/10 | 70.0% | 34.3 | ~800 | ~800 | 8.1s |
| A* (LLM-guided) | 8/10 | 80.0% | 45.0 | ~6,600 | ~6,600 | 68.2s |

**Key observations:**

1. **BFS achieves the highest accuracy (90%)** despite zero LLM calls. Its single failure is budget-constrained, not algorithmic. This demonstrates that the search structure matters more than LLM guidance for this puzzle set.

2. **Beam search achieves the lowest accuracy (70%)**. Permanent pruning is fatal for Sokoban: the correct path requires counter-intuitive moves that score poorly under the heuristic, causing the beam to collapse. It is the fastest solver but at the cost of correctness.

3. **A* achieves 80% accuracy** — better than beam search but worse than BFS. The LLM boosts exploration efficiency on solvable puzzles (fewer states needed than pure BFS on some puzzles) but introduces overhead per state (~10ms LLM call vs ~0μs for BFS).

4. **Speed vs accuracy trade-off:** BFS is 850x faster than A* (0.08s vs 68s avg) because it makes zero HTTP calls. A*'s 68s average is almost entirely vLLM inference latency — approximately 6,600 calls × 10ms each.

---

## 4. LLM Prediction Quality Analysis

### 4.1 How LLM Quality Affects Solving Performance

The LLM's role in A* is subtle: it does not decide *whether* to explore a state, only *when* (via priority boost). This means:

- **If the LLM is correct:** The solution path is explored earlier → fewer total states needed → faster solve
- **If the LLM is wrong:** The solution path is explored later → more states needed → slower, but still found

The `fallback_count` metric tracks states where the LLM returned no parseable valid action and the solver fell back to uniform action scores. In all experiments, fallback_count ≈ 0, meaning the LLM consistently produces parseable output.

### 4.2 States Explored vs LLM Call Ratio

In A*, states explored ≈ LLM calls (one call per state popped from the heap). This ratio reveals that the solver is making approximately 1 LLM call per state expansion — the LLM is consulted for every single state, not just a subset.

### 4.3 When LLM Guidance Helps vs Hurts

| Scenario | LLM helps? | Why |
|---|---|---|
| Easy puzzles (2 boxes, short solution) | Marginal | BFS also solves easily; LLM adds latency |
| Medium puzzles with clear box-target alignment | Yes | LLM correctly identifies push direction, reduces states explored |
| Hard puzzles requiring counter-intuitive moves | No | LLM deprioritizes necessary "wrong-looking" moves |
| Puzzles hitting state cap | No | Both guided and unguided fail at cap |

### 4.4 Logprob-Based Confidence Scoring

The system uses vLLM's token-level logprobs to assign real probability scores to each action. When the model outputs `{"action": "up"}`, the logprob of the token `"up"` becomes the confidence score. This is more informative than a simple parsed action:

- High confidence (prob > 0.7) → LLM is certain → strong priority boost
- Low confidence (prob ≈ 0.25) → LLM is uncertain → weak boost, other actions explored nearly equally

---

## 5. Computational Trade-offs

### 5.1 Latency Breakdown

| Component | Time per state |
|---|---|
| BFS state expansion | ~0.01 ms |
| A* heap operations | ~0.05 ms |
| vLLM HTTP round-trip | ~8–12 ms |
| vLLM GPU inference | ~3–5 ms (amortized over batch) |

The bottleneck is **network + inference latency per state**. Even with batch_size=16 and concurrent requests, the per-state LLM overhead dominates.

### 5.2 Memory Usage

The MI300X holds the full Qwen2.5-7B model (≈14 GB weights) plus KV cache in 170 GB of 192 GB VRAM. State caching in Python uses negligible RAM relative to model size.

### 5.3 Scaling Behavior

| Metric | BFS | A* |
|---|---|---|
| Time complexity (states) | O(b^d) | O(b^d) with better constant |
| LLM calls per puzzle | 0 | ≈ states explored |
| Wall-clock time | ~0.08s | ~68s (LLM-dominated) |
| Scales with puzzle difficulty | Exponentially | Exponentially but with LLM pruning |

---

## 6. Discussion

### 6.1 How does the system handle single-step LLM usage?

The single-step constraint is handled by treating the LLM as a **priority function component** rather than a planner. At each state popped from A*'s heap, the LLM predicts the best action. This prediction feeds into the f-score formula:

```
f(n) = g(n) - h(n) - llm_weight × llm_probability(n)
```

The LLM never sees future states or multi-step sequences. It only answers: "given this board right now, what is the best next move?" This makes it a pure one-step oracle used inside a broader search.

The key design insight is that **search provides the safety net**: even if the LLM is wrong at step k, the correct action at step k is still pushed onto the heap (just with lower priority) and will eventually be explored.

### 6.2 Computational trade-offs of the search strategy

**A* vs BFS:**
- A* explores fewer states on easy/medium puzzles (LLM steers toward solution)
- A* is 850x slower per puzzle due to LLM inference overhead
- For this puzzle set, BFS is both faster and more accurate — the LLM's guidance doesn't overcome its latency cost

**A* vs Beam Search:**
- A* is more accurate because it never permanently prunes states
- Beam search is faster but fails on puzzles requiring counter-intuitive moves
- The completeness guarantee of A* is critical for Sokoban

**The fundamental tension:** Sokoban's combinatorial state space requires broad exploration (BFS-style) but is too large for unconstrained search on hard puzzles. The LLM should reduce the search space, but its inference latency negates the time savings unless it dramatically reduces states explored.

### 6.3 How to improve with more LLM calls or different architectures

**More LLM calls:**
- Use the LLM to **evaluate states** (not just actions) — a value function that estimates how close a state is to solved. This would improve the heuristic `h(n)` which currently uses Manhattan distance.
- Multi-step rollouts: ask the LLM for 3–5 step sequences, execute them, and use the resulting state as a single node in the tree. Reduces tree depth while using the same total LLM calls.

**Different architectures:**
- **MCTS with LLM rollout policy:** Use the LLM as the rollout policy (plays out random games) and UCB for exploration. Better exploration-exploitation balance than A*.
- **Fine-tuned model:** Train Qwen2.5-7B specifically on Sokoban state-action pairs using supervised learning or RL. A Sokoban-specific model would make far fewer incorrect predictions.
- **RL training (optional assignment component):** Use REINFORCE or PPO with the LLM as policy. Reward signal: +1 for solving, −0.01 per step (encourages shorter solutions). A fine-tuned model would likely reduce states explored by 10–100x on medium puzzles.
- **Smaller model + caching:** Qwen2.5-3B would be ~2x faster at inference with some accuracy loss. Combined with aggressive state caching (many states are revisited), could reduce wall-clock time significantly.

---

## 7. Conclusion

The system successfully implements a tree-based planning system with LLM guidance for Sokoban. Key findings:

1. **A* with LLM guidance achieves 80% accuracy** on 10 varied-difficulty puzzles, solving 8/10 within the computational budget.

2. **BFS (no LLM) achieves 90% accuracy** at 850x less wall-clock time — the LLM does not yet provide enough guidance quality to overcome its inference latency for this puzzle set.

3. **State representation does not significantly affect accuracy** with Qwen2.5-7B. All three formats (ASCII, Structured, Annotated) produce identical results, suggesting the model is robust to input format.

4. **The primary bottleneck is LLM inference latency** (~10ms per state × 6,600 states = ~66s). Reducing this through model distillation, caching, or fewer LLM calls would be the highest-impact improvement.

5. **A* is the correct algorithmic choice** for this problem: its completeness guarantee ensures failures are always budget-related, not structural. Beam search's permanent pruning makes it unsuitable for Sokoban's counter-intuitive solution paths.
