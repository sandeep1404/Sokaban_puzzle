"""
evaluation.py — Module 6
Runs your solver on multiple puzzles and collects structured metrics.
Also contains visualization functions for the analysis report.

Build order: CODE THIS SIXTH (after search.py works on puzzle 1)
Test: run `python evaluation.py` — evaluates on 10 puzzles and prints summary table
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import time
import json
from pathlib import Path
from typing import Any

from environment import SokobanState, from_parsed
from search import bfs_solve, BeamSearchSolver, AStarSolver, MCTSSolver
from llm_predictor import LLMPredictor
from representation import REPR_ASCII, REPR_STRUCTURED, REPR_ANNOTATED
from parser import load_and_parse_all

# For plots — install with: pip install matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from llm_predictor import LLMPredictor

# ---------------------------------------------------------------------------
# FUNCTION 1
# ---------------------------------------------------------------------------
def evaluate_single(puzzle_idx: int,
                    puzzle_dict: dict,
                    solver,
                    solver_name: str = "beam") -> dict:
    """
    WHAT IT DOES:
        Runs the solver on ONE puzzle and returns a structured result dict.

    HOW IT WORKS:
        1. Convert puzzle_dict → SokobanState via from_parsed()
        2. Reset predictor call count if solver has one (solver.predictor.reset_call_count())
        3. Record start time
        4. Call solver.solve(state) or bfs_solve(state)
        5. Record end time
        6. Build and return result dict

    ARGS:
        puzzle_idx  : int — which puzzle number (for labelling)
        puzzle_dict : dict from parse_puzzle()
        solver      : BeamSearchSolver | MCTSSolver | AStarSolver | None (None = use bfs_solve)
        solver_name : str — label for this solver config

    RETURNS:
        dict with keys:
            "puzzle_idx"      : int
            "solver"          : str
            "solved"          : bool
            "steps"           : int | None
            "llm_calls"       : int
            "time_seconds"    : float
            "n_boxes"         : int  (complexity indicator)
            "n_targets"       : int
            "solution_path"   : list[str] | None
    """

    state = from_parsed(puzzle_dict)

    # Reset LLM call counter
    if hasattr(solver, 'predictor'):
        solver.predictor.reset_call_count()

    start = time.time()

    ## bfs solver (no llm)

    if solver is None:
        raw= bfs_solve(state,max_states=50000)
        llm_calls = 0

    else :
        raw =solver.solve(state)
        llm_calls = solver.predictor.call_count if hasattr(solver, 'predictor') else 0

    elapsed = time.time() - start

    return {
    "puzzle_idx"   : puzzle_idx,
    "solver"       : solver_name,
    "solved"       : raw["solved"],
    "steps"        : raw.get("steps"),
    "llm_calls"    : llm_calls,
    "fallback_count": raw.get("fallback_count", 0),
    "time_seconds" : elapsed,
    "n_boxes"      : len(puzzle_dict["box_positions"]),
    "n_targets"    : len(puzzle_dict["target_positions"]),
    "solution_path": raw.get("path"),
    }



# ---------------------------------------------------------------------------
# FUNCTION 2
# ---------------------------------------------------------------------------
def evaluate_batch(puzzle_dicts: list[dict],
                   solver,
                   solver_name: str,
                   puzzle_indices: list[int] | None = None) -> list[dict]:
    """
    WHAT IT DOES:
        Calls evaluate_single() for each puzzle in the list.
        Prints progress to console as it runs (so you can watch it work).

    HOW IT WORKS:
        Simple loop. Print "Puzzle X/N: solved/failed" after each one.
        Catch exceptions per-puzzle so one failure doesn't stop the batch.

    ARGS:
        puzzle_dicts   : list of puzzle dicts
        solver         : solver instance or None for BFS
        solver_name    : label string
        puzzle_indices : which puzzle numbers to evaluate (default: first N)

    RETURNS:
        list of result dicts from evaluate_single()
    """
    # TODO: implement this (~20 lines)
    
    indices = puzzle_indices or list(range(len(puzzle_dicts)))
    results = []
    for i, idx in enumerate(indices):
        print(f"  [{i+1}/{len(indices)}] Puzzle {idx}...", end=" ", flush=True)
        try:
            r = evaluate_single(idx, puzzle_dicts[idx], solver, solver_name)
            status = "SOLVED" if r["solved"] else "failed"
            print(f"{status} ({r['steps']} steps, {r['llm_calls']} LLM calls, {r.get('fallback_count',0)} fallbacks)")
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"puzzle_idx": idx, "solver": solver_name,
                        "solved": False, "steps": None, "llm_calls": 0,
                        "time_seconds": 0, "n_boxes": 0, "n_targets": 0,
                        "solution_path": None})    
    return results       



# ---------------------------------------------------------------------------
# FUNCTION 3
# ---------------------------------------------------------------------------
def compute_summary(results: list[dict]) -> dict:
    """
    WHAT IT DOES:
        Aggregates a list of result dicts into summary statistics.

    COMPUTES:
        - success_rate       : float  (solved / total)
        - total_solved       : int
        - total_puzzles      : int
        - avg_steps          : float  (only over solved puzzles)
        - avg_llm_calls      : float
        - avg_time_seconds   : float
        - min_steps          : int | None
        - max_steps          : int | None

    RETURNS:
        dict with all the above keys
    """
    
    
    total = len(results)
    solved_results = [r for r in results if r["solved"]]
    n_solved = len(solved_results)

    return {
        "success_rate"    : n_solved / total if total > 0 else 0,
        "total_solved"    : n_solved,
        "total_puzzles"   : total,
        "avg_steps"       : sum(r["steps"] for r in solved_results) / n_solved if n_solved else None,
        "avg_llm_calls"   : sum(r["llm_calls"] for r in results) / total if total else 0,
        "avg_fallbacks"   : sum(r.get("fallback_count", 0) for r in results) / total if total else 0,
        "avg_time_seconds": sum(r["time_seconds"] for r in results) / total if total else 0,
        "min_steps"       : min((r["steps"] for r in solved_results), default=None),
        "max_steps"       : max((r["steps"] for r in solved_results), default=None),
    }


# ---------------------------------------------------------------------------
# FUNCTION 4
# ---------------------------------------------------------------------------
def print_summary_table(summary_by_solver: dict[str, dict]):
    """
    WHAT IT DOES:
        Prints a formatted comparison table to console.

    FORMAT:
        Solver             | Solved | Success% | Avg Steps | Avg LLM Calls | Avg Time
        -------------------|--------|----------|-----------|---------------|--------
        bfs                |  8/10  |  80.0%   |    23.1   |       0       |  0.12s
        beam_ascii_w5      |  7/10  |  70.0%   |    31.4   |     156       |  4.21s
        beam_structured_w5 |  6/10  |  60.0%   |    28.9   |     134       |  3.98s

    ARGS:
        summary_by_solver: dict mapping solver_name → summary dict
    """
    header = f"{'Solver':<25} | {'Solved':>8} | {'Success%':>9} | {'Avg Steps':>10} | {'Avg LLM Calls':>14} | {'Fallbacks':>10} | {'Avg Time':>9}"
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for solver_name, s in summary_by_solver.items():
        solved_str   = f"{s['total_solved']}/{s['total_puzzles']}"
        success_str  = f"{s['success_rate']*100:.1f}%"
        steps_str    = f"{s['avg_steps']:.1f}" if s['avg_steps'] is not None else "N/A"
        llm_str      = f"{s['avg_llm_calls']:.0f}"
        fallback_str = f"{s.get('avg_fallbacks', 0):.1f}"
        time_str     = f"{s['avg_time_seconds']:.2f}s"

        print(f"{solver_name:<25} | {solved_str:>8} | {success_str:>9} | {steps_str:>10} | {llm_str:>14} | {fallback_str:>10} | {time_str:>9}")

    print(separator)





# ---------------------------------------------------------------------------
# FUNCTION 5
# ---------------------------------------------------------------------------
def plot_success_by_solver(summary_by_solver: dict[str, dict],
                           output_path: str = "outputs/success_by_solver.png"):
    """
    WHAT IT DOES:
        Bar chart comparing success rates across solver configurations.

    CHART:
        X axis: solver names
        Y axis: success rate (0 to 1)
        One bar per solver, colored differently
        Title: "Sokoban Solver Success Rates"

    HOW TO SAVE:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)

    ARGS:
        summary_by_solver: dict mapping solver_name → summary dict
        output_path      : where to save the PNG
    """
    solver_names  = list(summary_by_solver.keys())
    success_rates = [summary_by_solver[s]["success_rate"] for s in solver_names]
    colors = plt.cm.tab10.colors[:len(solver_names)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(solver_names, success_rates, color=colors, edgecolor="black", width=0.5)

    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate*100:.1f}%",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Solver")
    ax.set_title("Sokoban Solver Success Rates")
    ax.set_xticks(range(len(solver_names)))
    ax.set_xticklabels(solver_names, rotation=20, ha="right", fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {output_path}")



# ---------------------------------------------------------------------------
# FUNCTION 6
# ---------------------------------------------------------------------------
def plot_complexity_vs_steps(all_results: list[dict],
                             output_path: str = "outputs/complexity_vs_steps.png"):
    """
    WHAT IT DOES:
        Scatter plot: puzzle complexity vs steps to solve.
        Only plots solved puzzles.

    AXES:
        X: complexity = n_boxes * n_targets  (proxy for difficulty)
        Y: steps to solve
        Color-code or marker-shape by solver name
        Title: "Puzzle Complexity vs Steps to Solve"

    ARGS:
        all_results: flat list of all result dicts (all solvers combined)
        output_path: where to save PNG
    """
    # Only keep solved puzzles
    solved = [r for r in all_results if r["solved"]]
    if not solved:
        print("No solved results to plot.")
        return

    # Assign a unique color to each solver name
    solver_names = list(dict.fromkeys(r["solver"] for r in solved))
    color_map = {name: plt.cm.tab10.colors[i % 10] for i, name in enumerate(solver_names)}

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in solver_names:
        group = [r for r in solved if r["solver"] == name]
        xs = [r["n_boxes"] * r["n_targets"] for r in group]   # complexity proxy
        ys = [r["steps"] for r in group]
        ax.scatter(xs, ys, label=name, color=color_map[name], alpha=0.75, edgecolors="black", s=60)

    ax.set_xlabel("Puzzle Complexity  (n_boxes × n_targets)")
    ax.set_ylabel("Steps to Solve")
    ax.set_title("Puzzle Complexity vs Steps to Solve")
    ax.legend(title="Solver")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# FUNCTION 7  — compare representation types
# ---------------------------------------------------------------------------
def run_representation_experiment(puzzle_dicts: list[dict],
                                  model_name: str,
                                  beam_width: int = 5,
                                  n_puzzles: int = 10) -> dict:
    """
    WHAT IT DOES:
        Runs BeamSearchSolver with all 3 representation types on the same
        n_puzzles and returns results for comparison.
        This answers the assignment question: "Compare different state
        representations for the LLM."

    HOW IT WORKS:
        For repr_type in [REPR_ASCII, REPR_STRUCTURED, REPR_ANNOTATED]:
            predictor = LLMPredictor(repr_type=repr_type)
            solver = BeamSearchSolver(predictor, beam_width=beam_width)
            results = evaluate_batch(puzzle_dicts[:n_puzzles], solver, repr_type)
            summary = compute_summary(results)
            store under repr_type key

    RETURNS:
        dict mapping repr_type → summary dict
    """
    indices = list(range(n_puzzles))
    results_by_repr = {}
    for repr_type in [REPR_ASCII, REPR_STRUCTURED, REPR_ANNOTATED]:
        predictor = LLMPredictor(model_name=model_name, backend="mlx",
                                repr_type=repr_type)
        solver = BeamSearchSolver(predictor, beam_width=beam_width,
                                  max_depth=200, max_llm_calls=200)
        results = evaluate_batch(puzzle_dicts, solver,
                                 solver_name=f"beam_{repr_type}",
                                 puzzle_indices=indices)
        results_by_repr[repr_type] = compute_summary(results)
    return results_by_repr

def run_beam_width_experiment(puzzle_dicts: list[dict],
                              model_name: str,
                              widths: list[int] = [1, 3, 5, 10],
                              n_puzzles: int = 10) -> dict:
    """
    WHAT IT DOES:
        Runs BeamSearchSolver with different beam_width values on the same
        puzzles. Answers: "Analyze the effect of different search strategies."

    RETURNS:
        dict mapping beam_width (int) → summary dict
    """
    indices = list(range(n_puzzles))
    results_by_width = {}
    predictor = LLMPredictor(model_name=model_name, backend="mlx")
    for w in widths:
        solver = BeamSearchSolver(predictor, beam_width=w,
                                  max_depth=200, max_llm_calls=500)
        results = evaluate_batch(puzzle_dicts, solver,
                                 solver_name=f"beam_w{w}",
                                 puzzle_indices=indices)
        results_by_width[w] = compute_summary(results)
    return results_by_width


# ---------------------------------------------------------------------------
# QUICK SELF-TEST  — run: python evaluation.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from llm_predictor import VLLM_MODEL, MLX_MODEL
    filepath = Path(__file__).parent / "data" / "Microban.txt"
    all_puzzles = load_and_parse_all(str(filepath))
    # 10 puzzles spanning varying difficulty across Microban's 150 puzzles:
    # indices chosen to cover easy (0-9), medium (20-60), hard (80-149)
    PUZZLES = [0, 1, 5, 10, 20, 30, 50, 70, 90, 110]

    # ------------------------------------------------------------------
    # 1. BFS — no LLM, fastest baseline
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Running BFS on puzzles", PUZZLES)
    print("=" * 50)
    bfs_results = evaluate_batch(all_puzzles, solver=None,
                                 solver_name="bfs", puzzle_indices=PUZZLES)
    bfs_summary = compute_summary(bfs_results)

    # Shared predictor — vLLM server on MI300x (start with: bash start_vllm.sh)
    # Uses REPR_ANNOTATED by default (annotated board + distance hints → better accuracy)
    # max_llm_calls=0 means UNLIMITED — LLM guides every single expansion.
    # Per assignment requirement: "LLM can only be used as a one-step predictor.
    # Given a board state, it should output a single action."
    # Every state in the search tree gets an LLM call — no heuristic fallback.
    predictor = LLMPredictor(model_name=VLLM_MODEL, backend="vllm")

    # ------------------------------------------------------------------
    # 2. A* — LLM guides every expansion (no fallback), batched vLLM
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Running Batched A* (vLLM, every state LLM-guided) on puzzles", PUZZLES)
    print("LLM model:", VLLM_MODEL, "| batch_size=64")
    print("=" * 50)
    predictor.reset_call_count()
    astar_solver = AStarSolver(predictor, max_states=10_000,
                               llm_weight=0.3,
                               max_llm_calls=10_000,  # fail after 10k LLM calls/puzzle
                               batch_size=64)
    astar_results = evaluate_batch(all_puzzles, solver=astar_solver,
                                   solver_name="astar_vllm", puzzle_indices=PUZZLES)
    astar_summary = compute_summary(astar_results)

    # ------------------------------------------------------------------
    # 3. MCTS — LLM-guided Monte Carlo Tree Search
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Running MCTS (vLLM, LLM-guided) on puzzles", PUZZLES)
    print("LLM model:", VLLM_MODEL)
    print("=" * 50)
    predictor.reset_call_count()
    mcts_solver = MCTSSolver(predictor, n_iterations=500,
                             exploration_c=1.4,
                             rollout_depth=30,
                             max_llm_calls=500)
    mcts_results = evaluate_batch(all_puzzles, solver=mcts_solver,
                                  solver_name="mcts_vllm", puzzle_indices=PUZZLES)
    mcts_summary = compute_summary(mcts_results)

    # ------------------------------------------------------------------
    # 4. Print comparison table
    # ------------------------------------------------------------------
    print()
    print("NOTE: bfs = pure BFS, zero LLM calls (baseline)")
    print("      astar_vllm = A* where EVERY state is LLM-guided (Qwen2.5-7B via vLLM)")
    print("      mcts_vllm  = MCTS with LLM priors + heuristic rollouts")
    print("      llm_calls/puzzle shows LLM involvement in solving")
    print()
    print_summary_table({
        "bfs (no LLM)"  : bfs_summary,
        "astar_vllm"    : astar_summary,
        "mcts_vllm"     : mcts_summary,
    })

    # ------------------------------------------------------------------
    # 5. Save plots
    # ------------------------------------------------------------------
    Path("outputs").mkdir(exist_ok=True)
    plot_success_by_solver({
        "bfs (no LLM)": bfs_summary,
        "astar_vllm"  : astar_summary,
        "mcts_vllm"   : mcts_summary,
    })
    all_results = bfs_results + astar_results + mcts_results
    plot_complexity_vs_steps(all_results)
    print("Plots saved to outputs/")
