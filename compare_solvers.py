"""
compare_solvers.py — Compare BFS vs Beam Search vs A* on the same puzzles.

Answers the assignment requirement:
    "Analyze the effect of different search strategies"

Run:
    python3 compare_solvers.py
    python3 compare_solvers.py --puzzles 0 1 5 10 20 30 50 70 90 110
    python3 compare_solvers.py --max-states 5000 --max-llm-calls 5000

Output:
  - Per-solver per-puzzle tables
  - Final side-by-side comparison table
  - Bar chart saved to outputs/solver_comparison.png
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt

from environment import from_parsed
from search import bfs_solve, BeamSearchSolver, AStarSolver
from llm_predictor import LLMPredictor, VLLM_MODEL
from representation import REPR_ANNOTATED
from parser import load_and_parse_all

SEPARATOR = "=" * 75
DEFAULT_PUZZLES = [0, 1, 5, 10, 20, 30, 50, 70, 90, 110]


def parse_args():
    p = argparse.ArgumentParser(description="Compare BFS vs Beam vs A* on Sokoban.")
    p.add_argument("--puzzles", type=int, nargs="+", default=DEFAULT_PUZZLES)
    p.add_argument("--max-states",    type=int, default=10_000)
    p.add_argument("--max-llm-calls", type=int, default=10_000)
    p.add_argument("--beam-width",    type=int, default=20)
    p.add_argument("--batch-size",    type=int, default=16)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-puzzle runners
# ---------------------------------------------------------------------------

def run_bfs(puzzle_dict: dict, puzzle_idx: int, max_states: int) -> dict:
    state = from_parsed(puzzle_dict)
    t0 = time.time()
    raw = bfs_solve(state, max_states=max_states)
    return {
        "puzzle_idx": puzzle_idx,
        "solved":     raw["solved"],
        "steps":      raw.get("steps"),
        "llm_calls":  0,
        "states":     raw.get("states_explored", 0),
        "fallbacks":  0,
        "time":       time.time() - t0,
        "n_boxes":    len(puzzle_dict["box_positions"]),
    }


def run_beam(puzzle_dict: dict, puzzle_idx: int,
             predictor: LLMPredictor, beam_width: int,
             max_llm_calls: int) -> dict:
    state = from_parsed(puzzle_dict)
    predictor.reset_call_count()
    solver = BeamSearchSolver(predictor,
                              beam_width=beam_width,
                              max_depth=200,
                              max_llm_calls=max_llm_calls)
    t0 = time.time()
    raw = solver.solve(state)
    return {
        "puzzle_idx": puzzle_idx,
        "solved":     raw["solved"],
        "steps":      raw.get("steps"),
        "llm_calls":  predictor.call_count,
        "states":     raw.get("states_explored", 0),
        "fallbacks":  raw.get("fallback_count", 0),
        "time":       time.time() - t0,
        "n_boxes":    len(puzzle_dict["box_positions"]),
    }


def run_astar(puzzle_dict: dict, puzzle_idx: int,
              predictor: LLMPredictor, max_states: int,
              max_llm_calls: int, batch_size: int) -> dict:
    state = from_parsed(puzzle_dict)
    predictor.reset_call_count()
    solver = AStarSolver(predictor,
                         max_states=max_states,
                         llm_weight=0.3,
                         max_llm_calls=max_llm_calls,
                         batch_size=batch_size)
    t0 = time.time()
    raw = solver.solve(state)
    return {
        "puzzle_idx": puzzle_idx,
        "solved":     raw["solved"],
        "steps":      raw.get("steps"),
        "llm_calls":  predictor.call_count,
        "states":     raw.get("states_explored", 0),
        "fallbacks":  raw.get("fallback_count", 0),
        "time":       time.time() - t0,
        "n_boxes":    len(puzzle_dict["box_positions"]),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_solver_table(results: list[dict], solver_label: str):
    header = (f"{'Puzzle':>7} | {'Boxes':>5} | {'Solved':>6} | "
              f"{'Steps':>6} | {'States':>7} | {'LLM Calls':>10} | "
              f"{'Fallbacks':>9} | {'Time':>7}")
    sep = "-" * len(header)
    print(f"\n  Solver: {solver_label}")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        steps_str = str(r["steps"]) if r["steps"] is not None else "—"
        print(f"  {r['puzzle_idx']:>5} | {r['n_boxes']:>5} | "
              f"{'YES' if r['solved'] else 'NO':>6} | {steps_str:>6} | "
              f"{r['states']:>7} | {r['llm_calls']:>10} | "
              f"{r['fallbacks']:>9} | {r['time']:>6.2f}s")
    print(sep)


def compute_summary(results: list[dict]) -> dict:
    total  = len(results)
    solved = [r for r in results if r["solved"]]
    n      = len(solved)
    return {
        "solved":     n,
        "total":      total,
        "rate":       n / total if total else 0,
        "avg_steps":  sum(r["steps"] for r in solved) / n if n else None,
        "avg_states": sum(r["states"]    for r in results) / total if total else 0,
        "avg_llm":    sum(r["llm_calls"] for r in results) / total if total else 0,
        "avg_fb":     sum(r["fallbacks"] for r in results) / total if total else 0,
        "avg_time":   sum(r["time"]      for r in results) / total if total else 0,
    }


def print_comparison_table(summaries: dict[str, dict]):
    header = (f"{'Solver':<22} | {'Solved':>8} | {'Accuracy':>9} | "
              f"{'Avg Steps':>10} | {'Avg States':>11} | {'Avg LLM':>9} | "
              f"{'Avg Fallbacks':>14} | {'Avg Time':>9}")
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  SOLVER COMPARISON SUMMARY")
    print(sep)
    print(header)
    print(sep)
    for label, s in summaries.items():
        steps_str = f"{s['avg_steps']:.1f}" if s["avg_steps"] is not None else "N/A"
        print(f"  {label:<20} | {s['solved']:>3}/{s['total']:<3} | "
              f"{s['rate']*100:>8.1f}% | {steps_str:>10} | "
              f"{s['avg_states']:>11.0f} | {s['avg_llm']:>9.0f} | "
              f"{s['avg_fb']:>14.1f} | {s['avg_time']:>8.2f}s")
    print(sep)


def save_comparison_chart(summaries: dict[str, dict],
                          output_path: str = "outputs/solver_comparison.png"):
    labels = list(summaries.keys())
    rates  = [summaries[l]["rate"] * 100 for l in labels]
    times  = [summaries[l]["avg_time"] for l in labels]
    states = [summaries[l]["avg_states"] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = plt.cm.tab10.colors[:len(labels)]

    # Accuracy
    bars = axes[0].bar(labels, rates, color=colors, edgecolor="black")
    for bar, v in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1, f"{v:.0f}%",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy by Solver")
    axes[0].tick_params(axis="x", rotation=15)

    # Avg time
    bars = axes[1].bar(labels, times, color=colors, edgecolor="black")
    for bar, v in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2, f"{v:.1f}s",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Avg Time (s)")
    axes[1].set_title("Avg Solve Time by Solver")
    axes[1].tick_params(axis="x", rotation=15)

    # Avg states explored
    bars = axes[2].bar(labels, states, color=colors, edgecolor="black")
    for bar, v in zip(bars, states):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=9)
    axes[2].set_ylabel("Avg States Explored")
    axes[2].set_title("Search Efficiency by Solver")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nChart saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_path = Path(__file__).parent / "data" / "Microban.txt"
    all_puzzles = load_and_parse_all(str(data_path))

    invalid = [i for i in args.puzzles if i < 0 or i >= len(all_puzzles)]
    if invalid:
        print(f"[ERROR] Invalid puzzle indices: {invalid} "
              f"(valid range 0–{len(all_puzzles)-1})")
        return

    puzzle_dicts = [all_puzzles[i] for i in args.puzzles]

    print(SEPARATOR)
    print("  SOLVER COMPARISON — BFS vs Beam Search vs A*")
    print(f"  Puzzles     : {args.puzzles}")
    print(f"  max_states  : {args.max_states}")
    print(f"  max_llm_calls: {args.max_llm_calls}")
    print(f"  beam_width  : {args.beam_width}")
    print(f"  Model       : {VLLM_MODEL} (vLLM)")
    print(SEPARATOR)

    all_summaries: dict[str, dict] = {}

    # ── 1. BFS (no LLM) ──────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  [1/3] Running BFS (no LLM)...")
    print(f"{'─'*40}")
    bfs_results = []
    for i, (pidx, pd) in enumerate(zip(args.puzzles, puzzle_dicts)):
        print(f"  [{i+1}/{len(args.puzzles)}] Puzzle {pidx}...", end=" ", flush=True)
        r = run_bfs(pd, pidx, max_states=args.max_states)
        status = f"SOLVED in {r['steps']} steps" if r["solved"] else "FAILED"
        print(f"{status}  |  {r['states']} states  |  {r['time']:.2f}s")
        bfs_results.append(r)
    print_solver_table(bfs_results, "BFS (no LLM)")
    all_summaries["BFS (no LLM)"] = compute_summary(bfs_results)

    # ── 2. Beam Search ────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  [2/3] Running Beam Search (width={args.beam_width}, vLLM)...")
    print(f"{'─'*40}")
    beam_predictor = LLMPredictor(model_name=VLLM_MODEL,
                                  backend="vllm",
                                  repr_type=REPR_ANNOTATED)
    beam_results = []
    for i, (pidx, pd) in enumerate(zip(args.puzzles, puzzle_dicts)):
        print(f"  [{i+1}/{len(args.puzzles)}] Puzzle {pidx}...", end=" ", flush=True)
        r = run_beam(pd, pidx, beam_predictor,
                     beam_width=args.beam_width,
                     max_llm_calls=args.max_llm_calls)
        status = f"SOLVED in {r['steps']} steps" if r["solved"] else "FAILED"
        print(f"{status}  |  {r['llm_calls']} LLM calls  |  {r['time']:.2f}s")
        beam_results.append(r)
    print_solver_table(beam_results, f"Beam Search (width={args.beam_width})")
    all_summaries[f"Beam (w={args.beam_width})"] = compute_summary(beam_results)

    # ── 3. A* ─────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  [3/3] Running A* (vLLM-guided, batched)...")
    print(f"{'─'*40}")
    astar_predictor = LLMPredictor(model_name=VLLM_MODEL,
                                   backend="vllm",
                                   repr_type=REPR_ANNOTATED)
    astar_results = []
    for i, (pidx, pd) in enumerate(zip(args.puzzles, puzzle_dicts)):
        print(f"  [{i+1}/{len(args.puzzles)}] Puzzle {pidx}...", end=" ", flush=True)
        r = run_astar(pd, pidx, astar_predictor,
                      max_states=args.max_states,
                      max_llm_calls=args.max_llm_calls,
                      batch_size=args.batch_size)
        status = f"SOLVED in {r['steps']} steps" if r["solved"] else "FAILED"
        print(f"{status}  |  {r['llm_calls']} LLM calls  |  {r['time']:.2f}s")
        astar_results.append(r)
    print_solver_table(astar_results, "A* (LLM-guided)")
    all_summaries["A* (LLM-guided)"] = compute_summary(astar_results)

    # ── Final summary ─────────────────────────────────────────────
    print_comparison_table(all_summaries)
    save_comparison_chart(all_summaries)


if __name__ == "__main__":
    main()
