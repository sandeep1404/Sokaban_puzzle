"""
compare_representations.py — Compare ASCII vs Structured vs Annotated
                               state representations using A* + vLLM.

Run:
    python3 compare_representations.py
    python3 compare_representations.py --puzzles 0 1 5 10 20 30 50 70 90 110
    python3 compare_representations.py --max-states 5000 --max-llm-calls 5000

For each representation it runs A* on the same 10 puzzles and prints:
  - Per-puzzle result table (solved / steps / LLM calls / time)
  - Final comparison summary table
"""

import argparse
import time
from pathlib import Path

from environment import from_parsed
from search import AStarSolver
from llm_predictor import LLMPredictor, VLLM_MODEL
from representation import REPR_ASCII, REPR_STRUCTURED, REPR_ANNOTATED
from parser import load_and_parse_all

SEPARATOR = "=" * 70
REPR_LABELS = {
    REPR_ASCII:       "ASCII       (raw grid)",
    REPR_STRUCTURED:  "Structured  (text description)",
    REPR_ANNOTATED:   "Annotated   (grid + distance hints)",
}

# Default puzzle indices — varied difficulty across Microban's 155 puzzles
DEFAULT_PUZZLES = [0, 1, 5, 10, 20, 30, 50, 70, 90, 110]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare state representations for A* + vLLM on Sokoban.")
    parser.add_argument("--puzzles", type=int, nargs="+",
                        default=DEFAULT_PUZZLES,
                        help="Puzzle indices to test (0-based, space-separated)")
    parser.add_argument("--max-states", type=int, default=10_000,
                        help="Max states A* can explore per puzzle (default: 10000)")
    parser.add_argument("--max-llm-calls", type=int, default=10_000,
                        help="Max LLM calls A* can make per puzzle (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="A* batch size for LLM calls (default: 16)")
    return parser.parse_args()


def run_single(puzzle_dict: dict, solver: AStarSolver, puzzle_idx: int) -> dict:
    """Run solver on one puzzle, return result dict."""
    state = from_parsed(puzzle_dict)
    solver.predictor.reset_call_count()
    t0 = time.time()
    raw = solver.solve(state)
    elapsed = time.time() - t0
    return {
        "puzzle_idx":  puzzle_idx,
        "solved":      raw["solved"],
        "steps":       raw.get("steps"),
        "llm_calls":   solver.predictor.call_count,
        "fallbacks":   raw.get("fallback_count", 0),
        "states":      raw.get("states_explored", 0),
        "time":        elapsed,
        "n_boxes":     len(puzzle_dict["box_positions"]),
    }


def print_per_puzzle_table(results: list[dict], repr_label: str):
    header = (f"{'Puzzle':>7} | {'Boxes':>5} | {'Solved':>6} | "
              f"{'Steps':>6} | {'States':>7} | {'LLM Calls':>10} | "
              f"{'Fallbacks':>9} | {'Time':>7}")
    sep = "-" * len(header)
    print(f"\n  Representation: {repr_label}")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        solved_str = "YES" if r["solved"] else "NO"
        steps_str  = str(r["steps"]) if r["steps"] is not None else "—"
        print(f"  {r['puzzle_idx']:>5} | {r['n_boxes']:>5} | {solved_str:>6} | "
              f"{steps_str:>6} | {r['states']:>7} | {r['llm_calls']:>10} | "
              f"{r['fallbacks']:>9} | {r['time']:>6.2f}s")
    print(sep)


def compute_summary(results: list[dict]) -> dict:
    total   = len(results)
    solved  = [r for r in results if r["solved"]]
    n       = len(solved)
    return {
        "solved":      n,
        "total":       total,
        "rate":        n / total if total else 0,
        "avg_steps":   sum(r["steps"] for r in solved) / n if n else None,
        "avg_states":  sum(r["states"] for r in results) / total if total else 0,
        "avg_llm":     sum(r["llm_calls"] for r in results) / total if total else 0,
        "avg_fb":      sum(r["fallbacks"] for r in results) / total if total else 0,
        "avg_time":    sum(r["time"] for r in results) / total if total else 0,
    }


def print_comparison_table(summaries: dict[str, dict]):
    header = (f"{'Representation':<35} | {'Solved':>8} | {'Accuracy':>9} | "
              f"{'Avg Steps':>10} | {'Avg States':>11} | {'Avg LLM':>9} | "
              f"{'Avg Fallbacks':>14} | {'Avg Time':>9}")
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  COMPARISON SUMMARY")
    print(sep)
    print(header)
    print(sep)
    for label, s in summaries.items():
        solved_str = f"{s['solved']}/{s['total']}"
        rate_str   = f"{s['rate']*100:.1f}%"
        steps_str  = f"{s['avg_steps']:.1f}" if s['avg_steps'] is not None else "N/A"
        print(f"  {label:<33} | {solved_str:>8} | {rate_str:>9} | "
              f"{steps_str:>10} | {s['avg_states']:>11.0f} | {s['avg_llm']:>9.0f} | "
              f"{s['avg_fb']:>14.1f} | {s['avg_time']:>8.2f}s")
    print(sep)


def main():
    args = parse_args()

    data_path = Path(__file__).parent / "data" / "Microban.txt"
    all_puzzles = load_and_parse_all(str(data_path))

    # Validate puzzle indices
    invalid = [i for i in args.puzzles if i < 0 or i >= len(all_puzzles)]
    if invalid:
        print(f"[ERROR] Invalid puzzle indices: {invalid} "
              f"(valid range: 0–{len(all_puzzles)-1})")
        return

    puzzle_dicts = [all_puzzles[i] for i in args.puzzles]

    print(SEPARATOR)
    print("  REPRESENTATION COMPARISON — A* with vLLM (Qwen2.5-7B)")
    print(f"  Puzzles: {args.puzzles}")
    print(f"  max_states={args.max_states}  max_llm_calls={args.max_llm_calls}  "
          f"batch_size={args.batch_size}")
    print(SEPARATOR)

    all_summaries: dict[str, dict] = {}

    for repr_type, repr_label in REPR_LABELS.items():
        print(f"\n{SEPARATOR}")
        print(f"  Running: {repr_label}")
        print(SEPARATOR)

        predictor = LLMPredictor(model_name=VLLM_MODEL,
                                 backend="vllm",
                                 repr_type=repr_type)
        solver = AStarSolver(predictor,
                             max_states=args.max_states,
                             llm_weight=0.3,
                             max_llm_calls=args.max_llm_calls,
                             batch_size=args.batch_size)

        results = []
        for i, (puzzle_idx, puzzle_dict) in enumerate(zip(args.puzzles, puzzle_dicts)):
            print(f"  [{i+1}/{len(args.puzzles)}] Puzzle {puzzle_idx} "
                  f"({len(puzzle_dict['box_positions'])} boxes)...",
                  end=" ", flush=True)
            r = run_single(puzzle_dict, solver, puzzle_idx)
            status = f"SOLVED in {r['steps']} steps" if r["solved"] else "FAILED"
            print(f"{status}  |  {r['llm_calls']} LLM calls  |  {r['time']:.2f}s")
            results.append(r)

        print_per_puzzle_table(results, repr_label)
        all_summaries[repr_label] = compute_summary(results)

    print_comparison_table(all_summaries)


if __name__ == "__main__":
    main()
