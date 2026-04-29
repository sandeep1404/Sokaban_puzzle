"""
solve_puzzle.py — Step-by-step puzzle solver visualizer.

Usage:
    python solve_puzzle.py --puzzle 3 --solver bfs
    python solve_puzzle.py --puzzle 7 --solver beam
    python solve_puzzle.py --puzzle 7 --solver astar
    python solve_puzzle.py --puzzle 0 --solver mcts

Arguments:
    --puzzle  : 0-based puzzle index (e.g. 0 = first puzzle)
    --solver  : one of  bfs | beam | astar | mcts
"""

import argparse
import sys
from pathlib import Path

from environment import from_parsed, apply_action, is_solved, render
from search import bfs_solve, BeamSearchSolver, AStarSolver, MCTSSolver
from llm_predictor import LLMPredictor
from parser import load_and_parse_all

SEPARATOR = "=" * 60
STEP_SEP  = "-" * 60


def parse_args():
    parser = argparse.ArgumentParser(description="Solve a Sokoban puzzle step by step.")
    parser.add_argument("--puzzle", type=int, required=True,
                        help="0-based puzzle index")
    parser.add_argument("--solver", type=str, required=True,
                        choices=["bfs", "beam", "astar", "mcts"],
                        help="Solver to use: bfs | beam | astar | mcts")
    return parser.parse_args()


def print_state(state, step_num: int, action_taken: str | None = None):
    print(STEP_SEP)
    if step_num == 0:
        print(f"  INITIAL STATE")
    else:
        print(f"  Step {step_num:>3}  |  Action: {action_taken.upper()}")
    print(STEP_SEP)
    print(render(state))
    print(f"  Player: {state.player_pos}  |  Boxes: {sorted(state.box_positions)}")


def main():
    args = parse_args()

    # ── Load puzzles ──────────────────────────────────────────────
    data_path = Path(__file__).parent / "data" / "Microban.txt"
    puzzles = load_and_parse_all(str(data_path))

    if args.puzzle < 0 or args.puzzle >= len(puzzles):
        print(f"[ERROR] Puzzle index {args.puzzle} out of range "
              f"(0 – {len(puzzles) - 1})")
        sys.exit(1)

    puzzle_dict = puzzles[args.puzzle]
    initial_state = from_parsed(puzzle_dict)

    print(SEPARATOR)
    print(f"  Puzzle #{args.puzzle}  |  Solver: {args.solver.upper()}")
    print(f"  Boxes: {len(puzzle_dict['box_positions'])}  |  "
          f"Targets: {len(puzzle_dict['target_positions'])}")
    print(SEPARATOR)

    # ── Run solver to get the solution path ───────────────────────
    print(f"\nRunning {args.solver.upper()} solver…", flush=True)

    if args.solver == "bfs":
        result = bfs_solve(initial_state, max_states=500_000)
        llm_note = "no LLM calls"

    else:
        predictor = LLMPredictor()
        if args.solver == "beam":
            solver = BeamSearchSolver(predictor, beam_width=20, max_depth=300, max_llm_calls=20_000)
        elif args.solver == "mcts":
            solver = MCTSSolver(predictor, n_iterations=1000, exploration_c=1.4,
                                rollout_depth=30, max_llm_calls=1_000)
        else:  # astar
            solver = AStarSolver(predictor, max_states=50_000, max_llm_calls=50_000, batch_size=16)

        result = solver.solve(initial_state)
        llm_note = f"{predictor.call_count} LLM calls"

    # ── Display results ───────────────────────────────────────────
    if not result["solved"]:
        print(f"\n[FAILED] Solver could not find a solution. ({llm_note})")
        print(f"States explored: {result.get('states_explored', 'N/A')}")
        sys.exit(1)

    path = result["path"]
    print(f"[SOLVED] {len(path)} steps  |  {llm_note}")
    print(f"States explored: {result.get('states_explored', 'N/A')}\n")

    # ── Replay state by state ─────────────────────────────────────
    state = initial_state
    print_state(state, step_num=0, action_taken=None)

    for i, action in enumerate(path, start=1):
        state = apply_action(state, action)
        print_state(state, step_num=i, action_taken=action)

    # ── Final summary ─────────────────────────────────────────────
    print()
    print(SEPARATOR)
    if is_solved(state):
        print(f"  PUZZLE SOLVED in {len(path)} steps!")
    else:
        print("  WARNING: replayed path did not reach solved state.")
    print(f"  Full path: {' -> '.join(path)}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
