"""
debug_llm.py — Diagnostic tool to inspect LLM predictions step by step.

Run:
    python debug_llm.py                        # vLLM default (must run start_vllm.sh first)
    python debug_llm.py --backend ollama       # ollama local
    python debug_llm.py --backend groq         # groq cloud
    python debug_llm.py --n 5 --puzzles 0 1 2 # first 5 steps of puzzles 0, 1, 2

For each step it shows:
  - The board (REPR_ANNOTATED)
  - Raw LLM output
  - Parsed action vs. valid actions
  - Whether the predicted action is legal and whether it pushes a box
"""

import argparse
import sys
from pathlib import Path

from environment import from_parsed, apply_action, is_solved, get_valid_actions, render, ACTION_DELTAS
from llm_predictor import (
    LLMPredictor, call_llm, parse_action_from_output,
    VLLM_MODEL, OLLAMA_MODEL, GROQ_MODEL,
    VLLM_BASE_URL, VLLM_API_KEY,
)
from representation import (
    build_prompt, to_annotated,
    REPR_ASCII, REPR_STRUCTURED, REPR_ANNOTATED,
)
from parser import load_and_parse_all


# ---------------------------------------------------------------------------
SEPARATOR = "=" * 70


def debug_single_puzzle(puzzle_dict: dict,
                        predictor: LLMPredictor,
                        n_steps: int = 10,
                        puzzle_label: str = "?") -> None:
    """Walk through one puzzle for n_steps, printing LLM debug info at each step."""
    state = from_parsed(puzzle_dict)

    print(f"\n{SEPARATOR}")
    print(f"  PUZZLE {puzzle_label}  |  boxes={len(state.box_positions)}"
          f"  targets={len(state.target_positions)}")
    print(SEPARATOR)

    for step in range(n_steps):
        valid = get_valid_actions(state)
        if is_solved(state):
            print(f"  [Step {step}] SOLVED!")
            break
        if not valid:
            print(f"  [Step {step}] No valid actions — stuck.")
            break

        print(f"\n--- Step {step+1} ---")
        print("Board (REPR_ANNOTATED):")
        print(to_annotated(state))
        print(f"Valid actions: {valid}")

        # Build prompt and call LLM
        prompt = build_prompt(state, REPR_ANNOTATED)

        try:
            response = call_llm(prompt, model_name=predictor.model_name,
                                backend=predictor.backend)
        except Exception as e:
            print(f"  [LLM ERROR] {e}")
            break

        # Extract raw text from ChatCompletion object (vLLM) or plain string
        if hasattr(response, 'choices'):
            raw_text = response.choices[0].message.content or ""
            # Also show logprobs if available
            lp_data = None
            try:
                lp_data = response.choices[0].logprobs
            except AttributeError:
                pass
        else:
            raw_text = response
            lp_data = None

        parsed = parse_action_from_output(raw_text)

        print(f"\nRaw LLM output:\n  {raw_text[:300]!r}")

        if lp_data and lp_data.content:
            first_token_lp = lp_data.content[0]
            print("Token logprobs (first output token):")
            for top_lp in (first_token_lp.top_logprobs or [])[:6]:
                import math
                prob = math.exp(top_lp.logprob)
                print(f"  {top_lp.token!r:15s}  logprob={top_lp.logprob:7.3f}  prob={prob:.4f}")

        print(f"\nParsed action : {parsed!r}")
        is_valid_pred = parsed in valid
        print(f"Legal?        : {'YES' if is_valid_pred else 'NO — invalid move!'}")

        if is_valid_pred:
            # Check if it pushes a box
            pr, pc = state.player_pos
            dr, dc = ACTION_DELTAS[parsed]
            nr, nc = pr + dr, pc + dc
            pushes_box = (nr, nc) in state.box_positions
            print(f"Pushes box?   : {'YES' if pushes_box else 'no (player walk)'}")

            # Apply and continue
            state = apply_action(state, parsed)
        else:
            # Illegal prediction — pick the first valid action as fallback
            fallback = valid[0]
            print(f"  → Falling back to first valid action: {fallback!r}")
            state = apply_action(state, fallback)

    if is_solved(state):
        print(f"\n{SEPARATOR}")
        print("  PUZZLE SOLVED within the step limit!")
    else:
        print(f"\n{SEPARATOR}")
        print(f"  Not solved after {n_steps} steps.")


def test_batch_inference(puzzle_dicts: list[dict],
                         predictor: LLMPredictor,
                         n_puzzles: int = 3) -> None:
    """Test predict_batch_states: check timing and validity across multiple states."""
    import time

    states = [from_parsed(p) for p in puzzle_dicts[:n_puzzles]]
    print(f"\n{SEPARATOR}")
    print(f"  BATCH INFERENCE TEST — {n_puzzles} states simultaneously")
    print(SEPARATOR)

    t0 = time.time()
    batch_results = predictor.predict_batch_states(states, k=4)
    elapsed = time.time() - t0

    for i, (state, preds) in enumerate(zip(states, batch_results)):
        valid = get_valid_actions(state)
        print(f"\nState {i}: valid={valid}")
        for action, score in preds:
            mark = "✓" if action in valid else "✗ INVALID"
            print(f"  {action:6s}  score={score:.4f}  {mark}")

    print(f"\nTotal batch time : {elapsed:.3f}s  ({elapsed/n_puzzles:.3f}s per state)")
    print(f"LLM calls made   : {predictor.call_count}")
    print(f"Cache hits       : {predictor.cache_hits}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Debug LLM predictions on Sokoban")
    parser.add_argument("--backend", default="vllm",
                        choices=["vllm", "ollama", "groq", "mlx"],
                        help="LLM backend to use (default: vllm)")
    parser.add_argument("--model", default=None,
                        help="Override model name")
    parser.add_argument("--repr", default="annotated",
                        choices=["ascii", "structured", "annotated"],
                        help="State representation (default: annotated)")
    parser.add_argument("--puzzles", nargs="+", type=int, default=[0, 1, 2],
                        help="Which puzzle indices to debug (default: 0 1 2)")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of steps per puzzle (default: 8)")
    parser.add_argument("--batch-test", action="store_true",
                        help="Also run batch inference test")
    args = parser.parse_args()

    # Resolve model name
    model = args.model
    if model is None:
        model = {"vllm": VLLM_MODEL, "ollama": OLLAMA_MODEL,
                 "groq": GROQ_MODEL, "mlx": "mlx-community/Qwen2.5-3B-Instruct-4bit"
                 }[args.backend]

    repr_map = {"ascii": REPR_ASCII, "structured": REPR_STRUCTURED,
                "annotated": REPR_ANNOTATED}
    repr_type = repr_map[args.repr]

    print(f"Backend : {args.backend}")
    print(f"Model   : {model}")
    print(f"Repr    : {repr_type}")

    # Load puzzles
    data_path = Path(__file__).parent / "data" / "Microban.txt"
    if not data_path.exists():
        print(f"ERROR: Microban.txt not found at {data_path}")
        print("       Place the puzzle file at data/Microban.txt relative to workspace root.")
        sys.exit(1)

    all_puzzles = load_and_parse_all(str(data_path))
    print(f"Loaded  : {len(all_puzzles)} puzzles from {data_path}")

    predictor = LLMPredictor(model_name=model, backend=args.backend,
                             repr_type=repr_type)

    # Step-by-step debug
    for idx in args.puzzles:
        if idx >= len(all_puzzles):
            print(f"Puzzle {idx} out of range (max {len(all_puzzles)-1})")
            continue
        debug_single_puzzle(all_puzzles[idx], predictor,
                            n_steps=args.n, puzzle_label=str(idx))

    # Batch test
    if args.batch_test:
        predictor.reset_call_count()
        test_batch_inference(all_puzzles, predictor, n_puzzles=len(args.puzzles))


if __name__ == "__main__":
    main()
