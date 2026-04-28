"""
representation.py — Module 3
Converts a SokobanState into different text formats for the LLM.
This is your EXPERIMENTAL VARIABLE — the assignment asks you to compare
which representation helps the LLM give better action predictions.

Build order: CODE THIS THIRD (after environment.py works)
Test: run `python representation.py` — prints all 3 formats for puzzle 1
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from environment import SokobanState, from_parsed, render, get_valid_actions
from parser import load_and_parse_all
from pathlib import Path


# ---------------------------------------------------------------------------
# REPRESENTATION TYPE CONSTANTS
# ---------------------------------------------------------------------------
REPR_ASCII      = "ascii"       # raw ASCII grid
REPR_STRUCTURED = "structured"  # text description of positions
REPR_ANNOTATED  = "annotated"   # ASCII grid + added distance hints


# ---------------------------------------------------------------------------
# FUNCTION 1
# ---------------------------------------------------------------------------
def to_ascii(state: SokobanState) -> str:
    """
    WHAT IT DOES:
        Returns the raw ASCII board — simplest representation.
        Just calls render() from environment.py.

    RETURNS:
        Multi-line string like:
            ####
            # .#
            #  ###
            #*@  #
            #  $ #
            #  ###
            ####

    WHY USEFUL:
        LLMs have seen Sokoban-like ASCII grids in training data.
        Direct visual representation — no information added or removed.
    """
    # TODO: 1 line — just call render()
    
    return render(state)


# ---------------------------------------------------------------------------
# FUNCTION 2
# ---------------------------------------------------------------------------
def to_structured(state: SokobanState) -> str:
    """
    WHAT IT DOES:
        Returns a text description of the board positions — no grid visual.

    FORMAT (return exactly this layout):
        Board size: {height} rows x {width} cols
        Player position: row {r}, col {c}
        Box positions: [(r1,c1), (r2,c2), ...]
        Target positions: [(r1,c1), (r2,c2), ...]
        Boxes on targets: {n} of {total}
        Unplaced boxes: [(r,c), ...]
        Unmatched targets: [(r,c), ...]

    WHY USEFUL:
        Removes visual noise. LLM gets exact numbers.
        Easier to reason about distances mathematically.
    """
    # TODO: implement this (~15 lines of string building)
    
    # board_size = (state.height)*(state.width)

    # player_pos_row, player_pos_col = state.player_pos
    # box_positions=[]
    # for box in state.box_positions:
    #     box_positions.append(box)

    # target_positions =[]
    # for target in state.target_positions:
    #     target_positions.append(target)
    # common_on_target = state.box_positions.intersection(state.target_positions)
    # box_on_target = len(common_on_target)/len(target_positions)

    # unplaced_boxes_set = state.box_positions - common_on_target

    # unplace_boxes =[]
    # for unplaced in unplaced_boxes_set:
    #     unplace_boxes.append(unplaced)

    lines = [
    f"Board size: {state.height} rows x {state.width} cols",
    f"Player position: row {state.player_pos[0]}, col {state.player_pos[1]}",
    f"Box positions: {sorted(state.box_positions)}",
    f"Target positions: {sorted(state.target_positions)}",
    f"Boxes on targets: {len(state.box_positions & state.target_positions)} of {len(state.target_positions)}",
    f"Unplaced boxes: {sorted(state.box_positions - state.target_positions)}",
    f"Unmatched targets: {sorted(state.target_positions - state.box_positions)}",
]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# FUNCTION 3
# ---------------------------------------------------------------------------
def to_annotated(state: SokobanState) -> str:
    """
    WHAT IT DOES:
        Returns the ASCII grid (like to_ascii) PLUS a text section below
        it with distance hints.

    FORMAT:
        [ASCII grid from to_ascii()]

        --- Hints ---
        Player is at (r, c).
        Box at (r1,c1) → nearest target at (r2,c2), distance {d} steps.
        Box at (r3,c3) → nearest target at (r4,c4), distance {d} steps.
        {n} boxes already on targets.

    HOW TO COMPUTE DISTANCE:
        Use Manhattan distance: abs(r1-r2) + abs(c1-c2)
        For each unplaced box, find the nearest unmatched target.

    WHY USEFUL:
        Combines spatial visual with quantitative hints.
        Helps the LLM reason about which direction to push first.
    """
    # TODO: implement this (~25 lines)
    ascii_part = to_ascii(state)

    hints = ["--- Hints ---"]
    hints.append(f"Player is at {state.player_pos}.")

    unplaced = state.box_positions - state.target_positions
    unmatched = state.target_positions - state.box_positions

    for box in sorted(unplaced):
        if unmatched:
            nearest = min(unmatched, key=lambda t: abs(box[0]-t[0]) + abs(box[1]-t[1]))
            dist = abs(box[0]-nearest[0]) + abs(box[1]-nearest[1])
            hints.append(f"Box at {box} → nearest target at {nearest}, distance {dist} steps.")

    placed = len(state.box_positions & state.target_positions)
    hints.append(f"{placed} of {len(state.target_positions)} boxes already on targets.")

    return ascii_part + '\n\n' + '\n'.join(hints)


# ---------------------------------------------------------------------------
# FUNCTION 4  — the main entry point called by llm_predictor.py
# ---------------------------------------------------------------------------
def build_prompt(state: SokobanState, repr_type: str = REPR_ASCII) -> str:
    """
    WHAT IT DOES:
        Wraps the chosen state representation in a full instruction prompt
        ready to be sent to the LLM.

    HOW IT WORKS:
        1. Call the matching to_*() function based on repr_type.
        2. Insert it into the prompt template below.

    PROMPT TEMPLATE:
        -----------------------------------------------
        You are solving a Sokoban puzzle.

        Rules:
        - Push boxes ($) onto all target locations (.)
        - You can push a box by walking into it (box moves one step ahead)
        - You cannot pull boxes
        - You cannot push a box into a wall (#) or another box
        - Symbols: # wall, @ player, $ box, . target, * box on target

        Current board state:
        {state_representation}

        Valid actions: up, down, left, right

        What is the single best next action to make progress toward solving
        the puzzle? Reply with exactly one word: up, down, left, or right.
        -----------------------------------------------

    ARGS:
        state     : SokobanState
        repr_type : one of REPR_ASCII, REPR_STRUCTURED, REPR_ANNOTATED

    RETURNS:
        str — the full prompt string
    """
    # TODO: implement this (~15 lines)
    
    if repr_type == REPR_ASCII:
        board_str = to_ascii(state)
    elif repr_type == REPR_STRUCTURED:
        board_str = to_structured(state)
    elif repr_type == REPR_ANNOTATED:
        board_str = to_annotated(state)
    else:
        raise ValueError(f"Unknown repr_type: {repr_type}")

    if repr_type == REPR_ASCII:
        # For ASCII, also append explicit position info to help the model
        pos_info = (
            f"Player position: row {state.player_pos[0]}, col {state.player_pos[1]}\n"
            f"Box positions: {sorted(state.box_positions)}\n"
            f"Target positions: {sorted(state.target_positions)}\n"
            f"Unplaced boxes: {sorted(state.box_positions - state.target_positions)}\n"
            f"Unmatched targets: {sorted(state.target_positions - state.box_positions)}"
        )
    else:
        pos_info = ""

    indented = "\n    ".join(board_str.splitlines())
    valid_actions_list = get_valid_actions(state)
    valid_actions = ", ".join(valid_actions_list)
    n_boxes = len(state.box_positions)
    n_placed = len(state.box_positions & state.target_positions)

    return f"""You are a skilled Sokoban solver. Your task is to choose the single best NEXT action.

Sokoban rules:
1. Walls (#) are impassable — the player and boxes cannot enter wall cells.
2. Player (@) moves up/down/left/right into empty space ( ) or a target (.).
3. Pushing a box ($): if the player moves into a box, the box shifts one step in the same direction.
   The cell behind the box must be empty or a target — cannot push into a wall or another box.
4. Goal: place ALL boxes onto target (.) locations. {n_placed} of {n_boxes} boxes placed so far.
5. Symbols: # wall | @ player | $ box | . target | * box on target | + player on target

Example:
Board:
    #####
    #.  #
    # $ #
    # @ #
    #####
Player: row 3, col 2. Box: (2,2). Target: (1,2). Valid moves: up, left, right.
{{"action": "up"}}

Now solve this:
Current board state:
    {indented}

{pos_info}

LEGAL ACTIONS (only these are physically possible right now): {valid_actions}

What is the single best next action?
Respond in JSON only, example: {{"action": "up"}}"""


# ---------------------------------------------------------------------------
# FUNCTION 5  — extended prompt for chain-of-thought (optional experiment)
# ---------------------------------------------------------------------------
def build_prompt_with_reasoning(state: SokobanState, repr_type: str = REPR_ASCII) -> str:
    """
    WHAT IT DOES:
        Like build_prompt() but asks the LLM to reason step by step
        before giving its final answer.

    PROMPT ADDITION:
        Instead of "Reply with exactly one word", ask:
            "Think step by step about what needs to happen, then on the
             LAST line write exactly: ACTION: <up|down|left|right>"

    WHY:
        Chain-of-thought sometimes improves LLM action quality.
        You can compare this vs. direct single-word prompting.

    RETURNS:
        str — full prompt string
    """
    # TODO: implement this (~similar to build_prompt but different ending)
    
    if repr_type == REPR_ASCII:
        board_str = to_ascii(state)
    elif repr_type == REPR_STRUCTURED:
        board_str = to_structured(state)
    elif repr_type == REPR_ANNOTATED:
        board_str = to_annotated(state)
    else:
        raise ValueError(f"Unknown repr_type: {repr_type}")

    if repr_type == REPR_ASCII:
        pos_info = (
            f"Player position: row {state.player_pos[0]}, col {state.player_pos[1]}\n"
            f"Box positions: {sorted(state.box_positions)}\n"
            f"Target positions: {sorted(state.target_positions)}\n"
            f"Unplaced boxes: {sorted(state.box_positions - state.target_positions)}\n"
            f"Unmatched targets: {sorted(state.target_positions - state.box_positions)}"
        )
    else:
        pos_info = ""
    
    indented = "\n    ".join(board_str.splitlines())
    valid_actions = ", ".join(get_valid_actions(state))
    n_boxes = len(state.box_positions)
    n_placed = len(state.box_positions & state.target_positions)

    return f"""You are a skilled Sokoban solver. Your task is to choose the single best NEXT action.

Sokoban rules:
1. Walls (#) are impassable — the player and boxes cannot enter wall cells.
2. Player (@) moves up/down/left/right into empty space ( ) or a target (.).
3. Pushing a box ($): if the player moves into a box, the box shifts one step in the same direction.
   The cell behind the box must be empty or a target — cannot push into a wall or another box.
4. Goal: place ALL {n_boxes} boxes onto target (.) locations. {n_placed} of {n_boxes} boxes placed so far.
5. Symbols: # wall | @ player | $ box | . target | * box on target | + player on target

Current board state:
    {indented}

{pos_info}

Currently valid moves: {valid_actions}

Think step by step:
- Where is each unplaced box relative to its nearest target?
- Which move brings a box closer to a target, or positions the player to push a box?
- Avoid pushing boxes into corners or against walls where they cannot reach a target.

On your LAST line write exactly: ACTION: up  (or down, left, right)"""


# ---------------------------------------------------------------------------
# QUICK SELF-TEST  — run: python representation.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    filepath = Path(__file__).parent.parent / "data" / "Microban.txt"
    all_puzzles = load_and_parse_all(str(filepath))
    state = from_parsed(all_puzzles[0])

    print("=" * 60)
    print("REPR 1: ASCII")
    print("=" * 60)
    print(to_ascii(state))

    print()
    print("=" * 60)
    print("REPR 2: STRUCTURED")
    print("=" * 60)
    print(to_structured(state))

    print()
    print("=" * 60)
    print("REPR 3: ANNOTATED")
    print("=" * 60)
    print(to_annotated(state))

    print()
    print("=" * 60)
    print("FULL PROMPT (ascii repr)")
    print("=" * 60)
    print(build_prompt(state, REPR_ASCII))
