"""
parser.py — Module 1
Loads the Microban.txt file and converts each puzzle block into a
structured dict that the rest of the system can use.

Build order: CODE THIS FIRST
Test: run `python parser.py` — it should print puzzle 1 grid + detected positions
"""

# ---------------------------------------------------------------------------
# WHAT YOU NEED TO IMPORT
# ---------------------------------------------------------------------------
# (no external deps — stdlib only)
from pathlib import Path


# ---------------------------------------------------------------------------
# CONSTANTS  — Sokoban cell symbols
# ---------------------------------------------------------------------------
WALL        = '#'
FLOOR       = ' '
PLAYER      = '@'
BOX         = '$'
TARGET      = '.'
BOX_ON_TGT  = '*'   # box already on a target
PLAYER_ON_TGT = '+' # player standing on a target


# ---------------------------------------------------------------------------
# FUNCTION 1
# ---------------------------------------------------------------------------

def load_microban(filepath: str) -> list[str]:
    """
    WHAT IT DOES:
        Reads the raw Microban.txt file and splits it into a list of
        raw puzzle strings — one string per puzzle.

    HOW IT WORKS:
        1. Read all lines from the file.
        2. Lines starting with ';' are comments (puzzle title/number) — strip them.
        3. Split the remaining lines into groups separated by blank lines.
        4. Discard empty groups.
        5. Join each group back into a single multi-line string.

    ARGS:
        filepath: path to Microban.txt

    RETURNS:
        list of raw puzzle strings, e.g. ["####\n# .#\n...", "######\n..."]

    EXAMPLE:
        puzzles = load_microban("data/Microban.txt")
        print(len(puzzles))   # should be 155 puzzles in Microban
        print(puzzles[0])     # raw text of puzzle 1
    """
    # TODO: implement this
    
    groups =[]
    current_group =[]
    with open(filepath,'r') as file:
        for line in file:
            if line.strip() == "":
                if current_group:
                    # print('hi')
                    groups.append(current_group)
                    # print(groups)
                    current_group=[]
            else:
                # print('else block in hello')
                current_group.append(line.rstrip())
                # print(current_group)
                # print(groups)

        if current_group:
            groups.append(current_group)

    final_groups=[]

    for i in range(len(groups)):
        if i%2!=0:
            result = '\n'.join(groups[i])
            final_groups.append(result)

    return final_groups 


# ---------------------------------------------------------------------------
# FUNCTION 2
# ---------------------------------------------------------------------------
def parse_puzzle(raw_block: str, puzzle_idx: int = 0) -> dict:
    """
    WHAT IT DOES:
        Converts a raw puzzle string into a structured dict with all
        positions extracted.

    HOW IT WORKS:
        1. Split raw_block into lines (split on '\n').
        2. Walk every (row, col) cell character.
        3. Based on the character:
            '@' or '+' → record as player_pos, mark cell as FLOOR (or TARGET if '+')
            '$' or '*' → add to box_positions, mark cell as FLOOR (or TARGET if '*')
            '.'        → add to target_positions, mark cell as TARGET
            '#'        → keep as WALL
            ' '        → FLOOR
        4. Build a 2D list `grid` of STATIC elements only (walls + targets).
           Player and boxes are NOT stored in the grid — they are dynamic.

    ARGS:
        raw_block:  multi-line string for one puzzle
        puzzle_idx: int index (for labelling, default 0)

    RETURNS:
        dict with keys:
            "idx"              : int
            "grid"             : list[list[str]]  — 2D grid, static only (#, ., ' ')
            "player_pos"       : tuple(row, col)
            "box_positions"    : set of tuple(row, col)
            "target_positions" : set of tuple(row, col)
            "height"           : int  (number of rows)
            "width"            : int  (max number of cols)

    EXAMPLE:
        puzzle = parse_puzzle(puzzles[0], puzzle_idx=0)
        print(puzzle["player_pos"])       # e.g. (3, 2)
        print(puzzle["box_positions"])    # e.g. {(4, 3)}
        print(puzzle["target_positions"]) # e.g. {(1, 2)}
    """
    # TODO: implement this
    out ={}
    lines = raw_block.split('\n')
    grid = [] 
    box_positions = set()
    target_positions = set() 
    player_pos = None
    for row_idx, line in enumerate(lines):
        grid_row =[]
        for col_idx, char in enumerate(line):
            # print(col_id)
            # print(char)
            if char == '@':
                player_pos = (row_idx, col_idx)
                grid_row.append(' ') 
            elif char == '+':
                player_pos = (row_idx, col_idx)
                grid_row.append('.')
                target_positions.add((row_idx, col_idx))
            elif char == '$':
                box_positions.add((row_idx,col_idx))
                grid_row.append(' ') 
            elif char == '*':
                box_positions.add((row_idx, col_idx))
                target_positions.add((row_idx,col_idx))
                grid_row.append('.') 
            else:
                grid_row.append(char)
                if char == '.':
                    target_positions.add((row_idx, col_idx))
        grid.append(grid_row)

    height = len(grid)
    width = (max(len(row) for row in grid))

    final_grid = [row + [' ']*(width-len(row)) for row in grid] ## add padding

    out['idx'] = puzzle_idx
    out['grid'] = final_grid
    out['player_pos'] = player_pos
    out['target_positions'] = target_positions
    out['box_positions']= box_positions
    out['height'] = height
    out['width'] = width

    return out 



# ---------------------------------------------------------------------------
# FUNCTION 3  (helper)
# ---------------------------------------------------------------------------
def load_and_parse_all(filepath: str) -> list[dict]:
    """
    WHAT IT DOES:
        Convenience wrapper — calls load_microban() then parse_puzzle()
        for every raw block. Returns a list of parsed puzzle dicts.

    ARGS:
        filepath: path to Microban.txt

    RETURNS:
        list of parsed puzzle dicts (one per puzzle)

    EXAMPLE:
        all_puzzles = load_and_parse_all("data/Microban.txt")
        print(f"Loaded {len(all_puzzles)} puzzles")
    """
    # TODO: implement this (2-3 lines using the two functions above)
    puzzles = load_microban(filepath)
    puzzle_dict = []
    for i in range(len(puzzles)):
        puzzle_dict_perpuzzle = parse_puzzle(raw_block=puzzles[i],puzzle_idx=i)
        puzzle_dict.append(puzzle_dict_perpuzzle)
    
    return puzzle_dict




# ---------------------------------------------------------------------------
# QUICK SELF-TEST  — run: python parser.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # filepath = Path(__file__).parent.parent / "data" / "microban_test.txt"
    filepath =  './data/Microban.txt'
    all_puzzles = load_and_parse_all(str(filepath))

    print(f"Total puzzles loaded: {len(all_puzzles)}")
    print()

    p = all_puzzles[10]
    print(f"=== Puzzle 1 ===")
    print(f"Size: {p['height']} rows x {p['width']} cols")
    print(f"Player: {p['player_pos']}")
    print(f"Boxes:  {p['box_positions']}")
    print(f"Targets:{p['target_positions']}")
    print()
    print("Grid (static):")
    print(f'the value of p is {p}')
    for row in p["grid"]:
        print("".join(row))
