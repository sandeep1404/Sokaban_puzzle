"""
environment.py — Module 2
Defines SokobanState and all game logic: move validation, action application,
win detection, deadlock detection, and rendering.

Build order: CODE THIS SECOND (after parser.py works)
Test: run `python environment.py` — manually walk through puzzle 1 step by step
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional
from parser import parse_puzzle, load_and_parse_all, WALL, FLOOR, TARGET
from pathlib import Path


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
# Map action name → (row_delta, col_delta)
ACTION_DELTAS = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}

                                                                                                                                                                                                                                                                      
#Moving up means going to the row above → row decreases by 1 → (-1, 0)                                                                                                                                                                                             
#Moving down means going to the row below → row increases by 1 → (+1, 0)                                                                                                                                                                                           
#Moving left means col decreases by 1 → (0, -1)                                                                                                                                                                                                                    
#Moving right means col increases by 1 → (0, +1)   

ACTIONS = list(ACTION_DELTAS.keys())


# ---------------------------------------------------------------------------
# DATA CLASS — the game state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)   # frozen=True makes it hashable (needed for visited set)
class SokobanState:
    """
    WHAT IT IS:
        Immutable snapshot of the puzzle at one point in time.
        'grid' holds only static elements (walls, targets, floor).
        Player and boxes are stored separately so we can hash (player, boxes)
        cheaply for the visited-set in tree search.

    FIELDS:
        grid            : tuple of tuple of str  — static board (#, ., ' ')
                          Use tuple-of-tuples (not list) so it's hashable.
        player_pos      : (row, col)
        box_positions   : frozenset of (row, col)  — frozenset is hashable
        target_positions: frozenset of (row, col)  — never changes during a game
        height          : int
        width           : int
    """
    grid            : tuple          # tuple[tuple[str, ...], ...]
    player_pos      : tuple          # (row, col)
    box_positions   : frozenset
    target_positions: frozenset
    height          : int
    width           : int




# ---------------------------------------------------------------------------
# FUNCTION 1
# ---------------------------------------------------------------------------
def from_parsed(puzzle: dict) -> SokobanState:
    """
    WHAT IT DOES:
        Converts a parsed puzzle dict (from parser.py) into a SokobanState.

    HOW IT WORKS:
        1. Convert puzzle["grid"] (list of lists) → tuple of tuples.
        2. Convert puzzle["box_positions"] (set) → frozenset.
        3. Convert puzzle["target_positions"] (set) → frozenset.
        4. Build and return SokobanState.

    ARGS:
        puzzle: dict from parse_puzzle()

    RETURNS:
        SokobanState

    EXAMPLE:
        state = from_parsed(all_puzzles[0])
    """
    # TODO: implement this (~5 lines)
    grid_tuple = tuple(tuple(row) for row in puzzle["grid"])
    
    return SokobanState(
        grid=grid_tuple,
        player_pos= puzzle['player_pos'],
        box_positions= frozenset(puzzle['box_positions']),
        target_positions= frozenset(puzzle['target_positions']),
        height= puzzle['height'],
        width= puzzle['width']            
                        
    )
    
    


# ---------------------------------------------------------------------------
# FUNCTION 2
# ---------------------------------------------------------------------------
def get_cell(state: SokobanState, row: int, col: int) -> str:
    """
    WHAT IT DOES:
        Returns what is at (row, col) considering dynamic pieces too.
        Checks bounds → boxes → player → static grid, in that order.

    RETURNS:
        '#' if out of bounds or wall
        '@' if player is here (and cell is not a target)
        '+' if player is here AND cell is a target
        '$' if box is here (and cell is not a target)
        '*' if box is here AND cell is a target
        '.' if it's a target (no box, no player)
        ' ' otherwise (floor)

    WHY:
        Used by render() to reconstruct the full visual board.
    """
    # TODO: implement this


    if row <0 or row>=state.height or col<0 or col>=state.width:
        return '#'
    pos = (row,col)

    if pos in state.target_positions:
        if pos == state.player_pos:
            return '+'
        if pos in state.box_positions:
            return '*'
    else:
        if pos == state.player_pos:
            return '@'
        if pos in state.box_positions:
            return '$'

    return state.grid[row][col]




# ---------------------------------------------------------------------------
# FUNCTION 3
# ---------------------------------------------------------------------------
def render(state: SokobanState) -> str:
    """
    WHAT IT DOES:
        Reconstructs the full visual board as a printable string.
        Calls get_cell() for every (row, col) position.

    RETURNS:
        Multi-line string of the board (rows joined with '\n').

    EXAMPLE:
        print(render(state))
        # ####
        # # .#
        # #  ###
        # #*@  #
        # #  $ #
        # #  ###
        # ####
    """
    # TODO: implement this (~5 lines using get_cell)

    out =[]
    for row in range(state.height):
        sub_out = ''
        for col in range(state.width):
            sub_out = sub_out + get_cell(state,row,col)
        
        # sub_out: str = sub_out + '\n'
        out.append(sub_out)
    out_final = '\n'.join(out)
    return out_final



# ---------------------------------------------------------------------------
# FUNCTION 4
# ---------------------------------------------------------------------------
def is_solved(state: SokobanState) -> bool:
    """
    WHAT IT DOES:
        Returns True if every box is on a target (puzzle is complete).

    HOW IT WORKS:
        box_positions == target_positions  (set equality)

    RETURNS:
        bool
    """
    # TODO: 1 line
    

    return state.box_positions == state.target_positions




# ---------------------------------------------------------------------------
# FUNCTION 5
# ---------------------------------------------------------------------------
def state_key(state: SokobanState) -> tuple:
    """
    WHAT IT DOES:
        Returns a compact hashable key for this state.
        Used as the key in the visited-set during tree search to avoid
        revisiting the same (player position, box configuration).

    RETURNS:
        tuple: (player_pos, box_positions)
        e.g.  ((2, 3), frozenset({(4, 3)}))
    """
    # TODO: 1 line

    box_pos = state.box_positions
    player_pos = state.player_pos

    return (player_pos,box_pos)
    


# ---------------------------------------------------------------------------
# FUNCTION 6
# ---------------------------------------------------------------------------
def is_valid_move(state: SokobanState, action: str) -> bool:
    """
    WHAT IT DOES:
        Checks if an action is legal from the current state WITHOUT
        actually applying it.

    RULES:
        1. Compute next_pos = player_pos + delta(action)
        2. If next_pos is a WALL → invalid
        3. If next_pos has a BOX:
               Compute beyond_pos = next_pos + delta(action)
               If beyond_pos is a WALL or another BOX → invalid (can't push)
        4. Otherwise → valid

    ARGS:
        state  : current SokobanState
        action : one of "up", "down", "left", "right"

    RETURNS:
        bool
    """
    # TODO: implement this (~15 lines)


    player_row, player_col = state.player_pos
    action_row, action_col = ACTION_DELTAS[action]
    next_row, next_col = player_row+action_row , player_col+action_col


    if next_row < 0 or next_row >= state.height or next_col<0 or next_col >= state.width:
        return False

    # Wall check
    if state.grid[next_row][next_col] == '#':
        return False
    
    # Box push check

    if (next_row, next_col) in state.box_positions:
        box_row, box_col = next_row + action_row , next_col + action_col ## since if the player moves up and if there is a box infront of him, then the box also moves up. (same with other directions as well)

        if box_row < 0 or box_row >= state.height or box_col < 0 or box_col >= state.width:
            return False
        
        ## if its a wall so move should not be defined 

        if state.grid[box_row][box_col] == '#':
            return False
        
        ## if there is an other box infront of the box 

        if (box_row,box_col) in state.box_positions:
            return False
        

    ## rest all are valid move 

    return True 



# ---------------------------------------------------------------------------
# FUNCTION 7
# ---------------------------------------------------------------------------
def get_valid_actions(state: SokobanState) -> list[str]:
    """
    WHAT IT DOES:
        Returns the list of all currently legal actions.

    HOW IT WORKS:
        Filter ACTIONS list by is_valid_move().

    RETURNS:
        list of action strings, subset of ["up", "down", "left", "right"]
    """
    # TODO: 1 line using list comprehension + is_valid_move
    


    return [action for action in ACTIONS if is_valid_move(state,action)]


# ---------------------------------------------------------------------------
# FUNCTION 8
# ---------------------------------------------------------------------------
## assuming this function only takes valid action 

def apply_action(state: SokobanState, action: str) -> SokobanState:
    """
    WHAT IT DOES:
        Applies an action and returns the NEW resulting SokobanState.
        Does NOT modify the original state (immutable design).

    HOW IT WORKS:
        1. Compute next_pos = player_pos + delta
        2. If next_pos has a box:
               Move box from next_pos → beyond_pos
               Update box_positions: remove next_pos, add beyond_pos
        3. Move player to next_pos
        4. Return new SokobanState with updated player_pos and box_positions

    IMPORTANT:
        Call is_valid_move() first if you're unsure — this function does
        NOT re-validate. Applying an invalid action = undefined behavior.

    ARGS:
        state  : current SokobanState
        action : one of "up", "down", "left", "right"

    RETURNS:
        new SokobanState
    """
    # TODO: implement this (~15 lines)


    player_row , player_col = state.player_pos ## current player pos 
    # current_box_positions = state.box_positions ## current box positions
    action_row, action_col = ACTION_DELTAS[action]
    new_row, new_col = player_row + action_row, player_col + action_col
    new_boxes = set(state.box_positions)

    ## if the new position has a box then the box will also move 
    if (new_row,new_col) in new_boxes:
        new_boxes.remove((new_row,new_col)) ## remove the old position and update with new one
        new_boxes.add((new_row+action_row,new_col+action_col))

    return SokobanState(
        grid= state.grid,
        player_pos= (new_row,new_col),
        box_positions= frozenset(new_boxes),
        target_positions= state.target_positions,
        height= state.height,
        width= state.width
    )


    
    # valid_actions = get_valid_actions(state)

    # for action in valid_actions:
    #     print(f'current action is {action}')

    #     next_row, next_col = player_row + ACTION_DELTAS[action], player_col + ACTION_DELTAS[action]

    #     if (next_row,next_col) in current_box_positions:

    #         new_box_row, new_box_col = next_row + ACTION_DELTAS[action] , next_col + ACTION_DELTAS[action]
            

    #     new_player_pos = (next_row,next_col)
    #     new_box_pos = (new_box_row,new_box_col)

    #     return SokobanState(grid=state.grid,player_pos=new_player_pos,box_positions=new)



# ---------------------------------------------------------------------------
# FUNCTION 9  (optional but very useful — prunes dead states)
# ---------------------------------------------------------------------------
def is_corner_deadlock(state: SokobanState) -> bool:
    """
    WHAT IT DOES:
        Detects the simplest deadlock: a box is stuck in a corner with
        no way to ever reach a target.

    DEFINITION OF CORNER DEADLOCK:
        A box is in a corner if it has walls (or borders) on two
        PERPENDICULAR sides simultaneously.
        e.g. wall above AND wall to the left → trapped, can never be pushed out.
        If that box is NOT on a target → deadlock (unsolvable from here).

    HOW IT WORKS:
        For each box position:
            Check 4 corner combinations:
                (up+left), (up+right), (down+left), (down+right)
            For each pair, if BOTH cells in that pair are walls/borders:
                AND the box is not on a target → return True

    WHY THIS MATTERS:
        Corner-deadlocked states can be pruned immediately during search,
        saving huge amounts of computation.

    RETURNS:
        True if ANY box is in an unrecoverable corner, else False
    """
    # TODO: implement this (~20 lines)

    box_pos = state.box_positions

    for box in box_pos:
        if box in state.target_positions:
            continue
        
        r,c = box 

        box_up = (r-1<0 or state.grid[r-1][c]=='#')
        box_down = (r+1>=state.height or state.grid[r+1][c]=='#')
        box_left = (c-1<0 or state.grid[r][c-1]=='#')
        box_right = (c+1>=state.width or state.grid[r][c+1]=='#')

        # Must be perpendicular walls forming an actual corner
        if (box_up and box_left) or (box_up and box_right) or (box_down and box_left) or (box_down and box_right):
            return True
    
    return False



# ---------------------------------------------------------------------------
# FUNCTION 10  (heuristic — used by MCTS/A* as a reward signal)
# ---------------------------------------------------------------------------
def heuristic_score(state: SokobanState) -> float:
    """
    WHAT IT DOES:
        Returns a float score estimating how close the state is to solved.
        Higher = better (closer to solution).

    TWO COMPONENTS (add them together):
        1. boxes_on_target / total_boxes
           (fraction of boxes already placed)
        2. -sum of min_manhattan_distance(box, nearest unmatched target)
           normalized by board size
           (penalizes states where boxes are far from targets)

    RETURNS:
        float in roughly [-1, 1] range
        0.0 = starting state, 1.0 = solved

    WHY:
        Used in MCTS rollout reward and in beam search to rank states
        when LLM confidence scores are not available.
    """
    # TODO: implement this (~20 lines)

    boxes = state.box_positions
    targets = state.target_positions
    n = len(targets)

    placed = len(boxes & targets)    # intersection
    if n == 0: return 1.0

    # Component 1: fraction placed
    frac = placed / n

    ## Component 2 : calculate distance from each unplaced box to all the targets 

    unplaced_boxes   = boxes - targets
    unmatched_targets = targets - boxes

    total_dist = 0

    for box in unplaced_boxes:
        min_d = min((abs(box[0]-target[0]) + abs(box[1]-target[1])) for target in unmatched_targets)
        total_dist += min_d
    
    # Normalize distance: divide by (board_area * n_boxes)

    board_area = state.height * state.width
    dist_penalty = total_dist / (board_area * max(1, len(unplaced_boxes)))

    return frac - dist_penalty

    



# ---------------------------------------------------------------------------
# QUICK SELF-TEST  — run: python environment.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    filepath = Path(__file__).parent / "data" / "Microban.txt"
    all_puzzles = load_and_parse_all(str(filepath))

    state = from_parsed(all_puzzles[5])
    print("=== Puzzle 1 Initial State ===")
    print(render(state))
    print(f"Solved: {is_solved(state)}")
    print(f"Valid actions: {get_valid_actions(state)}")
    print(f"Deadlock: {is_corner_deadlock(state)}")
    print(f"Heuristic: {heuristic_score(state):.3f}")
    print()

    # Apply a move and check result
    action = get_valid_actions(state)[0]
    print(f"Applying action: '{action}'")
    new_state = apply_action(state, action)
    print(render(new_state))
    print(f"State key changed: {state_key(state) != state_key(new_state)}")
