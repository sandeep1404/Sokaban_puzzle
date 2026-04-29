"""
search.py — Module 5
Two search algorithms that use the LLM as a guide:
  A) BeamSearchSolver  — simpler, implement first
  B) MCTSSolver        — more principled, implement second

Also contains a BFS baseline (no LLM) to verify your environment.py is
correct before adding LLM noise.

Build order: CODE THIS FIFTH
  Step 5a: BFS baseline first → confirm environment.py is bug-free
  Step 5b: BeamSearchSolver   → LLM-guided, main algorithm
  Step 5c: MCTSSolver         → optional, for comparison
Test: run `python search.py` — tries to solve puzzle 1 with all three methods
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import math
import time
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from environment import (
    SokobanState, from_parsed, apply_action, is_solved,
    get_valid_actions, is_corner_deadlock, heuristic_score, state_key
)
from llm_predictor import LLMPredictor, MLX_MODEL
from representation import REPR_ASCII
from parser import load_and_parse_all
from pathlib import Path



# ---------------------------------------------------------------------------
# ============================================================
#  PART A — BFS Baseline (NO LLM)
#  Code this FIRST to verify environment.py is correct
# ============================================================
# ---------------------------------------------------------------------------

def bfs_solve(initial_state: SokobanState,
              max_states: int = 5000_000) -> dict:
    """
    WHAT IT DOES:
        Solves Sokoban using plain BFS (breadth-first search).
        No LLM involved — explores all reachable states level by level.
        Use this to verify your environment.py is bug-free.
        If BFS can't solve puzzle 1, your environment has bugs.

    HOW IT WORKS:
        Standard BFS with a visited set:
        1. Initialize queue with (initial_state, path=[])
        2. Pop state from queue
        3. For each valid action:
               new_state = apply_action(state, action)
               if is_solved(new_state): return path + [action]
               if is_corner_deadlock(new_state): skip
               if state_key(new_state) in visited: skip
               Add to queue
        4. Return failure if queue empty or max_states reached

    ARGS:
        initial_state: SokobanState
        max_states   : int — stop exploring after this many states (safety limit)

    RETURNS:
        dict with keys:
            "solved"      : bool
            "path"        : list[str] | None   — action sequence
            "steps"       : int | None
            "states_explored": int
    """

    from collections import deque
    queue = deque()

    queue.append((initial_state,[]))

    visited_state = {state_key(initial_state)} ## gives the current state player pos and box pos 

    states_explored =0 

    while queue:
        state, path = queue.popleft()
        states_explored+=1

        if states_explored >max_states:
            break

        for action in get_valid_actions(state): ## get all the actions for a current state
            new_state = apply_action(state,action) ##  for each action get the new state by applying that action 

            new_path = path+ [action] ## add to path how did we reach that action 

            ## check if the new state is a deadlock or solved state 

            if is_solved(new_state):
                return {"solved": True, "path": new_path,
                    "steps": len(new_path), "states_explored": states_explored}

            key = state_key(new_state)

            if key in visited_state:
                continue

            if is_corner_deadlock(new_state):
                continue

            visited_state.add(key)
            queue.append((new_state,new_path))

    return {"solved": False, "path": None, "steps": None,
        "states_explored": states_explored}

            

# def _any_push_available(state: SokobanState) -> bool:
#     """Return True if any valid action would push a box."""
#     pr, pc = state.player_pos
#     for action in get_valid_actions(state):
#         dr, dc = ACTION_DELTAS[action]
#         nr, nc = pr + dr, pc + dc
#         if (nr, nc) in state.box_positions:
#             return True
#     return False

# ---------------------------------------------------------------------------
# ============================================================
#  PART B — Beam Search (LLM-guided)
#  Code this SECOND — main algorithm
# ============================================================
# ---------------------------------------------------------------------------

class BeamSearchSolver:
    """
    WHAT IT IS:
        LLM-guided beam search. Maintains a "beam" of the top-k most
        promising states at each step. At each step, asks the LLM for
        action rankings, generates candidate next-states, then keeps
        only the top beam_width by cumulative score.

    BEAM SEARCH LOGIC:
        beam = [(initial_state, path=[], score=1.0)]
        for each step:
            candidates = []
            for (state, path, score) in beam:
                top_k_actions = llm.predict_top_k(state, k=beam_width)
                for (action, action_prob) in top_k_actions:
                    if action not in valid_actions: skip
                    new_state = apply_action(state, action)
                    if is_solved(new_state): RETURN path + [action]  ← SUCCESS
                    if is_corner_deadlock(new_state): skip
                    if state_key in visited: skip
                    candidates.append((new_state, path+[action], score * action_prob))

            # prune to top beam_width
            beam = sorted(candidates, by score desc)[:beam_width]
            visited.update(all states in beam)

            if beam is empty: RETURN failure

        RETURN failure (max_depth reached)

    WHY BEAM SEARCH:
        - Simple to implement
        - beam_width is a direct trade-off: wider = more LLM calls but more thorough
        - Naturally handles LLM uncertainty by keeping multiple options open
    """

    def __init__(self, predictor: LLMPredictor,
                 beam_width: int = 5,
                 max_depth: int = 150,
                 max_llm_calls: int = 500):
        """
        WHAT IT DOES:
            Stores the configuration.

        ARGS:
            predictor      : LLMPredictor instance
            beam_width     : int — how many states to keep at each step
            max_depth      : int — max steps before giving up
            max_llm_calls  : int — stop early if LLM calls exceed this
        """
        self.predictor = predictor
        self.beam_width = beam_width                                                                                                                                                                       
        self.max_depth = max_depth
        self.max_llm_calls = max_llm_calls

    def solve(self, initial_state: SokobanState) -> dict:
        """
        WHAT IT DOES:
            Runs beam search from initial_state until solved or max_depth reached.

        RETURNS:
            dict with keys:
                "solved"    : bool
                "path"      : list[str] | None
                "steps"     : int | None
                "llm_calls" : int
                "states_explored": int
        """
      

        # llm_calls=0
        # states_explored=0
        # llm_predict = LLMPredictor()
        # action_list = llm_predict.predict_top_k(initial_state,self.beam_width,8)

        # visited_state = {state_key(initial_state)}
        # depth =0
        # path = []



        # for action_ele in action_list:
        #     action, confidence = action_ele
        #     new_state = apply_action(initial_state,action)
        #     path = path + [action]
        #     states_explored+=1
        #     llm_calls+=1
        #     depth = depth+1

        #     if is_solved(new_state):
        #         return {"solved":True, "path":path, "steps":len(path),'llm_calls':llm_calls,'stats_explored':states_explored}

        #     if depth >self.max_depth:
        #         return "Max depth exceeded"

        
        beam = [(initial_state, [], heuristic_score(initial_state))]
        visited = {state_key(initial_state)}
        states_explored = 0
        fallback_count = 0

        for step in range(self.max_depth):
            candidates = []

            if step % 20 == 0:
                print(f"  step={step}, beam={len(beam)}, visited={len(visited)}, "
                      f"llm_calls={self.predictor.call_count}", flush=True)

            # Early termination if LLM call budget exhausted
            if self.predictor.call_count >= self.max_llm_calls:
                break

            # --- Batch LLM calls for all beam states (thread-safe, real batching) ---
            # predict_batch_states fires concurrent HTTP requests to the vLLM server
            # which processes them in one continuous batch.  No shared mutable state
            # is accessed from threads — cache/counter updates are lock-protected.
            beam_states = [s for s, _, _ in beam]
            all_predictions = self.predictor.predict_batch_states(beam_states, k=4)

            for beam_idx, (state, path, _) in enumerate(beam):
                valid_actions = get_valid_actions(state)
                if not valid_actions:
                    continue

                predicted_actions = all_predictions[beam_idx]
                llm_scores = {a: p for a, p in predicted_actions if a in valid_actions}

                # Fallback: if LLM returned no valid actions, use uniform scores
                if not llm_scores:
                    llm_scores = {a: 1.0 / len(valid_actions) for a in valid_actions}
                    fallback_count += 1

                for action, action_prob in llm_scores.items():
                    new_state = apply_action(state, action)
                    new_path = path + [action]
                    states_explored += 1

                    if is_solved(new_state):
                        return {
                            "solved": True,
                            "path": new_path,
                            "steps": len(new_path),
                            "llm_calls": self.predictor.call_count,
                            "states_explored": states_explored,
                            "fallback_count": fallback_count,
                        }

                    key = state_key(new_state)
                    if key in visited:
                        continue

                    if is_corner_deadlock(new_state):
                        continue

                    h_score = heuristic_score(new_state)
                    pushed = new_state.box_positions != state.box_positions
                    push_bonus = 0.3 if pushed else 0.0
                    score = h_score + 0.1 * action_prob + push_bonus

                    candidates.append((new_state, new_path, score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:self.beam_width]
            for s, _, _ in beam:
                visited.add(state_key(s))

        return {
            "solved": False,
            "path": None,
            "steps": None,
            "llm_calls": self.predictor.call_count,
            "states_explored": states_explored,
            "fallback_count": fallback_count,
        }


# ---------------------------------------------------------------------------
# ============================================================
#  PART C — MCTS  (LLM-guided Monte Carlo Tree Search)
#  Code this THIRD — optional but impressive
# ============================================================
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ============================================================
#  PART B2 — A* Search (LLM-boosted, guaranteed to find solution)
#  Use this when beam search fails — A* never permanently discards states
# ============================================================
# ---------------------------------------------------------------------------

class AStarSolver:
    """
    WHY A* BEATS BEAM SEARCH FOR SOKOBAN:
        Beam search permanently prunes states — when the beam collapses it
        can't recover. Sokoban requires moving boxes AWAY from goals to set
        up future pushes; a greedy heuristic prunes exactly those moves.

        A* uses a priority queue (open set). Every candidate stays in the
        queue until visited. It is GUARANTEED to find the solution if one
        exists, regardless of LLM quality.

    HOW LLM IS USED:
        LLM predictions add a small bonus to f-score, biasing expansion
        order toward LLM-preferred actions. If the LLM is wrong, the
        correct path still gets explored — just slightly later.

        f(n) = g(n) - h(n) - llm_boost(n)
          g(n)        = steps taken so far (path cost)
          h(n)        = heuristic_score (higher = better, so we negate it)
          llm_boost   = small bonus from LLM prediction (0 to 0.3)

    USE: Call solve(initial_state) — same API as BeamSearchSolver.
    """

    def __init__(self, predictor: LLMPredictor,
                 max_states: int = 20_000,
                 llm_weight: float = 0.3,
                 max_llm_calls: int = 0,
                 batch_size: int = 16):
        self.predictor = predictor
        self.max_states = max_states
        self.llm_weight = llm_weight
        # max_llm_calls=0 means unlimited per puzzle.
        # Set a positive value to cap LLM calls per puzzle and fail fast.
        self.max_llm_calls_per_puzzle = max_llm_calls if max_llm_calls > 0 else float('inf')
        # batch_size: how many states to pop from the heap per iteration.
        # All their children are collected and sent to the LLM as one batch.
        # With vLLM this is nearly free (continuous batching on server side),
        # so larger values reduce wall-clock time without hurting accuracy.
        self.batch_size = batch_size

    def solve(self, initial_state: SokobanState) -> dict:
        # Priority queue entries: (priority, tie_break, state, path)
        # Lower priority = expanded first
        # priority = g_cost - h_score - llm_boost  (we minimize this)
        counter = 0  # tie-breaker to avoid comparing SokobanState objects

        init_h = heuristic_score(initial_state)
        init_key = state_key(initial_state)
        open_set = [(0.0 - init_h, counter, initial_state, [])]
        heapq.heapify(open_set)

        visited = {}  # state_key -> best g_cost seen
        visited[init_key] = 0
        states_explored = 0
        llm_calls_start = self.predictor.call_count
        fallback_count = 0  # times LLM returned nothing valid → used all valid actions

        while open_set and states_explored < self.max_states and (self.predictor.call_count - llm_calls_start) < self.max_llm_calls_per_puzzle:
            # --- Pop up to batch_size best states from the heap ---
            batch = []
            while open_set and len(batch) < self.batch_size:
                batch.append(heapq.heappop(open_set))

            # Check for solutions in the popped batch first
            for priority, _, state, path in batch:
                if is_solved(state):
                    return {
                        "solved": True,
                        "path": path,
                        "steps": len(path),
                        "llm_calls": self.predictor.call_count,
                        "states_explored": states_explored,
                        "fallback_count": fallback_count,
                    }

            # --- Step 1: Get LLM predictions for all batch states FIRST ---
            # The LLM decides WHICH actions to explore from each state.
            # We only expand actions the LLM recommends (intersected with valid).
            # If the LLM predicts "up" but only "left/right" are valid, "up" is
            # discarded — we never blindly trust an invalid LLM action.
            # Only when the LLM returns zero valid actions do we fall back to all
            # valid actions (parse failure safety net, not a normal code path).
            batch_states_for_llm = [state for _, _, state, _ in batch]
            batch_predictions = self.predictor.predict_batch_states(batch_states_for_llm, k=4)

            # --- Step 2: Expand only LLM-selected actions ---
            # action_prob_map[i] = {action: prob} for batch[i], LLM-filtered to valid
            expansions: list[tuple] = []  # (parent_state, path, action, child_state, new_g, prob)
            for i, (_priority, _, state, path) in enumerate(batch):
                states_explored += 1
                g_cost = len(path)

                if states_explored % 200 == 0:
                    print(f"  A* states={states_explored}, open={len(open_set)}, "
                          f"llm_calls={self.predictor.call_count} "
                          f"(LLM selects actions), "
                          f"cache={self.predictor.cache_hits}", flush=True)

                valid_actions = set(get_valid_actions(state))
                if not valid_actions:
                    continue

                # Keep only LLM-predicted actions that are physically valid
                preds = batch_predictions[i]  # list of (action, prob)
                llm_valid = [(a, p) for a, p in preds if a in valid_actions]

                # Fallback only on LLM parse failure (returns nothing usable)
                if not llm_valid:
                    fallback_count += 1
                    llm_valid = [(a, 1.0 / len(valid_actions)) for a in valid_actions]

                for action, prob in llm_valid:
                    new_state = apply_action(state, action)

                    if is_corner_deadlock(new_state):
                        continue

                    new_g = g_cost + 1
                    key = state_key(new_state)
                    prev_g = visited.get(key)
                    if prev_g is not None and prev_g <= new_g:
                        continue

                    visited[key] = new_g
                    expansions.append((state, path, action, new_state, new_g, prob))

            if not expansions:
                continue

            # --- Step 3: Push LLM-selected children into the heap ---
            for (parent_state, path, action, child_state, new_g, prob) in expansions:
                if is_solved(child_state):
                    return {
                        "solved": True,
                        "path": path + [action],
                        "steps": new_g,
                        "llm_calls": self.predictor.call_count,
                        "states_explored": states_explored,
                        "fallback_count": fallback_count,
                    }

                h = heuristic_score(child_state)
                # LLM probability directly boosts priority — higher prob = explored sooner
                boost = self.llm_weight * prob
                pushed = child_state.box_positions != parent_state.box_positions
                push_bonus = 0.2 if pushed else 0.0

                new_priority = new_g - h - boost - push_bonus
                counter += 1
                heapq.heappush(open_set, (new_priority, counter,
                                          child_state, path + [action]))

        llm_calls_used = self.predictor.call_count - llm_calls_start
        hit_llm_cap = llm_calls_used >= self.max_llm_calls_per_puzzle
        hit_state_cap = states_explored >= self.max_states
        if hit_llm_cap:
            print(f"  A* LLM CAP reached ({llm_calls_used} calls, {states_explored} states, {fallback_count} fallbacks) — marking as failure", flush=True)
        elif hit_state_cap:
            print(f"  A* STATE CAP reached ({states_explored} states, {llm_calls_used} LLM calls, {fallback_count} fallbacks) — marking as failure", flush=True)
        return {
            "solved": False,
            "path": None,
            "steps": None,
            "llm_calls": self.predictor.call_count,
            "states_explored": states_explored,
            "fallback_count": fallback_count,
        }


@dataclass
class MCTSNode:
    """
    WHAT IT IS:
        One node in the MCTS tree. Represents a game state.

    FIELDS:
        state       : SokobanState at this node
        parent      : parent MCTSNode (None for root)
        action_taken: action that led from parent to this node
        children    : list of child MCTSNode
        visit_count : how many times MCTS has visited this node
        total_reward: sum of rewards from all visits (for Q value)
        llm_prior   : LLM's predicted probability for this action (0-1)
                      Used in PUCT score to bias toward LLM-favored moves
        is_terminal : bool — True if state is solved or deadlocked
    """
    state        : SokobanState
    parent       : Optional['MCTSNode'] = field(default=None, repr=False)
    action_taken : Optional[str] = None
    children     : list = field(default_factory=list)
    visit_count  : int = 0
    total_reward : float = 0.0
    llm_prior    : float = 0.25   # uniform default before LLM expansion
    is_terminal  : bool = False

    def q_value(self) -> float:
        """Average reward = total_reward / visit_count (0 if never visited)"""
        avg_reward = 0 
        if self.visit_count>0:
            avg_reward = self.total_reward/self.visit_count 
        return avg_reward
    



    def puct_score(self, exploration_c: float = 1.4) -> float:
        """
        PUCT = Q(node) + c * prior * sqrt(parent.visit_count) / (1 + node.visit_count)
        """
        if self.parent is None:
            return 0.0
        q = self.q_value()
        exploration = (exploration_c * self.llm_prior
                       * math.sqrt(self.parent.visit_count)
                       / (1 + self.visit_count))
        return q + exploration


class MCTSSolver:
    """
    WHAT IT IS:
        MCTS solver with LLM as the policy network (via llm_prior).

    MCTS LOOP (4 phases repeated n_iterations times):
        1. SELECT   — walk tree from root, at each node pick child with highest PUCT score
                      until reaching an unexpanded leaf node
        2. EXPAND   — call LLM to get action priors, create child nodes
        3. ROLLOUT  — from the new node, simulate a random/heuristic game
                      to get a reward signal
        4. BACKPROP — walk back up to root, updating visit_count and total_reward

    AFTER n_iterations:
        Follow the path of highest visit_count from root → return as solution
    """

    def __init__(self, predictor: LLMPredictor,
                 n_iterations: int = 200,
                 exploration_c: float = 1.4,
                 rollout_depth: int = 20,
                 max_llm_calls: int = 300):
        self.predictor = predictor
        self.n_iterations = n_iterations
        self.exploration_c = exploration_c
        self.rollout_depth = rollout_depth
        self.max_llm_calls = max_llm_calls
        # Global visited set across the whole tree — prevents cycles
        self._visited: set = set()

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk tree picking highest PUCT child until reaching a leaf."""
        while node.children:
            # Filter out terminal children and pick from the rest
            live = [c for c in node.children if not c.is_terminal]
            if not live:
                node.is_terminal = True
                return node
            # If any child is unvisited, pick it
            unvisited = [c for c in live if c.visit_count == 0]
            if unvisited:
                return unvisited[0]
            # Otherwise pick best PUCT
            node = max(live, key=lambda c: c.puct_score(self.exploration_c))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Create children using LLM priors, skip duplicate/deadlock states."""
        if node.is_terminal or node.children:
            return node

        valid = get_valid_actions(node.state)
        if not valid:
            node.is_terminal = True
            return node

        # Get LLM priors (single call)
        if self.predictor.call_count < self.max_llm_calls:
            ranked = self.predictor.predict_top_k(node.state, k=4, n_samples=1)
            prior_map = {a: p for a, p in ranked}
        else:
            prior_map = {}

        for action in valid:
            new_state = apply_action(node.state, action)
            key = state_key(new_state)

            # Skip states we've already seen anywhere in the tree
            if key in self._visited:
                continue

            if is_corner_deadlock(new_state):
                continue

            self._visited.add(key)
            prior = prior_map.get(action, 0.05)
            solved = is_solved(new_state)
            child = MCTSNode(
                state=new_state,
                parent=node,
                action_taken=action,
                llm_prior=prior,
                is_terminal=solved,  # only terminal if solved (deadlocks already filtered)
            )
            node.children.append(child)

        if not node.children:
            node.is_terminal = True
            return node

        # Return first unvisited child
        return node.children[0]

    def _rollout(self, node: MCTSNode) -> float:
        """Heuristic-guided simulation from node's state; return reward in [0, 1]."""
        import random
        if is_solved(node.state):
            return 1.0
        if node.is_terminal:  # deadlock
            return 0.0

        state = node.state
        best_h = heuristic_score(state)

        for _ in range(self.rollout_depth):
            valid = get_valid_actions(state)
            if not valid:
                return 0.0

            # Heuristic-guided: pick the action that improves score most
            # (with 20% randomness to avoid getting stuck in loops)
            if random.random() < 0.8:
                best_action = None
                best_score = -999
                for a in valid:
                    s = apply_action(state, a)
                    if is_corner_deadlock(s):
                        continue
                    h = heuristic_score(s)
                    # Bonus for pushing a box
                    pushed = s.box_positions != state.box_positions
                    score = h + (0.2 if pushed else 0.0)
                    if score > best_score:
                        best_score = score
                        best_action = a
                if best_action is None:
                    best_action = random.choice(valid)
                action = best_action
            else:
                action = random.choice(valid)

            state = apply_action(state, action)
            if is_solved(state):
                return 1.0
            if is_corner_deadlock(state):
                return 0.0

        # Reward based on improvement from starting node
        h = heuristic_score(state)
        # Scale: if h improved over start, reward > 0.5; if worse, reward < 0.5
        improvement = h - best_h
        return max(0.0, min(1.0, 0.5 + improvement))

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Walk back to root, updating visit counts and rewards."""
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def _extract_path(self, node: MCTSNode) -> list[str]:
        """Walk from node back to root and return action sequence."""
        path = []
        while node.parent is not None:
            path.append(node.action_taken)
            node = node.parent
        path.reverse()
        return path

    def solve(self, initial_state: SokobanState) -> dict:
        """
        Run MCTS for n_iterations, then extract best path by following
        highest visit_count children from root.
        """
        root = MCTSNode(state=initial_state)
        self._visited = {state_key(initial_state)}
        best_solution = None
        states_explored = 0

        for i in range(self.n_iterations):
            if i % 50 == 0:
                print(f"  mcts iter={i}/{self.n_iterations}, "
                      f"llm_calls={self.predictor.call_count}, "
                      f"visited={len(self._visited)}", flush=True)

            # 1. Select
            leaf = self._select(root)
            states_explored += 1

            if leaf.is_terminal:
                if is_solved(leaf.state):
                    path = self._extract_path(leaf)
                    if best_solution is None or len(path) < len(best_solution):
                        best_solution = path
                reward = 1.0 if is_solved(leaf.state) else 0.0
                self._backpropagate(leaf, reward)
                continue

            # 2. Expand — always expand leaf nodes (don't wait for second visit)
            leaf = self._expand(leaf)
            states_explored += len(leaf.parent.children) if leaf.parent and leaf.parent.children else 1

            # Check if expansion found solution
            if is_solved(leaf.state):
                path = self._extract_path(leaf)
                if best_solution is None or len(path) < len(best_solution):
                    best_solution = path
                self._backpropagate(leaf, 1.0)
                continue

            # 3. Rollout
            reward = self._rollout(leaf)

            # 4. Backpropagate
            self._backpropagate(leaf, reward)

        # If we found a real solution during search, return it
        if best_solution is not None:
            return {
                "solved": True,
                "path": best_solution,
                "steps": len(best_solution),
                "llm_calls": self.predictor.call_count,
                "states_explored": states_explored,
                "fallback_count": 0,
            }

        # Otherwise, extract most-visited path from root
        path = []
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visit_count)
            path.append(node.action_taken)
            if is_solved(node.state):
                return {
                    "solved": True,
                    "path": path,
                    "steps": len(path),
                    "llm_calls": self.predictor.call_count,
                    "states_explored": states_explored,
                    "fallback_count": 0,
                }

        return {
            "solved": False,
            "path": None,
            "steps": None,
            "llm_calls": self.predictor.call_count,
            "states_explored": states_explored,
            "fallback_count": 0,
        }


# ---------------------------------------------------------------------------
# QUICK SELF-TEST  — run: python search.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    filepath = Path(__file__).parent / "data" / "Microban.txt"
    all_puzzles = load_and_parse_all(str(filepath))

    puzzle_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    state = from_parsed(all_puzzles[puzzle_idx])
    print(f"Puzzle index: {puzzle_idx}")

    # --- BFS baseline (no LLM) ---
    print("\n=== BFS Baseline ===")
    result = bfs_solve(state, max_states=100_000)
    print(f"Solved: {result['solved']}")
    print(f"Steps : {result['steps']}")
    print(f"States explored: {result['states_explored']}")

    # ----------------------------------------------------------------
    # Choose backend:
    #   Option A (current)  — Ollama CPU:  ~15 tok/s, easy setup
    #   Option B (fastest)  — MLX GPU:    ~100-150 tok/s on M3, install mlx-lm
    #
    # To use MLX:
    #   pip install mlx-lm
    #   (model auto-downloads from HuggingFace on first run)
    # ----------------------------------------------------------------
    USE_MLX = False  # set True to use MLX on your M3 Pro GPU

    if USE_MLX:
        predictor = LLMPredictor(model_name=MLX_MODEL, backend="mlx")
        print(f"\nUsing MLX backend: {MLX_MODEL}")
    else:
        predictor = LLMPredictor(model_name='claude-sonnet-4-6',backend='groq')  # Ollama default
        print(f"\nUsing claude backend: {predictor.model_name}")

    # # --- Beam Search ---
    print("\n=== Beam Search (LLM-guided) ===")
    predictor.reset_call_count()
    solver = BeamSearchSolver(predictor, beam_width=10, max_depth=300)
    result = solver.solve(state)
    print(f"Solved    : {result['solved']}")
    print(f"Steps     : {result['steps']}")
    print(f"LLM calls : {result['llm_calls']}")
    print(f"Cache hits: {predictor.cache_hits}")
    print(f"States    : {result['states_explored']}")
    print(f"Path to solution: {result['path']}")

    # # --- A* Search (guaranteed to solve if BFS can) ---
    # print("\n=== A* Search (LLM-boosted) ===")
    # predictor.reset_call_count()
    # # max_llm_calls=200: use LLM for first 200 push-states, then pure heuristic A*
    # # This caps wall-clock LLM time while still solving the puzzle
    # astar = AStarSolver(predictor, max_states=50_000, llm_weight=0.3, max_llm_calls=100)
    # result = astar.solve(state)
    # print(f"Solved    : {result['solved']}")
    # print(f"Steps     : {result['steps']}")
    # print(f"LLM calls : {result['llm_calls']}")
    # print(f"Cache hits: {predictor.cache_hits}")
    # print(f"States    : {result['states_explored']}")

    # # --- MCTS (LLM-guided Monte Carlo Tree Search) ---
    # print("\n=== MCTS (LLM-guided) ===")
    # predictor.reset_call_count()
    # mcts = MCTSSolver(predictor, n_iterations=1500, exploration_c=1.4,
    #                   rollout_depth=50, max_llm_calls=300)
    # result = mcts.solve(state)
    # print(f"Solved    : {result['solved']}")
    # print(f"Steps     : {result['steps']}")
    # print(f"LLM calls : {result['llm_calls']}")
    # print(f"States    : {result['states_explored']}")

