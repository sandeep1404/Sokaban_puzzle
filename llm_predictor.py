"""
llm_predictor.py — Module 4
Wraps a local LLM (via Ollama or HuggingFace transformers) and exposes
two methods: predict_action() for single best action, predict_top_k()
for ranked action list used by beam search.

Build order: CODE THIS FOURTH (after representation.py works)
Test: run `python llm_predictor.py` — should print predicted action for puzzle 1

SETUP BEFORE CODING:
    Option A — Ollama (RECOMMENDED, easiest):
        1. Install: https://ollama.com
        2. Pull model: `ollama pull mistral` or `ollama pull llama3`
        3. Start server: `ollama serve`
        4. pip install ollama

    Option B — HuggingFace transformers (heavier, needs GPU or patience):
        pip install transformers torch accelerate
        Use: "mistralai/Mistral-7B-Instruct-v0.2" or "meta-llama/Llama-3.2-1B-Instruct"
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import re
import json
from pathlib import Path
from environment import SokobanState, from_parsed, get_valid_actions
from representation import build_prompt, build_prompt_with_reasoning, REPR_ASCII, REPR_ANNOTATED
from parser import load_and_parse_all

from typing import Optional

# Uncomment ONE of these based on your setup:
try:
    import ollama                          # Option A
except ImportError:
    ollama = None  # not installed on this machine; use vllm/groq backend instead

# MLX-LM: Apple Silicon GPU inference — 6-10x faster than Ollama on M1/M2/M3
# Install: pip install mlx-lm
# First use auto-downloads the model from HuggingFace Hub
try:
    from mlx_lm import load as _mlx_load, generate as _mlx_generate
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

try:
    from mlx_lm.server import make_prompt_cache as _make_prompt_cache
    from mlx_lm.generate import generate_step as _generate_step
    import mlx.core as mx
    _MLX_CACHE_AVAILABLE = True
except ImportError:
    _MLX_CACHE_AVAILABLE = False

# Module-level cache so the model is loaded only once
_MLX_MODEL_CACHE: dict = {}
# Module-level prompt cache for KV reuse across calls
_MLX_PROMPT_CACHE: dict = {}

def _get_mlx_model(model_name: str):
    """Load (or retrieve cached) MLX model + tokenizer."""
    if model_name not in _MLX_MODEL_CACHE:
        print(f"[MLX] Loading {model_name} onto Metal GPU (first call only)...", flush=True)
        model, tokenizer = _mlx_load(model_name)
        _MLX_MODEL_CACHE[model_name] = (model, tokenizer)
    return _MLX_MODEL_CACHE[model_name]


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
VALID_ACTIONS = ["up", "down", "left", "right"]

# Default model names — change to whatever you have available
OLLAMA_MODEL = "llama3.2:latest"
# MLX model — Apple Silicon GPU inference. Choose one:
#   "mlx-community/Qwen2.5-7B-Instruct-4bit"   — best quality, ~100 tok/s on M3
#   "mlx-community/Qwen2.5-3B-Instruct-4bit"   — faster, ~180 tok/s on M3
#   "mlx-community/Qwen2.5-1.5B-Instruct-4bit" — fastest Qwen, ~350 tok/s on M3
#   "mlx-community/Qwen2.5-0.5B-Instruct-4bit" — ultra-fast, ~500 tok/s on M3
#   "mlx-community/Llama-3.2-1B-Instruct-4bit" — smallest Llama, ~400 tok/s
#   "mlx-community/Llama-3.2-3B-Instruct-4bit" — alternative 3B
#   "mlx-community/SmolLM2-1.7B-Instruct-4bit" — very fast for short tasks
MLX_MODEL    = "mlx-community/Qwen2.5-3B-Instruct-4bit"
HF_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"
# Groq models — check your tier limits at console.groq.com/settings/limits
# Free tier:       llama-3.1-8b-instant   6,000 TPM
# Dev/Enterprise:  llama-3.3-70b-versatile or llama3-8b-8192 with higher limits
GROQ_MODEL    = "llama-3.1-8b-instant"
CLAUDE_MODEL  = "claude-sonnet-4-6" #"claude-sonnet-4-5-20250929"

# vLLM — OpenAI-compatible local server on AMD MI300x (192 GB VRAM)
# Start server:  bash start_vllm.sh
# Models (≤10B, open-source):
#   Qwen/Qwen2.5-7B-Instruct           ~14 GB — recommended, best spatial reasoning
#   Qwen/Qwen2.5-3B-Instruct           ~6  GB — faster, slightly less accurate
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  ~14 GB — reasoning distil
#   mistralai/Mistral-7B-Instruct-v0.3 ~14 GB — good alternative
VLLM_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY  = "token-abc123"   # dummy key — vLLM accepts any non-empty string

# Module-level singleton — created once, reused for all 22k+ calls.
# Avoids recreating the HTTP connection pool on every call_llm() invocation.
_VLLM_CLIENT = None

def _get_vllm_client():
    global _VLLM_CLIENT
    if _VLLM_CLIENT is None:
        from openai import OpenAI
        _VLLM_CLIENT = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    return _VLLM_CLIENT


# ---------------------------------------------------------------------------
# FUNCTION 1  — parse the raw LLM text output into a clean action string
# ---------------------------------------------------------------------------
def parse_action_from_output(raw_output: str) -> Optional[str]:
    """
    WHAT IT DOES:
        Extracts the action word from the LLM's raw text output.
        LLMs don't always output just one word — they might say
        "I think the best move is LEFT." or "ACTION: down"

    HOW IT WORKS:
        1. Lowercase the raw output.
        2. Look for "ACTION: <word>" pattern first (chain-of-thought format).
        3. If not found, scan for the FIRST occurrence of any valid action word
           using regex word boundaries (\bup\b, \bdown\b, etc.).
        4. Return the matched action or None if nothing found.

    ARGS:
        raw_output: str — raw text from LLM

    RETURNS:
        str like "up", "down", "left", "right", or None if unparseable
    """
    # TODO: implement this (~15 lines)
    text = raw_output.lower().strip()

    # Strip <think>...</think> blocks from thinking models (qwen3, deepseek-r1)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Handle JSON format: {"action": "up"} or "action": "up"
    match = re.search(r'"action"\s*:\s*"(up|down|left|right)"', text)
    if match:
        return match.group(1)

    match = re.search(r'(?:action|answer):\s*(up|down|left|right)', text)

    if match:
        return match.group(1)
    

    ## fall back 

    for action in ["up","down", "left", "right"]:
        if re.search(r'\b' + action + r'\b', text):
            return action

    return None




# ---------------------------------------------------------------------------
# FUNCTION 2  — uniform API regardless of backend
# ---------------------------------------------------------------------------
def call_llm(prompt: str, model_name: str = OLLAMA_MODEL,
             backend: str = "ollama") -> str:
    """
    WHAT IT DOES:
        Sends a prompt to the LLM and returns the raw text response.
        This is the ONLY place that touches the actual LLM library.
        Everything else in this file calls call_llm().

    HOW IT WORKS (Ollama backend):
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

    HOW IT WORKS (HuggingFace backend):
        pipe = pipeline("text-generation", model=model_name, ...)
        output = pipe(prompt, max_new_tokens=50, temperature=0.1)
        return output[0]['generated_text']

    ARGS:
        prompt    : str — the full prompt
        model_name: str — model identifier
        backend   : "ollama" or "hf"

    RETURNS:
        str — raw LLM output text
    """
    # TODO: implement ONE of the backends'

    if backend == 'ollama':
        if ollama is None:
            raise RuntimeError("ollama package not installed. Run: pip install ollama")
        # Limit output tokens — we only need 1 word (up/down/left/right)
        # Set low temperature for more deterministic outputs
        opts = {"num_predict": 30, "temperature": 0.3}

        messages = [{"role": "user", "content": prompt}]

        # For qwen3 (thinking model): disable thinking mode to avoid
        # generating thousands of <think> tokens that make it 100x slower
        if "qwen3" in model_name.lower():
            opts["num_predict"] = 50  # a bit more room since no_think instruction
            messages = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt}
            ]
        # For deepseek-r1 (thinking model): similar issue
        elif "deepseek-r1" in model_name.lower():
            opts["num_predict"] = 50

        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=opts
        )
        return response['message']['content']

    if backend == 'mlx': 
        if not _MLX_AVAILABLE:
            raise RuntimeError(
                "mlx-lm not installed. Run: pip install mlx-lm\n"
                "Then the model is auto-downloaded on first use."
            )
        model, tokenizer = _get_mlx_model(model_name)
        # Apply chat template so the model runs in instruction-following mode,
        # not text completion mode — this makes it respect "reply with one word"
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = _mlx_generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=10,
            verbose=False
        )
        return response

    if backend == 'mlx_cached':
        if not _MLX_AVAILABLE:
            raise RuntimeError("mlx-lm not installed. Run: pip install mlx-lm")
        if not _MLX_CACHE_AVAILABLE:
            # Fall back to uncached MLX
            return call_llm(prompt, model_name=model_name, backend='mlx')
        model, tokenizer = _get_mlx_model(model_name)
        # Build prompt cache on first call, reuse on subsequent
        if model_name not in _MLX_PROMPT_CACHE:
            _MLX_PROMPT_CACHE[model_name] = _make_prompt_cache(model)
        prompt_cache = _MLX_PROMPT_CACHE[model_name]
        # Tokenize and generate with prompt cache
        tokens = mx.array(tokenizer.encode(prompt))
        output_tokens = []
        for (token, _logprobs), n in zip(
            _generate_step(tokens, model, max_tokens=15,
                           prompt_cache=prompt_cache),
            range(15)
        ):
            # token is already int in mlx-lm 0.31+
            tok_id = token if isinstance(token, int) else token.item()
            if tok_id == tokenizer.eos_token_id:
                break
            output_tokens.append(tok_id)
        return tokenizer.decode(output_tokens)

    if backend == 'hf':
        raise NotImplementedError('Please use ollama as backend')

    if backend == 'groq':
        import os, time
        from groq import Groq, RateLimitError
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                # Parse wait time from error message, default 5s
                wait_match = re.search(r'try again in ([\d.]+)s', str(e))
                wait = float(wait_match.group(1)) + 0.5 if wait_match else 5.0
                print(f"  [Groq rate limit] waiting {wait:.1f}s...", flush=True)
                time.sleep(wait)
        raise RuntimeError("Groq rate limit exceeded after 5 retries")

    if backend == 'vllm':
        # OpenAI-compatible client pointing at local vLLM server.
        # vLLM handles continuous batching server-side — just fire requests
        # concurrently and it groups them automatically.
        client = _get_vllm_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.1,
            logprobs=True,
            top_logprobs=5,
        )
        return response  # return full response object; caller extracts text or logprobs

    if backend == 'claude':
        import os
        from dotenv import load_dotenv
        load_dotenv()
        from anthropic import Anthropic
        api_key = os.getenv("RAKUTEN_AI_GATEWAY_KEY")
        if not api_key:
            raise ValueError("RAKUTEN_AI_GATEWAY_KEY not set in environment or .env")
        client = Anthropic(
            base_url="https://api.ai.public.rakuten-it.com/anthropic/",
            auth_token=api_key,
        )
        # Two-step approach:
        # Step 1: Let the model reason briefly about the board
        # Step 2: Force a final answer
        # This works better than fighting the model's tendency to reason
        response = client.messages.create(
            model=model_name,
            max_tokens=512,
            temperature=0.0,
            system=(
                "You are a Sokoban puzzle solver. Think step by step in 2-3 sentences MAX, "
                "then on the LAST line write EXACTLY: ACTION: <direction>\n"
                "where <direction> is one of: up, down, left, right.\n"
                "Keep analysis SHORT. The last line MUST be ACTION: followed by one direction."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Look for "ACTION: <direction>" pattern (last one in response)
        action_matches = re.findall(r'ACTION:\s*(up|down|left|right)', raw, re.IGNORECASE)
        if action_matches:
            return action_matches[-1].lower()
        # Fallback: find the LAST action word in the response (closest to the conclusion)
        all_actions = re.findall(r'\b(up|down|left|right)\b', raw.lower())
        if all_actions:
            return all_actions[-1]
        return raw


# ---------------------------------------------------------------------------
# CLASS — main interface used by search.py
# ---------------------------------------------------------------------------
class LLMPredictor:
    """
    WHAT IT IS:
        The main LLM wrapper class. Search algorithms import this and call
        predict_action() or predict_top_k() — they don't care which backend
        is used underneath.

    FIELDS:
        model_name : str
        backend    : str ("ollama" or "hf")
        repr_type  : str — which state representation to use (from representation.py)
        call_count : int — tracks total LLM calls made (for evaluation metrics)
    """

    def __init__(self, model_name: str = VLLM_MODEL,
                 backend: str = "vllm",
                 repr_type: str = REPR_ANNOTATED):
        """
        WHAT IT DOES:
            Initializes the predictor. For HuggingFace backend, loads the
            model here (slow, do once). For Ollama, nothing to load — the
            server is already running.

        TODO:
            - Store model_name, backend, repr_type as self attributes
            - Initialize self.call_count = 0
            - If backend == "hf": load the pipeline here
        """
        self.model_name = model_name
        self.backend = backend
        self.repr_type = repr_type
        self.call_count = 0
        self.cache_hits = 0
        self._cache_lock = __import__('threading').Lock()
        # Cache: state_key -> [(action, score), ...]
        # Prevents calling LLM twice for the same board configuration
        self._cache: dict = {}

    # -----------------------------------------------------------------------
    def predict_action(self, state: SokobanState,
                       use_reasoning: bool = False) -> tuple:
        """
        WHAT IT DOES:
            Asks the LLM for the single best next action given the state.

        HOW IT WORKS:
            1. Build prompt using build_prompt() or build_prompt_with_reasoning()
            2. Call call_llm() to get raw response
            3. Increment self.call_count
            4. Parse action using parse_action_from_output()
            5. Return (action, confidence)

        CONFIDENCE SCORE:
            - Ollama: no log-probs available by default, return 1.0 as placeholder
            - HuggingFace: you CAN get token log-probs — return softmax of the
              action token. See predict_top_k() for how to do this.
            - Simple alternative: call the LLM 3 times, return (most_common_action,
              count/3) — majority vote as confidence

        ARGS:
            state        : SokobanState
            use_reasoning: bool — use chain-of-thought prompt if True

        RETURNS:
            (action_str, confidence_float)
            action_str may be None if LLM output is unparseable
        """
        # TODO: implement this (~15 lines)
        
        if use_reasoning:
            prompt = build_prompt_with_reasoning(state, self.repr_type)
        else:
            prompt = build_prompt(state, self.repr_type)

        response = call_llm(prompt, model_name=self.model_name, backend=self.backend)
        with self._cache_lock:
            self.call_count += 1

        # vLLM returns a ChatCompletion object; other backends return plain text
        if hasattr(response, 'choices'):
            text = response.choices[0].message.content or ""
        else:
            text = response

        pred_action = parse_action_from_output(text)
        confidence = 1.0

        return (pred_action, confidence)

    # -----------------------------------------------------------------------
    def predict_top_k(self, state: SokobanState, k: int = 4,
                      n_samples: int = 3) -> list[tuple[str, float]]:
        """
        WHAT IT DOES:
            Returns up to k ranked (action, score) pairs.
            This is what BeamSearchSolver uses to decide which branches
            to keep.

        HOW IT WORKS — SAMPLING APPROACH (works with any backend):
            1. Call predict_action() n_samples times (with temperature > 0).
            2. Count how often each action appears.
            3. Sort by frequency descending.
            4. Return top-k as (action, frequency/n_samples) pairs.

        HOW IT WORKS — LOG-PROB APPROACH (HuggingFace only, more accurate):
            1. Build prompt.
            2. Tokenize the prompt.
            3. Run model.forward() to get logits over the next token.
            4. Find token IDs for " up", " down", " left", " right".
            5. Softmax those 4 logits → probabilities.
            6. Return sorted list of (action, prob).

        WHY n_samples=8:
            With sampling, you need enough samples to get a stable
            frequency distribution. 8 is fast and usually enough.

        ARGS:
            state    : SokobanState
            k        : max number of (action, score) pairs to return
            n_samples: number of LLM samples (for sampling approach)

        RETURNS:
            list of (action_str, score_float) sorted by score descending
            e.g. [("right", 0.625), ("down", 0.25), ("up", 0.125)]
        """
        from collections import Counter

        # Fast path: single-call ranked prediction using a dedicated prompt
        if n_samples == 1:
            return self._predict_top_k_single_call(state, k)

        # Multi-sample approach: call LLM n_samples times, count frequencies
        counts = Counter()
        for _ in range(n_samples):
            action, _ = self.predict_action(state)
            if action:
                counts[action] += 1

        total = sum(counts.values())
        if total == 0:
            valid = get_valid_actions(state)
            return [(a, 1.0/len(valid)) for a in valid[:k]]

        ranked = counts.most_common(k)
        return [(action, count/total) for action, count in ranked]

    def _predict_top_k_single_call(self, state: SokobanState, k: int) -> list[tuple[str, float]]:
        """Get ranked actions from a single LLM call with logprob scoring when available."""
        from representation import build_prompt
        from environment import state_key as _state_key
        import math

        # --- Cache check: same board = same prediction ---
        cache_key = _state_key(state)
        with self._cache_lock:
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key][:k]

        valid = get_valid_actions(state)
        prompt = build_prompt(state, self.repr_type)

        response = call_llm(prompt, model_name=self.model_name, backend=self.backend)
        with self._cache_lock:
            self.call_count += 1

        result = self._score_actions(response, valid)

        # Store in cache before returning
        with self._cache_lock:
            self._cache[cache_key] = result
        return result[:k]

    def _score_actions(self, response, valid: list) -> list[tuple[str, float]]:
        """Score actions from an LLM response using logprobs (vllm) or text-position fallback.

        For the vllm backend, `response` is the full ChatCompletion object which
        carries per-token logprobs.  We scan ALL output tokens' top_logprobs to
        find the token where an action word appears (e.g. token 5 in
        '{"action": "up"}' is "up").  This gives real probability distributions
        even when the model outputs JSON format.
        For every other backend, `response` is a plain string and we fall back to
        the text-position heuristic.
        """
        import math

        # ---- vLLM path: use real log-probabilities ----
        if hasattr(response, 'choices'):  # ChatCompletion object from openai client
            text = response.choices[0].message.content or ""

            lp_data = None
            try:
                lp_data = response.choices[0].logprobs
            except AttributeError:
                pass

            action_logprobs: dict[str, float] = {}
            if lp_data and lp_data.content:
                # Scan ALL output tokens — the action word may appear at any position
                # (e.g. token 5 in '{"action": "up"}' is "up")
                for token_lp in lp_data.content:
                    # Check if this token or its siblings are an action word
                    all_candidates = [token_lp] + list(token_lp.top_logprobs or [])
                    for candidate in all_candidates:
                        token_text = candidate.token.strip().lower()
                        for action in VALID_ACTIONS:
                            if token_text == action or token_text.startswith(action):
                                # Keep the best (highest) logprob seen for this action
                                if action not in action_logprobs or candidate.logprob > action_logprobs[action]:
                                    action_logprobs[action] = candidate.logprob

            if action_logprobs:
                valid_set = set(valid)
                scored = [(a, math.exp(lp)) for a, lp in action_logprobs.items() if a in valid_set]
                scored.sort(key=lambda x: x[1], reverse=True)
                seen = {a for a, _ in scored}
                for a in valid:
                    if a not in seen:
                        scored.append((a, 0.01))
                return scored

            # logprobs unavailable — fall back to parsing the text
            action = parse_action_from_output(text)
            return self._text_fallback_ranking(action, valid)

        # ---- Text-only path (ollama / groq / claude) ----
        text = response if isinstance(response, str) else str(response)
        action = parse_action_from_output(text)
        return self._text_fallback_ranking(action, valid)

    @staticmethod
    def _text_fallback_ranking(top_action, valid: list) -> list[tuple[str, float]]:
        """Fallback ranking when logprobs are not available.
        Puts the parsed top action first with score 0.9, rest uniform 0.1.
        """
        result = []
        valid_set = set(valid)
        if top_action and top_action in valid_set:
            result.append((top_action, 0.9))
        for a in valid:
            if a != top_action:
                result.append((a, 0.1))
        if not result:  # LLM returned nothing valid
            result = [(a, 1.0 / len(valid)) for a in valid]
        return result

    # -----------------------------------------------------------------------
    def predict_batch(self, states: list[SokobanState], k: int = 4) -> list[list[tuple[str, float]]]:
        """
        Predict ranked actions for a list of states.
        Each state uses build_prompt via _predict_top_k_single_call (with caching).
        """
        return [self._predict_top_k_single_call(state, k) for state in states]

    def predict_batch_states(self, states: list[SokobanState],
                             k: int = 4) -> list[list[tuple[str, float]]]:
        """Batch-predict ranked actions for a list of states using concurrent HTTP requests.

        For the vllm backend, all requests are fired concurrently via a
        ThreadPoolExecutor.  The vLLM server receives them all and processes
        them in a single continuous batch — wall-clock time is roughly equal
        to one serial call regardless of batch size.

        For other backends (ollama, groq, etc.) this still issues concurrent
        requests which helps when the backend supports it.

        Cache-aware: states already in self._cache are served immediately
        without a network call.

        Args:
            states: list of SokobanState to predict for
            k:      max actions to return per state

        Returns:
            list of (action, score) lists, one per state, in the same order
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from environment import state_key as _state_key
        from representation import build_prompt

        if not states:
            return []

        # Split into cached vs. needs-inference
        results: list[list[tuple[str, float]] | None] = [None] * len(states)
        pending: list[tuple[int, SokobanState, str]] = []  # (index, state, prompt)

        for i, state in enumerate(states):
            cache_key = _state_key(state)
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                with self._cache_lock:
                    self.cache_hits += 1
                results[i] = cached[:k]
            else:
                prompt = build_prompt(state, self.repr_type)
                pending.append((i, state, prompt))

        if not pending:
            return results  # type: ignore[return-value]

        # Fire all pending requests concurrently
        def _call_one(args):
            idx, state, prompt = args
            response = call_llm(prompt, model_name=self.model_name,
                                backend=self.backend)
            scored = self._score_actions(response, get_valid_actions(state))
            cache_key = _state_key(state)
            with self._cache_lock:
                self.call_count += 1
                self._cache[cache_key] = scored
            return idx, scored

        max_workers = min(len(pending), 64)  # cap; vLLM queues excess requests anyway
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, scored in ex.map(_call_one, pending):
                results[idx] = scored[:k]

        return results  # type: ignore[return-value]

    # -----------------------------------------------------------------------
    def reset_call_count(self):
        """Resets the LLM call counter and cache. Call before each puzzle evaluation."""
        with self._cache_lock:
            self.call_count = 0
            self.cache_hits = 0
            self._cache.clear()


# ---------------------------------------------------------------------------
# QUICK SELF-TEST  — run: python llm_predictor.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    filepath = Path(__file__).parent / "data" / "Microban.txt"
    all_puzzles = load_and_parse_all(str(filepath))
    state = from_parsed(all_puzzles[0])

    predictor = LLMPredictor(model_name=OLLAMA_MODEL, backend="ollama",
                             repr_type=REPR_ASCII)

    print("=== Testing predict_action ===")
    action, conf = predictor.predict_action(state)
    print(f"Predicted action : {action}")
    print(f"Confidence       : {conf:.3f}")
    print(f"Valid actions    : {get_valid_actions(state)}")
    print(f"LLM calls so far : {predictor.call_count}")

    print()
    print("=== Testing predict_top_k ===")
    top_k = predictor.predict_top_k(state, k=4)
    print("Ranked actions:")
    for act, score in top_k:
        print(f"  {act:6s}  {score:.3f}")
