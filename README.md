# 65536 Game with AI-Ready Architecture

A clean, well-documented implementation of the 65536 game (similar to 2048) designed specifically for AI development and tree search algorithms.

## Features

- **Two spawning modes:**
  - **Classic Mode**: 2048-style (90% 2, 10% 4)
  - **Dynamic Mode**: 65536-style with 1/16th rule (uniform distribution)
- **AI-optimized design:**
  - Fast state hashing for transposition tables
  - Efficient successor enumeration for Expectimax
  - Comprehensive evaluation heuristics
  - Exposed probability distributions
- **Clean architecture:**
  - Game logic completely separate from player implementation
  - Interface-based player system
  - Immutable game states for tree search
- **Thoroughly documented:**
  - Every class and method has detailed docstrings
  - Mathematical formulas for evaluation functions
  - Usage examples throughout

## Quick Start

### Installation

```bash
# Clone or download this repository
cd 65536-game

# Install dependencies
pip install -r requirements.txt
```

### Run a Game

```python
from game_core import Game, SpawnMode
from random_ai_player import RandomAIPlayer
from main import run_game

# Create game in Classic mode
game = Game(spawn_mode=SpawnMode.CLASSIC, seed=42)

# Create RandomAI player
player = RandomAIPlayer(game, verbose=True)

# Run game
final_state = run_game(game, player, verbose=True)

print(f"Final score: {final_state.score}")
print(f"Max tile: {final_state.get_max_tile()}")
```

### Run Benchmark

```python
from game_core import Game, SpawnMode
from random_ai_player import RandomAIPlayer
from main import run_benchmark, print_benchmark_results

game = Game(spawn_mode=SpawnMode.CLASSIC, seed=42)

stats = run_benchmark(
    game=game,
    player_factory=lambda: RandomAIPlayer(game, verbose=False),
    num_games=100,
    verbose_every=20
)

print_benchmark_results(stats)
```

## Game Modes

### Classic Mode (2048-style)

- Spawns tile value 2 with 90% probability
- Spawns tile value 4 with 10% probability
- Fixed branching factor for AI search

```python
game = Game(spawn_mode=SpawnMode.CLASSIC)
```

### Dynamic Mode (65536 1/16th Rule)

- Max spawn value = Current highest tile ÷ 16
- Uniform distribution among all valid powers of 2
- Growing branching factor as game progresses

**Example progression:**
- Max tile = 256 → Spawns {2, 4, 8, 16} each @ 25%
- Max tile = 1024 → Spawns {2, 4, 8, 16, 32, 64} each @ 16.67%
- Max tile = 65536 → Spawns {2, 4, ..., 4096} (12 values!) each @ 8.33%

```python
game = Game(spawn_mode=SpawnMode.DYNAMIC)
```

## Player API Reference

All players (AI or human) must implement the `Player` interface from `player_interface.py`.

### Required Methods

#### `get_move(state: GameState) -> Direction`

Choose a move for the given game state.

**Called:** Before each move
**Returns:** Direction.UP, Direction.DOWN, Direction.LEFT, or Direction.RIGHT

```python
def get_move(self, state: GameState) -> Direction:
    # Option 1: Get valid moves
    valid_moves = self.game.move_engine.get_valid_moves(state)
    
    # Option 2: Use evaluation heuristics
    best_score = float('-inf')
    best_move = None
    for direction in valid_moves:
        new_state, _ = self.game.move_engine.execute_move(state, direction)
        score = new_state.get_position_weights()
        if score > best_score:
            best_score = score
            best_move = direction
    
    return best_move
```

#### `on_move_result(result: MoveResult) -> None`

Callback after each successful move.

**Called:** After every move and tile spawn
**Use for:** Logging, learning, statistics

```python
def on_move_result(self, result: MoveResult) -> None:
    print(f"Score gained: {result.score_gained}")
    if result.spawned_tile:
        r, c, v = result.spawned_tile
        print(f"Spawned {v} at ({r}, {c})")
```

#### `on_game_over(state: GameState) -> None`

Callback when game ends.

**Called:** Once at game end
**Use for:** Final statistics, saving results

```python
def on_game_over(self, state: GameState) -> None:
    print(f"Final score: {state.score}")
    print(f"Max tile: {state.get_max_tile()}")
    print(f"Moves: {state.move_count}")
```

#### `reset() -> None`

Reset player state for a new game.

**Called:** Before each new game
**Use for:** Clearing history, resetting internal state

```python
def reset(self) -> None:
    self.move_history = []
```

## Implementing Custom AI

### Example 1: Heuristic AI

See `examples/heuristic_ai.py` for a simple greedy AI that evaluates each move with a weighted heuristic.

### Example 2: Expectimax AI (Depth 2)

See `examples/expectimax_ai.py` for a basic tree search implementation.

Both examples are intentionally simple to demonstrate the core concepts.

## Evaluation Helpers

AI heuristics are provided in the `evaluator.py` module as pure functions:

| Function | Description | Interpretation |
|----------|-------------|----------------|
| `get_smoothness(state)` | Sum of differences between adjacent tiles | Lower = smoother board |
| `get_monotonicity(state)` | Measure of monotonic rows/columns | Higher = better organization |
| `get_merge_potential(state)` | Count of possible merges | Higher = more opportunities |
| `get_corner_weight(state)` | Weighted sum favoring corners | Higher = high tiles in corners |
| `get_gradient_score(state)` | Weighted sum with snake pattern | Higher = good tile placement |
| `count_distinct_tiles(state)` | Number of different tile values | Lower = more consolidated |
| `is_symmetric(state)` | Check for board symmetry | For search optimization |

**Usage:**
```python
import evaluator
from game_core import GameState

score = (
    evaluator.get_gradient_score(state) * 1.0 +
    evaluator.get_monotonicity(state) * 1.0 -
    evaluator.get_smoothness(state) * 0.1
)
```

**Convenience function:**
```python
# Use with default weights
score = evaluator.evaluate_state(state)

# Or customize weights
score = evaluator.evaluate_state(state, {
    'gradient': 1.0,
    'monotonicity': 1.0,
    'smoothness': -0.1,  # negative because lower is better
    'empty_cells': 100.0,
    'merge_potential': 10.0
})
```

See `evaluator.py` for detailed mathematical definitions and examples.

## Tree Search Helpers

For advanced AI algorithms:

### Get All Successors (for Expectimax)

```python
# Get all (direction, state, probability) tuples
successors = game.enumerate_all_successors(state)

for direction, resulting_state, prob in successors:
    print(f"{direction.name}: Probability {prob:.4f}")
```

### Get Move Outcomes (for MCTS)

```python
# Get states after moves (before random spawn)
outcomes = game.get_deterministic_move_states(state)

for direction, after_move_state in outcomes.items():
    print(f"{direction.name}: {after_move_state.score}")
```

### Simulate Random Game (for Rollouts)

```python
# Play randomly until game over
final_state = game.simulate_random_game(state, max_moves=1000)
print(f"Rollout result: {final_state.score}")
```

## File Structure

```
65536-game/
├── game_core.py           # Core game logic (game mechanics only)
├── evaluator.py           # AI evaluation functions (heuristics)
├── player_interface.py    # Abstract Player class
├── random_ai_player.py    # Baseline Random AI
├── human_player.py        # Tkinter GUI for human play
├── main.py                # Game runner & benchmarking
├── tournament.py          # Tournament runner & plotting
├── run_ai.py              # Universal AI runner script
├── agents/                # Advanced AI Implementations
│   ├── stochastic_ai.py          # Standard Expectimax
│   ├── stochastic_gradient_ai.py # Expectimax + Gradient Heuristic
│   ├── caching_ai.py             # Transposition Table Optimization
│   ├── dynamic_depth_ai.py       # Variable Depth Search
│   ├── dynamic_pruning_ai.py     # Variable Pruning Thresholds
│   └── dynamic_hybrid_ai.py      # Combined Dynamic Strategy
├── examples/              # Simple Example AIs
│   ├── heuristic_ai.py    # Simple greedy AI
│   └── expectimax_ai.py   # Basic tree search AI
├── requirements.txt       # Dependencies (numpy, matplotlib)
└── README.md              # This file
```

## Performance Notes

**RandomAI Baseline (Classic Mode, 100 games):**
- Average score: ~2,000-3,000
- Max tile: Usually 128-256, occasionally 512-1024
- Very fast execution (negligible computation)

**State Operations:**
- State hashing: Fast (cached)
- Move execution: Fast (fixed 4x4 grid)
- Successor enumeration: Fast in classic mode, grows with max tile in dynamic mode

## Testing

Run the included benchmark to test your AI:

```bash
python -m main
```

This will run 100 games with RandomAI and print statistics.

## Advanced AI Agents

The `agents/` directory contains a progression of sophisticated AI implementations:

1.  **`StochasticAI`**: Standard Expectimax with probability-based pruning.
2.  **`StochasticGradientAI`**: Adds a highly optimized "Gradient" heuristic (Snake pattern) for better tile organization.
3.  **`CachingAI`**: Adds a Transposition Table (Zobrist hashing) to cache board states. This provides a **~2x speedup**, allowing for deeper search (Depth 4) in the same time.
4.  **`DynamicDepthAI`**: Adjusts search depth (3-6) based on board emptiness. Fast in early game, deep in late game.
5.  **`DynamicHybridAI`**: The current SOTA. Combines:
    - **Gradient Heuristic** (Organization)
    - **Transposition Table** (Speed)
    - **Dynamic Depth** (3-6 based on complexity)
    - **Dynamic Pruning** (Variable risk tolerance)

## Tournament & Benchmarking

Use `tournament.py` to compare agents head-to-head.

```bash
# Run a 5-game tournament
python tournament.py --games 5

# Run in Dynamic Mode (1/16th rule)
python tournament.py --games 10 --mode dynamic
```

**Features:**
- Runs games in parallel (conceptually)
- Tracks Win Rate (2048+ and 65536+)
- Generates a **Score vs Time** plot (`benchmark_results.png`) to visualize the efficiency trade-offs.

## Future Development

Ideas for extending this codebase:

1. **More sophisticated AI:**
   - Implement full Expectimax with alpha-beta pruning
   - Monte Carlo Tree Search (MCTS)
   - Deep learning with neural networks

2. **Human player:**
   - Web-based UI (HTML/CSS/JavaScript)
   - Keyboard controls
   - Visual animations

3. **Additional features:**
   - Save/load game state
   - Replay system
   - Different grid sizes (5x5, 6x6)
   - Custom spawn probabilities

4. **Analysis tools:**
   - Visualization of AI search tree
   - Position database
   - Opening book

## License

Free to use for educational and research purposes.

## Contributing

This is a complete, self-contained implementation. Feel free to extend it for your own AI experiments!
