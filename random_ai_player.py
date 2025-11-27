"""
Random AI Player for 65536 Game

This module implements a simple AI that randomly selects from valid moves.
It serves as:
- A baseline for comparing more sophisticated AI implementations
- A demonstration of how to properly use the Player interface
- A reference implementation for AI developers
"""

import numpy as np
from player_interface import Player
from game_core import GameState, Direction, MoveResult, Game


class RandomAIPlayer(Player):
    """
    AI player that randomly selects from valid moves.
    
    This is the simplest possible AI - it makes no attempt to evaluate
    positions or plan ahead. Despite its simplicity, it can sometimes
    achieve the 2048 tile in classic mode.
    
    Purpose:
        - Baseline for performance comparisons
        - Demonstrates Player interface usage
        - Validates game implementation
    
    Performance Expectations (Classic Mode):
        - Average score: ~5,000-15,000
        - Max tile: Usually 512-1024
        - Occasionally reaches 2048
        - Games per second: Very fast (1000+/sec)
    
    Usage:
        ```python
        game = Game(spawn_mode=SpawnMode.CLASSIC, seed=42)
        player = RandomAIPlayer(game)
        final_state = run_game(game, player)
        ```
    """
    
    def __init__(self, game: Game, verbose: bool = False):
        """
        Initialize random AI player.
        
        Args:
            game: Game instance to play
            verbose: If True, print move details
        """
        self.game = game
        self.verbose = verbose
        self.rng = np.random.default_rng()
        self.move_count = 0
    
    def get_move(self, state: GameState) -> Direction:
        """
        Choose a random valid move.
        
        Strategy:
            1. Get list of all valid moves
            2. Randomly select one
            3. Return it
        
        Args:
            state: Current game state
            
        Returns:
            Randomly chosen valid direction
            
        Raises:
            ValueError: If no valid moves available (should never happen
                       since game calls this only when moves exist)
        """
        # Get all valid moves
        valid_moves = self.game.move_engine.get_valid_moves(state)
        
        if not valid_moves:
            raise ValueError("No valid moves available - this should never happen!")
        
        # Choose randomly
        chosen_direction = valid_moves[self.rng.integers(0, len(valid_moves))]
        
        if self.verbose:
            print(f"Move {self.move_count + 1}: Choosing {chosen_direction.name} "
                  f"from {len(valid_moves)} valid moves")
        
        return chosen_direction
    
    def on_move_result(self, result: MoveResult) -> None:
        """
        Handle move result (optional logging).
        
        Args:
            result: Details about the move outcome
        """
        self.move_count += 1
        
        if self.verbose and result.success:
            print(f"  Score gained: {result.score_gained}")
            if result.spawned_tile:
                row, col, value = result.spawned_tile
                print(f"  Spawned {value} at ({row}, {col})")
            print(f"  Current score: {result.state.score}")
            print(f"  Max tile: {result.state.get_max_tile()}")
            print()
    
    def on_game_over(self, state: GameState) -> None:
        """
        Print final statistics when game ends.
        
        Args:
            state: Final game state
        """
        if self.verbose:
            print("=" * 50)
            print("GAME OVER")
            print("=" * 50)
        
        if self.verbose:
            print(f"Final Score: {state.score:,}")
            print(f"Max Tile: {state.get_max_tile()}")
            print(f"Total Moves: {state.move_count}")
        
        if state.won:
            print("🎉 YOU WON! Reached 65536!")
        
        if self.verbose:
            print("=" * 50)
    
    def reset(self) -> None:
        """
        Reset for a new game.
        """
        self.move_count = 0


# ========== Performance Notes ==========

"""
RandomAI Performance Characteristics:

Classic Mode (90% 2, 10% 4):
    - Very fast execution (no computation needed)
    - Highly variable outcomes (purely random)
    - Useful for stress-testing game implementation
    - Baseline for AI comparisons

Dynamic Mode (1/16th rule):
    - Similar performance to classic mode early game
    - May perform slightly worse late game due to more spawn values
    - Still very fast

Statistical Expectations:
    To get reliable statistics, run 100+ games:
    ```python
    from main import run_benchmark
    
    stats = run_benchmark(
        game=Game(spawn_mode=SpawnMode.CLASSIC, seed=42),
        player_factory=lambda: RandomAIPlayer(game, verbose=False),
        num_games=1000
    )
    
    print(f"Average score: {stats['avg_score']:.0f}")
    print(f"Max tile distribution: {stats['max_tile_distribution']}")
    print(f"Games reaching 2048: {stats['games_reaching_2048']}")
    ```
"""
