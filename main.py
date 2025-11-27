"""
Main Game Runner and Benchmarking Utilities

This module provides functions to:
- Run complete games with any player
- Display game state in terminal
- Benchmark AI performance over multiple games
- Collect statistics for analysis
"""

import numpy as np
from typing import Callable, Dict
from game_core import Game, GameState, Direction, SpawnMode
from player_interface import Player


def print_grid(grid: np.ndarray) -> None:
    """
    Pretty print a game grid to the terminal.
    
    Args:
        grid: 4x4 numpy array of tile values
        
    Example output:
            2     4     8    16
           32    64   128   256
          512  1024     .     .
            .     .     .     .
    """
    print()
    for row in grid:
        print(" ".join(f"{int(val):5d}" if val != 0 else "    ." for val in row))
    print()


def run_game(game: Game, player: Player, verbose: bool = False) -> GameState:
    """
    Run a complete game with the given player.
    
    This is the main game loop that orchestrates interaction between
    the game engine and the player.
    
    Args:
        game: Game instance
        player: Player implementation (AI or human)
        verbose: If True, print game state after each move
        
    Returns:
        Final game state
        
    Process:
        1. Reset game and player
        2. Loop until game over:
            a. Get move from player
            b. Execute move
            c. Notify player of result
            d. Display state if verbose
        3. Notify player of game over
        4. Return final state
    
    Example:
        ```python
        from game_core import Game, SpawnMode
        from random_ai_player import RandomAIPlayer
        
        game = Game(spawn_mode=SpawnMode.CLASSIC, seed=42)
        player = RandomAIPlayer(game, verbose=False)
        
        final_state = run_game(game, player, verbose=True)
        print(f"Final score: {final_state.score}")
        ```
    """
    # Reset player for new game
    player.reset()
    
    # Initialize game
    state = game.reset()
    
    if verbose:
        print("=" * 50)
        print("NEW GAME STARTED")
        print("=" * 50)
        print_grid(state.grid)
    
    # Main game loop
    while not game.is_game_over(state):
        try:
            # Get move from player
            direction = player.get_move(state)
            
            # Execute move
            new_state, result = game.make_move(state, direction)
            
            # Handle illegal moves (should not happen with proper AI)
            if not result.success:
                if verbose:
                    print(f"⚠️  Warning: Illegal move attempted: {direction.name}")
                    print("   Board unchanged, trying again...")
                continue
            
            # Update state
            state = new_state
            
            # Notify player
            player.on_move_result(result)
            
            # Display if verbose
            if verbose:
                print(f"Move {state.move_count}: {direction.name}")
                print(f"Score: {state.score:,} (+{result.score_gained})")
                print_grid(state.grid)
                
                if state.won and not state.game_over:
                    print("🎉 " * 10)
                    print("    YOU REACHED 65536!")
                    print("🎉 " * 10)
                    print()
        
        except Exception as e:
            print(f"❌ Player error: {e}")
            break
    
    # Game over
    player.on_game_over(state)
    
    return state


def run_benchmark(game: Game, player_factory: Callable[[], Player], 
                  num_games: int = 100, verbose_every: int = 0) -> Dict:
    """
    Run multiple games and collect statistics.
    
    This function is useful for:
    - Evaluating AI performance
    - Comparing different AI strategies
    - Testing game implementation
    - Gathering data for analysis
    
    Args:
        game: Game instance (will be reset for each game)
        player_factory: Function that returns a new Player instance
                       Example: lambda: RandomAIPlayer(game)
        num_games: Number of games to run
        verbose_every: Print progress every N games (0 = no progress)
        
    Returns:
        Dictionary with statistics:
            - avg_score: Average final score
            - std_score: Standard deviation of scores
            - max_score: Highest score achieved
            - min_score: Lowest score achieved
            - avg_moves: Average number of moves per game
            - max_tile_distribution: Dict mapping tile value to count
            - games_reaching_2048: Number of games that reached 2048
            - games_reaching_4096: Number of games that reached 4096
            - games_reaching_8192: Number of games that reached 8192
            - games_reaching_16384: Number of games that reached 16384
            - games_reaching_32768: Number of games that reached 32768
            - games_reaching_65536: Number of games that reached 65536 (won)
    
    Example:
        ```python
        game = Game(spawn_mode=SpawnMode.CLASSIC, seed=42)
        
        stats = run_benchmark(
            game=game,
            player_factory=lambda: RandomAIPlayer(game, verbose=False),
            num_games=1000,
            verbose_every=100
        )
        
        print(f"Average score: {stats['avg_score']:.0f} ± {stats['std_score']:.0f}")
        print(f"Best game: {stats['max_score']}")
        print(f"Max tile distribution:")
        for tile, count in sorted(stats['max_tile_distribution'].items()):
            print(f"  {tile}: {count} games ({count/num_games*100:.1f}%)")
        ```
    """
    if verbose_every > 0:
        print(f"Running {num_games} games...")
        print()
    
    scores = []
    max_tiles = []
    move_counts = []
    
    for i in range(num_games):
        # Create new player for this game
        player = player_factory()
        
        # Run game
        final_state = run_game(game, player, verbose=False)
        
        # Collect statistics
        scores.append(final_state.score)
        max_tiles.append(final_state.get_max_tile())
        move_counts.append(final_state.move_count)
        
        # Progress update
        if verbose_every > 0 and (i + 1) % verbose_every == 0:
            print(f"Completed {i + 1}/{num_games} games")
            print(f"  Current avg score: {np.mean(scores):.0f}")
            print(f"  Current best: {np.max(scores)}")
            print()
    
    # Calculate statistics
    scores_array = np.array(scores)
    max_tiles_array = np.array(max_tiles)
    
    # Count games reaching each milestone
    milestones = {
        'games_reaching_2048': np.sum(max_tiles_array >= 2048),
        'games_reaching_4096': np.sum(max_tiles_array >= 4096),
        'games_reaching_8192': np.sum(max_tiles_array >= 8192),
        'games_reaching_16384': np.sum(max_tiles_array >= 16384),
        'games_reaching_32768': np.sum(max_tiles_array >= 32768),
        'games_reaching_65536': np.sum(max_tiles_array >= 65536),
    }
    
    # Max tile distribution
    unique_tiles, counts = np.unique(max_tiles_array, return_counts=True)
    max_tile_dist = {int(tile): int(count) for tile, count in zip(unique_tiles, counts)}
    
    results = {
        'num_games': num_games,
        'avg_score': float(np.mean(scores_array)),
        'std_score': float(np.std(scores_array)),
        'max_score': int(np.max(scores_array)),
        'min_score': int(np.min(scores_array)),
        'avg_moves': float(np.mean(move_counts)),
        'max_tile_distribution': max_tile_dist,
        **milestones
    }
    
    return results


def print_benchmark_results(stats: Dict) -> None:
    """
    Pretty print benchmark statistics.
    
    Args:
        stats: Dictionary returned by run_benchmark()
    """
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print()
    
    print(f"Games Played: {stats['num_games']}")
    print()
    
    print("Score Statistics:")
    print(f"  Average: {stats['avg_score']:,.0f} ± {stats['std_score']:.0f}")
    print(f"  Best:    {stats['max_score']:,}")
    print(f"  Worst:   {stats['min_score']:,}")
    print()
    
    print(f"Average Moves: {stats['avg_moves']:.1f}")
    print()
    
    print("Milestone Achievements:")
    milestones = [2048, 4096, 8192, 16384, 32768, 65536]
    for tile in milestones:
        key = f'games_reaching_{tile}'
        count = stats.get(key, 0)
        pct = count / stats['num_games'] * 100
        print(f"  {tile:>6}: {count:>4} games ({pct:>5.1f}%)")
    print()
    
    print("Max Tile Distribution:")
    for tile, count in sorted(stats['max_tile_distribution'].items(), reverse=True):
        pct = count / stats['num_games'] * 100
        bar = '█' * int(pct / 2)  # Simple bar chart
        print(f"  {tile:>6}: {count:>4} games ({pct:>5.1f}%) {bar}")
    
    print("=" * 60)


if __name__ == "__main__":
    """
    Example usage: Run RandomAI benchmark
    """
    from random_ai_player import RandomAIPlayer
    
    print("Running RandomAI benchmark on Classic mode...")
    print()
    
    # Create game
    game = Game(spawn_mode=SpawnMode.CLASSIC, seed=42)
    
    # Run benchmark
    stats = run_benchmark(
        game=game,
        player_factory=lambda: RandomAIPlayer(game, verbose=False),
        num_games=100,
        verbose_every=20
    )
    
    # Print results
    print_benchmark_results(stats)
    
    print()
    print("To run with Dynamic mode:")
    print("  game = Game(spawn_mode=SpawnMode.DYNAMIC, seed=42)")
